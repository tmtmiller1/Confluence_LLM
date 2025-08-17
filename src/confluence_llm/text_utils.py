from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import re

# Utilities for HTML → Markdown and heading-based chunking.


_HEADING_BLOCK_RX = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)


def heading_chunks(md: str) -> list[str]:
    """Split Markdown into chunks by headings; keep each heading with its body."""
    if not md.strip():
        return []
    chunks: list[str] = []
    last = 0
    for match in _HEADING_BLOCK_RX.finditer(md):
        start = match.start()
        if start > last:
            block = md[last:start].strip()
            if block:
                chunks.append(block)
        last = start
    tail = md[last:].strip()
    if tail:
        chunks.append(tail)
    return chunks


class _MDHTMLParser(HTMLParser):
    """
    Minimal-but-capable HTML→Markdown converter (stdlib only).

    Handles:
      - <h1..h6>, <p>, <br>, <hr>
      - <strong>/<b>, <em>/<i>, <u>
      - <code>, <pre>
      - <blockquote>
      - <ul>/<ol>/<li> (nested)
      - <a href>, <img src alt>
      - unknown tags are stripped but text is preserved
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: list[str] = []
        self._stack: list[str] = []
        self._list_stack: list[tuple[str, int]] = []  # ("ul"|"ol", next index for ol)
        self._bold = 0
        self._italic = 0
        self._underline = 0
        self._code_inline = 0
        self._in_pre = 0
        self._in_blockquote = 0
        self._pending_space = False
        self._just_started_line = True
        self._link_stack: list[str] = []

    # ---- writers ---------------------------------------------------------

    def _write(self, text: str) -> None:
        if not text:
            return
        if self._pending_space and not text.startswith(("\n", " ", "\t")):
            self.out.append(" ")
        self.out.append(text)
        self._pending_space = False
        self._just_started_line = text.endswith("\n")

    def _nl(self, count: int = 1) -> None:
        count = 1 if count <= 1 else 2
        self.out.append("\n" * count)
        self._pending_space = False
        self._just_started_line = True

    def _start_line_prefix(self) -> None:
        prefix_parts: list[str] = []
        if self._in_blockquote > 0:
            prefix_parts.append("> " * self._in_blockquote)
        if self._list_stack and self._stack and self._stack[-1] == "li":
            list_type, idx = self._list_stack[-1]
            indent = "  " * (len(self._list_stack) - 1)
            if list_type == "ul":
                prefix_parts.append(f"{indent}- ")
            else:
                prefix_parts.append(f"{indent}{idx}. ")
        if prefix_parts:
            self._write("".join(prefix_parts))

    # ---- stack helpers ---------------------------------------------------

    def _push(self, tag: str) -> None:
        self._stack.append(tag)

    def _pop(self, tag: str) -> None:
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
            return
        while self._stack and self._stack[-1] != tag:
            self._stack.pop()
        if self._stack:
            self._stack.pop()

    # ---- tag handlers ----------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrd = {k.lower(): (v or "") for k, v in attrs}

        if tag == "br":
            self._nl(1)
            return
        if tag == "hr":
            self._nl(2)
            self._write("---")
            self._nl(2)
            return
        if tag == "p":
            self._nl(2)
            self._push(tag)
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._nl(2)
            self._push(tag)
            level = int(tag[1])
            self._write("#" * level + " ")
            return
        if tag in {"strong", "b"}:
            self._bold += 1
            self._write("**")
            self._push(tag)
            return
        if tag in {"em", "i"}:
            self._italic += 1
            self._write("*")
            self._push(tag)
            return
        if tag == "u":
            self._underline += 1
            self._write("__")
            self._push(tag)
            return
        if tag == "code" and not self._in_pre:
            self._code_inline += 1
            self._write("`")
            self._push(tag)
            return
        if tag == "pre":
            self._in_pre += 1
            self._nl(2)
            self._write("```")
            self._nl(1)
            self._push(tag)
            return
        if tag == "blockquote":
            self._in_blockquote += 1
            self._nl(1)
            self._push(tag)
            return
        if tag == "ul":
            self._list_stack.append(("ul", 0))
            self._nl(1)
            self._push(tag)
            return
        if tag == "ol":
            self._list_stack.append(("ol", 0))
            self._nl(1)
            self._push(tag)
            return
        if tag == "li":
            if self._list_stack:
                typ, idx = self._list_stack.pop()
                if typ == "ol":
                    self._list_stack.append((typ, idx + 1))
                else:
                    self._list_stack.append((typ, idx))
            self._nl(1)
            self._push(tag)
            self._start_line_prefix()
            return
        if tag == "a":
            href = attrd.get("href", "")
            self._link_stack.append(href)
            self._push(tag)
            return
        if tag == "img":
            src = attrd.get("src", "")
            alt = attrd.get("alt", "")
            if src:
                alt_text = alt.replace("]", "\\]")
                self._write(f"![{alt_text}]({src})")
            return

        self._push(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "p":
            self._pop(tag)
            self._nl(2)
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._pop(tag)
            self._nl(2)
            return
        if tag in {"strong", "b"}:
            if self._bold > 0:
                self._write("**")
                self._bold -= 1
            self._pop(tag)
            return
        if tag in {"em", "i"}:
            if self._italic > 0:
                self._write("*")
                self._italic -= 1
            self._pop(tag)
            return
        if tag == "u":
            if self._underline > 0:
                self._write("__")
                self._underline -= 1
            self._pop(tag)
            return
        if tag == "code" and not self._in_pre:
            if self._code_inline > 0:
                self._write("`")
                self._code_inline -= 1
            self._pop(tag)
            return
        if tag == "pre":
            if self._in_pre > 0:
                self._nl(1)
                self._write("```")
                self._nl(2)
                self._in_pre -= 1
            self._pop(tag)
            return
        if tag == "blockquote":
            if self._in_blockquote > 0:
                self._in_blockquote -= 1
            self._pop(tag)
            self._nl(1)
            return
        if tag in {"ul", "ol"}:
            if self._list_stack:
                self._list_stack.pop()
            self._pop(tag)
            self._nl(1)
            return
        if tag == "li":
            self._pop(tag)
            return
        if tag == "a":
            href = self._link_stack.pop() if self._link_stack else ""
            if href:
                self._write(f" ({href})")
            self._pop(tag)
            return

        self._pop(tag)

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self._in_pre > 0:
            self._write(data)
            return
        text = unescape(data)
        text = re.sub(r"\s+", " ", text)
        if text:
            self._write(text)
            self._pending_space = True

    def handle_entityref(self, name: str) -> None:
        self._write(unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self._write(unescape(f"&#{name};"))


def _post_cleanup(md: str) -> str:
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def html_to_markdown(html: str) -> str:
    """Convert HTML string to Markdown using a tolerant stdlib parser."""
    if not html:
        return ""
    parser = _MDHTMLParser()
    parser.feed(html)
    parser.close()
    return _post_cleanup("".join(parser.out))
