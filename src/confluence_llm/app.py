# src/confluence_llm/app.py
from __future__ import annotations

import logging

import streamlit as st  # imports must be at the very top (fixes Ruff E402)

# Streamlit page config should be the first Streamlit call
st.set_page_config(page_title="Confluence LLM — Stage 0", layout="wide")

logger = logging.getLogger(__name__)


def render_app() -> None:
    st.title("Confluence LLM — Stage 0")
    st.write("✅ Project bootstrap is ready. Next step: wire Confluence ingestion (Stage 1).")

    with st.expander("Environment placeholders"):
        st.code(
            "CONFLUENCE_BASE_URL=https://your-domain.atlassian.net/wiki\n"
            "CONFLUENCE_EMAIL=you@example.com\n"
            "CONFLUENCE_API_TOKEN=atlassian_api_token\n",
            language="bash",
        )

    # Adjust the run command to your actual file path/name
    st.info(
        "Use Docker or run `pip install -r requirements.txt && "
        "streamlit run src/confluence_llm/app.py`."
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    logger.info("Confluence LLM app starting…")
    render_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
