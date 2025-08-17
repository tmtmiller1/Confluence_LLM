# src/confluence_llm/app.py
from __future__ import annotations

import logging

from confluence_client import ConfluenceClient
from confluence_views import (
    normalize_pages,
    page_counts_by_space,
    page_counts_by_user,
)
import pandas as pd  # noqa: F401  (imported for future dataframe ops/typing)
from response_generator import summarize
from settings import (
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_BASE_URL,
    CONFLUENCE_EMAIL,
)
import streamlit as st  # imports must be at the very top (fixes Ruff E402)

# Streamlit page config should be the first Streamlit call
st.set_page_config(page_title="Confluence LLM — Stage 1", layout="wide")

logger = logging.getLogger(__name__)


def render_app() -> None:
    st.title("Confluence LLM — Stage 1: Core Confluence Ingestion")

    with st.sidebar:
        st.header("Confluence Settings")
        base_url = st.text_input(
            "Base URL",
            value=CONFLUENCE_BASE_URL,
            help="e.g., https://your-domain.atlassian.net/wiki",
        )
        email = st.text_input("Email (username)", value=CONFLUENCE_EMAIL)
        token = st.text_input("API Token", value=CONFLUENCE_API_TOKEN, type="password")
        max_pages = st.number_input(
            "Max pages to fetch", min_value=1, max_value=5000, value=200, step=50
        )
        cql = st.text_input("CQL (optional)", value="type=page")

        fetch = st.button("Fetch Pages")

    # Stage-0 style guidance when idle
    if not fetch:
        st.write(
            "✅ Project bootstrap is ready. Next step: wire Confluence ingestion (Stage 1)."
        )
        with st.expander("Environment placeholders"):
            st.code(
                "CONFLUENCE_BASE_URL=https://your-domain.atlassian.net/wiki\n"
                "CONFLUENCE_EMAIL=you@example.com\n"
                "CONFLUENCE_API_TOKEN=atlassian_api_token\n",
                language="bash",
            )
        st.info(
            "Enter your Confluence credentials in the sidebar and click "
            "**Fetch Pages**."
        )
        # Adjust the run command to your actual file path/name
        st.info(
            "Use Docker or run `pip install -r requirements.txt && "
            "streamlit run src/confluence_llm/app.py`."
        )
        return

    # Stage-1 ingestion flow
    if not (base_url and email and token):
        st.error("Please provide Base URL, Email, and API Token.")
        st.stop()

    try:
        client = ConfluenceClient(base_url, email, token)
        pages = list(
            client.iter_all_pages(
                cql=cql or None,
                page_size=50,
                max_pages=int(max_pages),
            )
        )
        df = normalize_pages(pages)
    except Exception as e:
        st.exception(e)
        st.stop()

    st.success("Fetched pages successfully.")
    st.markdown(f"**Summary:** {summarize(df)}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pages by Space")
        by_space = page_counts_by_space(df)
        st.dataframe(by_space, use_container_width=True)
    with col2:
        st.subheader("Pages by Last Updater")
        by_user = page_counts_by_user(df)
        st.dataframe(by_user, use_container_width=True)

    with st.expander("Raw Pages (normalized)"):
        st.dataframe(df, use_container_width=True)

    # Keep the Stage-0 run hint visible after results too
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
