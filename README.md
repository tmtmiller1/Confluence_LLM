# Confluence_LLM

Fresh repo skeleton using src/ layout, pre-commit, and a CI workflow file (local only until you add a remote).
See CONTRIBUTING.md for workflow and standards.

# Confluence LLM Project — Stage 1 (Core Confluence Ingestion)

This stage introduces a minimal Confluence API client, basic normalization into a DataFrame,
and a simple Streamlit UI to fetch and visualize page counts.

## Quickstart

```bash
pip install -r requirements.txt
# Set env vars or fill the UI fields:
export CONFLUENCE_BASE_URL=https://your-domain.atlassian.net/wiki
export CONFLUENCE_EMAIL=you@example.com
export CONFLUENCE_API_TOKEN=atlassian_api_token

streamlit run streamlit_app.py
```
