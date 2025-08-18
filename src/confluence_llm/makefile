.PHONY: run api ingest streamlit

run: api

api:
\tuvicorn api:app --host 0.0.0.0 --port 8080 --reload

ingest:
\tpython ingest.py

streamlit:
\tstreamlit run streamlit_app.py
