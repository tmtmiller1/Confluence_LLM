# Contributing

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
mypy --strict .
pytest -q
