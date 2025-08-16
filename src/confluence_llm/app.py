from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
def main() -> int:
    logging.basicConfig(level=logging.INFO)
    logger.info("Confluence LLM app starting…")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
