"""CLDB live application entry point.

Launch the live web app:
    python main.py
    python main.py --qps 120 --port 8080

The original query-ingestion HTTP service remains available as
``python -m cldb.server``.
"""
from cldb.webapp import main

if __name__ == "__main__":
    main()
