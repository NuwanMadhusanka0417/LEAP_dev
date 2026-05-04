#!/usr/bin/env python3
"""
Serve this folder (PV_analysis) over HTTP so the JS dashboard can fetch data_for_viz.

Run from anywhere, e.g.:
  python serve_js_dashboard.py
  python path/to/PV_analysis/serve_js_dashboard.py

Then open (printed below): http://127.0.0.1:8080/JS_viz/
Cleaned-meter QC chart: http://127.0.0.1:8080/data_cleaned/inspect_cleaned.html
"""
from __future__ import annotations

import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    if not (ROOT / "JS_viz" / "index.html").is_file():
        print("ERROR: Expected JS_viz/index.html next to this script.", file=sys.stderr)
        sys.exit(1)
    if not (ROOT / "data_for_viz" / "hourly_library_master.csv").is_file():
        print(
            "WARNING: data_for_viz/hourly_library_master.csv not found — charts will fail until it exists.",
            file=sys.stderr,
        )

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def log_message(self, format, *args):
            return  # quieter

    print(f"Serving directory: {ROOT}")
    last_err = None
    for port in range(8080, 8090):
        try:
            httpd = socketserver.TCPServer(("", port), Handler)
        except OSError as e:
            last_err = e
            continue
        url = f"http://127.0.0.1:{port}/JS_viz/"
        print(f"Open in your browser: {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    print(f"ERROR: Could not bind a port (8080–8089): {last_err}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
