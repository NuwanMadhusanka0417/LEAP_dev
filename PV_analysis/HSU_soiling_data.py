"""
HSU soiling for the Bundoora library site — run via the main download script.

    python 0_download_data.py --soiling-only

Output: ``data_raw/hsu_soiling_output.csv`` and ``data_for_viz/hsu_soiling_bundoora.csv``
(hourly SR from pvlib.soiling.hsu, Bundoora library site).

This module is kept for reference; the implementation lives in ``0_download_data.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "0_download_data.py")
    return subprocess.call([sys.executable, script, "--soiling-only"], cwd=here)


if __name__ == "__main__":
    raise SystemExit(main())
