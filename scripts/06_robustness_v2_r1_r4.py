"""Reproduce the isolated R1 incremental and R4 split-stability analyses.

The full, audited implementation is preserved with its result namespace so
the post-hoc analysis remains separate from the frozen primary pipeline.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts/run_robustness.py"


if __name__ == "__main__":
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
