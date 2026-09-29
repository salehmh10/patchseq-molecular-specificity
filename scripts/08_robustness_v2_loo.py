#!/usr/bin/env python
"""Reproduce the R2 dominant-gene LOO target refits in their isolated namespace."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts/run_gene_omission.py"


if __name__ == "__main__":
    if not IMPLEMENTATION.is_file():
        raise FileNotFoundError(f"R2 implementation is missing: {IMPLEMENTATION}")
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
