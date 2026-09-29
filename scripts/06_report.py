#!/usr/bin/env python
"""Generate figures, paper tables, and prose from verified result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import generate_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    print(json.dumps(generate_documents(ROOT, runtime_level=args.level), indent=2))


if __name__ == "__main__":
    main()
