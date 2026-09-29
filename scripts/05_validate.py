#!/usr/bin/env python
"""Validate saved OOF predictions without contaminating model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation import finalize_saved_level0_validation, run_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--finalize-existing-level0",
        action="store_true",
        help="audit completed Level-0 artifacts without repeating bootstrap/permutations/fits",
    )
    parser.add_argument("--measured-core-runtime-seconds", type=float)
    args = parser.parse_args()
    if args.finalize_existing_level0:
        if args.level != 0 or args.measured_core_runtime_seconds is None:
            parser.error("existing Level-0 finalization requires --level 0 and measured runtime")
        result = finalize_saved_level0_validation(
            ROOT, measured_core_runtime_seconds=args.measured_core_runtime_seconds
        )
    else:
        result = run_validation(ROOT, level=args.level)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
