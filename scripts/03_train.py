#!/usr/bin/env python
"""Run the smoke benchmark or full grouped OOF training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore validated full/partial training caches and recompute.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_training(
                ROOT,
                level=args.level,
                smoke=args.smoke,
                use_cache=not args.force,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
