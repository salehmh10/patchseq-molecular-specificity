"""Command-line entry point for Matched-control analysis specificity analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_specificity import run_revision_specificity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("smoke", "full", "resume"))
    arguments = parser.parse_args()
    result = run_revision_specificity(arguments.config, mode=arguments.mode)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
