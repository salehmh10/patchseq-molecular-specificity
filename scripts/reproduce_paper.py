"""Run analysis stages in their declared dependency order."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reproduction import run_stage

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('primary', 'robustness', 'specificity', 'comparison', 'extended', 'all'), required=True)
    args = parser.parse_args()
    stages = ('primary', 'robustness', 'specificity', 'comparison', 'extended') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        run_stage(stage)
        if stage == 'comparison':
            run_stage(stage, 'resume')

if __name__ == '__main__':
    main()
