"""Run the 23-feature model-comparison and specificity-bridge analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reproduction import entrypoint
if __name__ == '__main__':
    entrypoint('comparison')
