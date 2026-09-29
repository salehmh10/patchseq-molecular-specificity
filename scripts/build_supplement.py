"""Compile the frozen supplement without overwriting distributed artifacts."""
import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    compiler = shutil.which('pdflatex')
    if not compiler:
        raise SystemExit('Install a TeX distribution providing pdflatex and the packages listed in the source.')
    destination = ROOT / 'build/supplement'
    destination.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        subprocess.run([compiler, '-interaction=nonstopmode', '-halt-on-error',
                        '-output-directory=' + str(destination), 'Supplementary_Material.tex'],
                       cwd=ROOT / 'supplement', check=True, stdout=subprocess.DEVNULL)
    print('Built build/supplement/Supplementary_Material.pdf')

if __name__ == '__main__':
    main()
