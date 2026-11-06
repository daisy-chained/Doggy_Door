"""usage: python3 sim/doorsim/radar_relabel.py <radar export> --out <new export>
Closed doors in view become negatives (no box) instead of ignore boxes; images shared by symlink (radar_export.relabel_closed)."""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from doorsim import radar_export

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(f"wrote {radar_export.relabel_closed(a.export, a.out)}")
