"""usage: train/.venv/bin/python train/build_familiar.py RUN_DIR OUT_DIR [--device cpu] [--keep 0.95] [--max-train N]"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import familiar


def main():
    p = argparse.ArgumentParser(usage=__doc__)
    p.add_argument("run_dir"); p.add_argument("out_dir")
    p.add_argument("--device", default="cpu"); p.add_argument("--keep", type=float, default=0.95)
    p.add_argument("--max-train", type=int, default=None)
    a = p.parse_args()
    r = familiar.build(a.run_dir, a.out_dir, a.device, a.keep, max_train=a.max_train)
    print(f"familiar check for {a.run_dir} -> {a.out_dir}  threshold {r['threshold']:.4f}  groups {len(r['combos'])}")
    print("test " + "  ".join(f"{k} {v:.4f}" for k, v in r["test"].items()))


if __name__ == "__main__":
    main()
