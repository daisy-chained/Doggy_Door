"""usage: train/.venv/bin/python train/train.py <config.yaml> [key=value ...]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import config, loop


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cfg = config.load(sys.argv[1], sys.argv[2:])
    r = loop.train(cfg)
    test = f"{r['test']['map']:.4f}" if r["test"] else "-"
    print(f"run {r['run_id']}  out {r['out']}  best epoch {r['best_epoch']}  best val_map {r['best_val_map']:.4f}  test_map {test}")


if __name__ == "__main__":
    main()
