"""usage: train/.venv/bin/python train/train_props.py <config.yaml> [key=value ...]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import config, props


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cfg = config.load(sys.argv[1], sys.argv[2:], resolve=props.resolve)
    r = props.train(cfg)
    print(f"run {r['run_id']}  out {r['out']}  best epoch {r['best_epoch']}  best val_acc {r['best_val_acc']:.4f}  temperature {r['temperature']}")


if __name__ == "__main__":
    main()
