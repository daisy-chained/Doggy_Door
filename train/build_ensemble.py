"""usage: train/.venv/bin/python train/build_ensemble.py OUT MEMBER_DIR... [--device cuda]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import ensemble


def main():
    args, device = sys.argv[1:], "cpu"
    if "--device" in args:
        i = args.index("--device")
        device = args[i + 1]
        del args[i:i + 2]
    if len(args) < 2:
        sys.exit(__doc__)
    r = ensemble.build(args[1:], args[0], device)
    print(f"ensemble of {len(r['members'])} -> {args[0]}  temperature {r['temperature']}")
    print("test " + "  ".join(f"{k} {v:.4f}" for k, v in r["test"].items()))


if __name__ == "__main__":
    main()
