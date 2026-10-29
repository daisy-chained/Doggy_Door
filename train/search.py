"""usage: train/.venv/bin/python train/search.py <search.yaml> (--budget NAME | --all) [key=value ...]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import search


def main():
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        sys.exit(__doc__)
    path, rest, budget, run_all = args[0], args[1:], None, False
    if "--all" in rest:
        rest.remove("--all"); run_all = True
    if "--budget" in rest:
        i = rest.index("--budget")
        budget = rest[i + 1]
        del rest[i:i + 2]
    if bool(budget) == run_all:
        sys.exit(__doc__)
    s = search.load(path, rest)
    names = list(s["budgets"]) if run_all else [budget]
    if budget and budget not in s["budgets"]:
        sys.exit(f"unknown budget {budget!r}; available: {', '.join(s['budgets'])}")
    for b in names:
        r = search.run(s, b)
        print(f"{r['study']}: {r['trials']} trials  best trial {r['best_trial']}  {s['objective']} {r['best_value']:.4f}  "
              f"test_map {r['test']['map']:.4f}  {r['best_params']}")


if __name__ == "__main__":
    main()
