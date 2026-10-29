"""usage: train/.venv/bin/python train/plot_search.py <search.yaml>   -> runs/search/<name>/accuracy_vs_speed.png"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from doordet import config, search

MARKERS = "osD^vP*X<>"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    s = search.load(sys.argv[1])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fig, ax = plt.subplots(figsize=(9, 6))
    ckpts = [m["checkpoint"] for m in s["space"]["models"]]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for bi, (b, ms) in enumerate(s["budgets"].items()):
        try:
            study = optuna.load_study(study_name=f"{s['name']}_{b}", storage=s["storage"])
        except KeyError:
            continue
        c = colors[bi % len(colors)]
        if ms is not None:
            ax.axvline(ms, ls="--", color=c, alpha=0.6)
        for t in study.trials:
            x = t.user_attrs.get("speed_ms")
            if x is None:
                continue
            mk = MARKERS[ckpts.index(t.params["model.checkpoint"]) % len(MARKERS)] if t.params.get("model.checkpoint") in ckpts else "o"
            if t.state == optuna.trial.TrialState.COMPLETE:
                ax.scatter(x, t.value, color=c, marker=mk, s=45, alpha=0.9)
            elif t.state == optuna.trial.TrialState.PRUNED and t.intermediate_values:
                ax.scatter(x, t.intermediate_values[max(t.intermediate_values)], color=c, marker="x", s=25, alpha=0.3)
    handles = [plt.Line2D([], [], color=colors[i % len(colors)], marker="o", ls="", label=f"{b}" + (f" ({ms} ms)" if ms else ""))
               for i, (b, ms) in enumerate(s["budgets"].items())]
    handles += [plt.Line2D([], [], color="gray", marker=MARKERS[i % len(MARKERS)], ls="", label=c.split("/")[-1]) for i, c in enumerate(ckpts)]
    handles.append(plt.Line2D([], [], color="gray", marker="x", ls="", alpha=0.4, label="pruned (last val)"))
    ax.legend(handles=handles, fontsize=8)
    ax.set(xlabel="inference time per image (ms, batch 1)", ylabel=f"val {s['objective']}", title=s["name"])
    ax.grid(alpha=0.3)
    out = config.REPO / "runs" / "search" / s["name"] / "accuracy_vs_speed.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
