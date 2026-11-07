"""usage: train/.venv/bin/python train/fit_tree.py <door run> <handle run> <detector run> <export dir> [--type-run RUN] [--state-run RUN] [--split-fit val] [--split-test test] [--score-min 0.3] [--device cpu] [--seed 0] [--out DIR]
Learned decision tree (and a boosted model) over the cascade's cues: fit on the fit split, scored on the test split, on the true boxes and on the
detector's boxes apart, next to the hand cascade on the same test doors. Writes <out>/learned_tree.json + rules_<target>_<boxes>.txt + boost_<target>_<boxes>.pkl (the backup order for the failsafe) (default out: <door run>/learned_tree)."""
import argparse, json, pathlib, pickle, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import cascade, learned_tree, score

TARGETS = ("opening", "hinge_side")
MIN_FIT = 20


def run_boxes(name, args, preds_fit, preds_test, out):
    """One box set (preds None = true boxes): cue tables of both splits -> {target: {"tree", "boost", "hand", "hand_boost", "tree_depth"}} or {"skipped": reason}."""
    kw = dict(score_min=args.score_min, type_run=args.type_run, state_run=args.state_run, hinges=True, angle=True)
    fit_rows, fit_truth = cascade.cue_table(args.door_run, args.handle_run, args.export, args.split_fit, preds=preds_fit, **kw)
    if len(fit_rows) < MIN_FIT:
        return {"skipped": f"only {len(fit_rows)} doors in the {args.split_fit} split (< {MIN_FIT})"}
    test_rows, test_truth, dec = cascade.cue_table(args.door_run, args.handle_run, args.export, args.split_test, preds=preds_test, with_decisions=True, **kw)
    if not test_rows:
        return {"skipped": f"no doors in the {args.split_test} split"}
    res = {}
    for target in TARGETS:
        y_fit, y_test = [t[target] for t in fit_truth], [t[target] for t in test_truth]
        hand = sum(d[target] == y for d, y in zip(dec, y_test)) / len(y_test)
        res[target] = {"hand": {"acc": hand, "top2": hand, "n": len(y_test)}}
        for kind in ("tree", "boost"):
            model = learned_tree.fit(fit_rows, y_fit, kind, args.seed)
            res[target][kind] = learned_tree.evaluate(model, test_rows, y_test)
            if kind == "tree":
                res[target]["tree_depth"] = model.get_depth()
                (out / f"rules_{target}_{name}.txt").write_text(learned_tree.rules(model))
            else:
                res[target]["hand_boost"] = learned_tree.evaluate_backup([d[target] for d in dec], model, test_rows, y_test)
                (out / f"boost_{target}_{name}.pkl").write_bytes(pickle.dumps(model))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("door_run")
    ap.add_argument("handle_run")
    ap.add_argument("detector_run")
    ap.add_argument("export")
    ap.add_argument("--type-run", help="props run with a door_type head (default: P(slide) of the door run's opening head)")
    ap.add_argument("--state-run", help="props run with a state head (P(closed) cue; without it NaN)")
    ap.add_argument("--split-fit", default="val")
    ap.add_argument("--split-test", default="test")
    ap.add_argument("--score-min", type=float, default=0.3)
    ap.add_argument("--device", help="detector device (default: from its config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="default: <door run>/learned_tree")
    args = ap.parse_args()
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.door_run) / "learned_tree"
    out.mkdir(parents=True, exist_ok=True)
    res = {"features": cascade.FEATURES, "door_run": args.door_run, "handle_run": args.handle_run, "state_run": args.state_run, "type_run": args.type_run,
           "detector": args.detector_run, "export": args.export, "split_fit": args.split_fit, "split_test": args.split_test}
    res["true_boxes"] = run_boxes("true_boxes", args, None, None, out)
    _, pf, _ = score.predict_run(args.detector_run, args.export, args.split_fit, args.device)
    _, pt, _ = score.predict_run(args.detector_run, args.export, args.split_test, args.device)
    res["detector_boxes"] = run_boxes("detector_boxes", args, pf, pt, out)
    (out / "learned_tree.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"{'boxes':<15}{'target':<12}{'hand':>7}{'tree':>7}{'(depth)':>8}{'boost':>7}{'tree top2':>10}{'boost top2':>11}{'hand+boost top2':>16}{'n':>6}")
    for boxes in ("true_boxes", "detector_boxes"):
        if "skipped" in res[boxes]:
            print(f"{boxes:<15}skipped: {res[boxes]['skipped']}")
            continue
        for t, r in res[boxes].items():
            print(f"{boxes:<15}{t:<12}{r['hand']['acc']:>7.3f}{r['tree']['acc']:>7.3f}{'(' + str(r['tree_depth']) + ')':>8}{r['boost']['acc']:>7.3f}"
                  f"{r['tree']['top2']:>10.3f}{r['boost']['top2']:>11.3f}{r['hand_boost']['top2']:>16.3f}{r['tree']['n']:>6}")
    print(f"wrote {out / 'learned_tree.json'}")


if __name__ == "__main__":
    main()
