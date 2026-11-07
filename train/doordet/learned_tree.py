"""Learned decision tree over the cascade's cues (Madelief 2026-11-07): the hand-built cascade fixes the order and thresholds itself,
here a small tree (readable rules) or a boosted model (a ceiling to compare with) learns them from the same per-door cue rows
(cascade.cue_table). NaN cues (no depth, no state run) are handled by sklearn itself."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from .cascade import FEATURES

class _Boost(HistGradientBoostingClassifier):
    """HistGradientBoosting that survives all-NaN columns (sklearn's binning fails on them: no state run, no depth): they are set to 0 in fit and predict."""

    def _blanked(self, X, fit=False):
        if fit:
            self.blank_ = np.isnan(X).all(0)
        X = X.copy()
        X[:, self.blank_] = 0.0
        return X

    def fit(self, X, y, **kw):
        return super().fit(self._blanked(X, True), y, **kw)

    def predict_proba(self, X):
        return super().predict_proba(self._blanked(X))

    def predict(self, X):
        return super().predict(self._blanked(X))


CV_FOLDS = 5
DEPTHS = range(2, 7)


def _X(rows):
    """Rows (dicts keyed by FEATURES) -> float matrix in FEATURES order, NaN kept."""
    return np.array([[float(r[f]) for f in FEATURES] for r in rows], float).reshape(-1, len(FEATURES))


def fit(rows, labels, kind="tree", seed=0):
    """Fit on cue rows + label strings. "tree": min_samples_leaf 10, max_depth 2..6 by CV_FOLDS-fold stratified CV (ties -> shallower; folds cut to the
    smallest class size, min 2; fewer than 2 members somewhere -> depth 3), refit on all rows. "boost": HistGradientBoosting, depth 3, 200 rounds.
    The model carries feature_names_ (FEATURES)."""
    X, y = _X(rows), np.asarray(labels)
    if kind == "boost":
        m = _Boost(max_depth=3, max_iter=200, learning_rate=0.1, early_stopping=False, random_state=seed)
    elif kind == "tree":
        smallest = int(min(np.unique(y, return_counts=True)[1]))
        folds = min(CV_FOLDS, smallest)
        depth = 3
        if folds >= 2:
            best = -1.0
            for dep in DEPTHS:
                acc = cross_val_score(DecisionTreeClassifier(max_depth=dep, min_samples_leaf=10, random_state=seed), X, y,
                                      cv=StratifiedKFold(folds, shuffle=True, random_state=seed)).mean()
                if acc > best + 1e-12:
                    best, depth = acc, dep
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=10, random_state=seed)
    else:
        raise ValueError(f"kind must be tree or boost, not {kind!r}")
    m.fit(X, y)
    m.feature_names_ = list(FEATURES)
    return m


def predict(model, rows):
    """-> (labels list, probabilities N x classes numpy, classes list)."""
    p = model.predict_proba(_X(rows))
    classes = [str(c) for c in model.classes_]
    return [classes[i] for i in p.argmax(1)], p, classes


def evaluate(model, rows, labels):
    """-> {"acc", "top2", "n"}; top2 = the right answer is among the two most likely (the robot's next try)."""
    labels_p, p, classes = predict(model, rows)
    n = len(labels)
    top2 = [set(np.array(classes)[np.argsort(-p[i])[:2]]) for i in range(n)]
    return {"acc": float(np.mean([a == b for a, b in zip(labels_p, labels)])) if n else 0.0,
            "top2": float(np.mean([t in s for t, s in zip(labels, top2)])) if n else 0.0, "n": n}


def rules(model):
    """Readable text of a tree (sklearn export_text with the feature names); "" for the boosted model."""
    return export_text(model, feature_names=list(FEATURES)) if isinstance(model, DecisionTreeClassifier) else ""
