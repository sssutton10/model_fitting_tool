"""Validated model evaluation metrics (NumPy input, Polars tabular output)."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np
import polars as pl


def _validate_arrays(*arrays: np.ndarray, names: Sequence[str] = ()) -> Tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(value, dtype=float) for value in arrays)
    if not converted or any(value.ndim != 1 for value in converted):
        raise ValueError("Metric inputs must be one-dimensional arrays.")
    if not len(converted[0]) or any(len(value) != len(converted[0]) for value in converted[1:]):
        raise ValueError("Metric inputs must be non-empty and have matching lengths.")
    for i, value in enumerate(converted):
        if not np.all(np.isfinite(value)):
            label = names[i] if i < len(names) else "input"
            raise ValueError(f"{label} must contain only finite values.")
    return converted


def _validate_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    values = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"weights must be one-dimensional with length {n}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have a positive total.")
    return values


def _validate_bucket_count(n_buckets: int, n: int) -> None:
    if not isinstance(n_buckets, int) or isinstance(n_buckets, bool) or n_buckets < 2 or n_buckets > n:
        raise ValueError(f"n_buckets must be an integer from 2 through {n}.")


def _gini(y_pred: np.ndarray, y_true: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(-y_pred, kind="mergesort")
    ys, ws = y_true[order], weights[order]
    loss = float(np.dot(ys, ws))
    if abs(loss) <= 1e-12:
        return 0.0
    cumulative_weight = np.concatenate(([0.0], np.cumsum(ws) / ws.sum()))
    cumulative_loss = np.concatenate(([0.0], np.cumsum(ys * ws) / loss))
    return float(2 * np.trapezoid(cumulative_loss, cumulative_weight) - 1)


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None, normalize: bool = True) -> float:
    yt, yp = _validate_arrays(y_true, y_pred, names=("y_true", "y_pred")); w = _validate_weights(weights, len(yt))
    model_gini = _gini(yp, yt, w)
    oracle = _gini(yt, yt, w) if normalize else 1.0
    return float(model_gini / oracle) if abs(oracle) > 1e-12 else 0.0


def _weighted_relativity(vals1: pl.Series, weights: pl.Series, vals2: Optional[pl.Series] = None):
    total = float(weights.sum())
    if total <= 0: raise ValueError("exposure must have a positive total.")
    mean1 = float(vals1.sum()) / total
    rel1 = vals1 / weights / mean1 if abs(mean1) > 1e-12 else pl.Series(vals1.name, np.zeros(len(vals1)))
    if vals2 is None: return rel1, None
    mean2 = float(vals2.sum()) / total
    rel2 = vals2 / weights / mean2 if abs(mean2) > 1e-12 else pl.Series(vals2.name, np.zeros(len(vals2)))
    return rel1, rel2


def _bucket_ids(scores: np.ndarray, weights: np.ndarray, n_buckets: int) -> np.ndarray:
    """Assign stable, approximately equal-exposure buckets without cut-label bugs."""
    order = np.argsort(scores, kind="mergesort")
    cumulative = np.cumsum(weights[order]) - weights[order] / 2
    raw = np.floor(cumulative / weights.sum() * n_buckets).astype(int)
    raw = np.clip(raw, 0, n_buckets - 1)
    ids = np.empty(len(scores), dtype=int); ids[order] = raw + 1
    # Compact empty buckets, which naturally occur with very concentrated weights.
    used = sorted(set(ids.tolist())); remap = {old: new for new, old in enumerate(used, 1)}
    return np.array([remap[item] for item in ids], dtype=int)


def lift_table(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None, n_buckets: int = 10) -> pl.DataFrame:
    yt, yp = _validate_arrays(y_true, y_pred, names=("y_true", "y_pred")); w = _validate_weights(weights, len(yt)); _validate_bucket_count(n_buckets, len(yt))
    bucket = _bucket_ids(yp, w, n_buckets)
    rows = []
    overall = float(np.dot(yt, w) / w.sum())
    for number in sorted(set(bucket.tolist())):
        mask = bucket == number; exposure = float(w[mask].sum())
        actual = float(np.dot(yt[mask], w[mask]) / exposure)
        predicted = float(np.dot(yp[mask], w[mask]) / exposure)
        rows.append({"bucket": number, "actual": actual, "predicted": predicted, "exposure": exposure,
                     "lift": actual / overall if abs(overall) > 1e-12 else 0.0})
    return pl.DataFrame(rows)


def lift_rmse(lift_tab: pl.DataFrame) -> float:
    _require_table(lift_tab, {"actual", "predicted", "exposure"}, "lift_tab")
    actual, predicted, exposure = _validate_arrays(lift_tab["actual"], lift_tab["predicted"], lift_tab["exposure"])
    w = _validate_weights(exposure, len(exposure))
    return float(np.sqrt(np.average((actual - predicted) ** 2, weights=w)))


def lift_range(lift_tab: pl.DataFrame) -> float:
    _require_table(lift_tab, {"actual"}, "lift_tab")
    values = lift_tab["actual"].to_numpy().astype(float)
    if not len(values) or not np.all(np.isfinite(values)): raise ValueError("lift_tab actual values must be finite and non-empty.")
    return float(values.max() - values.min())


def double_lift_table(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, weights: Optional[np.ndarray] = None, n_buckets: int = 10) -> pl.DataFrame:
    yt, p1, p2 = _validate_arrays(y_true, pred1, pred2, names=("y_true", "pred1", "pred2")); w = _validate_weights(weights, len(yt)); _validate_bucket_count(n_buckets, len(yt))
    if np.any(p1 == 0): raise ValueError("pred1 must not contain zero when computing a double-lift ratio.")
    ratio = p2 / p1
    if not np.all(np.isfinite(ratio)): raise ValueError("pred1/pred2 must produce finite double-lift ratios.")
    bucket = _bucket_ids(ratio, w, n_buckets); rows = []
    for number in sorted(set(bucket.tolist())):
        mask = bucket == number; exposure = float(w[mask].sum())
        rows.append({"bucket": number, "actual": float(np.dot(yt[mask], w[mask]) / exposure),
            "model1": float(np.dot(p1[mask], w[mask]) / exposure), "model2": float(np.dot(p2[mask], w[mask]) / exposure),
            "ratio_mean": float(np.dot(ratio[mask], w[mask]) / exposure), "exposure": exposure})
    return pl.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None, version_name: str = "model") -> pl.DataFrame:
    if not isinstance(version_name, str) or not version_name: raise ValueError("version_name must be a non-empty string.")
    yt, yp = _validate_arrays(y_true, y_pred, names=("y_true", "y_pred")); w = _validate_weights(weights, len(yt))
    if len(yt) < 2:
        raise ValueError("compute_metrics requires at least two observations to compute lift metrics.")
    errors = yt - yp; mse = float(np.average(errors ** 2, weights=w)); lifts = lift_table(yt, yp, w, n_buckets=min(20, len(yt)))
    metrics = {"mse": mse, "rmse": mse ** .5, "mae": float(np.average(np.abs(errors), weights=w)),
        "gini": gini_coefficient(yt, yp, w, False), "gini_norm": gini_coefficient(yt, yp, w, True),
        "lift_range": lift_range(lifts), "lift_rmse": lift_rmse(lifts)}
    return pl.DataFrame({"metric": list(metrics), version_name: list(metrics.values())})


def _require_table(table: pl.DataFrame, required: set[str], name: str) -> None:
    if not isinstance(table, pl.DataFrame): raise TypeError(f"{name} must be a polars DataFrame.")
    missing = sorted(required - set(table.columns))
    if missing: raise ValueError(f"{name} is missing required columns: {missing}.")


def double_lift_score(dl_table: pl.DataFrame, deviation: str = "absolute") -> float:
    _require_table(dl_table, {"actual", "model1", "model2"}, "dl_table")
    if deviation not in {"absolute", "relative"}: raise ValueError("deviation must be 'absolute' or 'relative'.")
    actual, m1, m2 = _validate_arrays(dl_table["actual"], dl_table["model1"], dl_table["model2"])
    if deviation == "absolute": return float((np.abs(m1 - actual) - np.abs(m2 - actual)).sum())
    if np.any(np.abs(m1) < 1e-12) or np.any(np.abs(m2) < 1e-12): raise ValueError("relative double-lift score requires non-zero model values.")
    return float((np.abs(actual / m1 - 1) - np.abs(actual / m2 - 1)).sum())


def compare_metrics(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, weights: Optional[np.ndarray] = None, name1: str = "model1", name2: str = "model2", dl_score: Optional[float] = None, deviation: str = "absolute") -> pl.DataFrame:
    if not all(isinstance(name, str) and name for name in (name1, name2)) or name1 == name2: raise ValueError("Model names must be distinct non-empty strings.")
    m1, m2 = compute_metrics(y_true, pred1, weights, name1), compute_metrics(y_true, pred2, weights, name2)
    result = m1.join(m2, on="metric"); lower = {"mse", "rmse", "mae", "lift_rmse"}; higher = {"gini", "gini_norm", "lift_range"}
    winners = [name1 if (row[name1] < row[name2] if row["metric"] in lower else row[name1] > row[name2]) else name2 if row[name1] != row[name2] else "tie" for row in result.iter_rows(named=True)]
    result = result.with_columns(pl.Series("winner", winners))
    if dl_score is not None:
        if not np.isfinite(dl_score): raise ValueError("dl_score must be finite.")
        winner = name1 if dl_score < 0 else name2 if dl_score > 0 else "tie"
        result = pl.concat([result, pl.DataFrame({"metric": ["double_lift_score"], name1: [dl_score], name2: [dl_score], "winner": [winner]})])
    return result


def vif_table(design_matrix: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(design_matrix, pl.DataFrame): raise TypeError("design_matrix must be a polars DataFrame.")
    if not design_matrix.width: raise ValueError("design_matrix must contain at least one feature.")
    X = design_matrix.to_numpy().astype(float)
    if X.ndim != 2 or len(X) < 2 or not np.all(np.isfinite(X)): raise ValueError("design_matrix must contain at least two rows of finite numeric values.")
    rows = []
    for i, name in enumerate(design_matrix.columns):
        y, others = X[:, i], np.delete(X, i, axis=1)
        if np.var(y) <= 1e-12: vif = float("inf")
        elif not others.shape[1]: vif = 1.0
        else:
            aug = np.column_stack((np.ones(len(y)), others)); pred = aug @ np.linalg.lstsq(aug, y, rcond=None)[0]
            r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2); vif = float("inf") if r2 >= 1 - 1e-12 else float(1 / (1 - r2))
        rows.append({"variable": name, "vif": vif})
    return pl.DataFrame(rows).sort("vif", descending=True)


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None, metric_fns: Optional[Dict[str, Callable]] = None, n_bootstrap: int = 500, ci: float = .95, random_state: int = 42) -> pl.DataFrame:
    yt, yp = _validate_arrays(y_true, y_pred, names=("y_true", "y_pred")); w = _validate_weights(weights, len(yt))
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1: raise ValueError("n_bootstrap must be a positive integer.")
    if not isinstance(ci, (int, float)) or not 0 < ci < 1: raise ValueError("ci must be strictly between 0 and 1.")
    fns = metric_fns or {"gini_norm": lambda a,b,c: gini_coefficient(a,b,c,True), "mse": lambda a,b,c: np.average((a-b)**2, weights=c)}
    if not isinstance(fns, dict) or not fns or any(not isinstance(k, str) or not callable(v) for k, v in fns.items()): raise TypeError("metric_fns must be a non-empty dict of string names to callables.")
    rng = np.random.RandomState(random_state); points = {}; samples = {name: [] for name in fns}
    for name, fn in fns.items():
        value = float(fn(yt, yp, w));
        if not np.isfinite(value): raise ValueError(f"Metric '{name}' returned a non-finite point estimate.")
        points[name] = value
    for _ in range(n_bootstrap):
        idx = rng.choice(len(yt), len(yt), replace=True)
        for name, fn in fns.items():
            value = float(fn(yt[idx], yp[idx], w[idx]))
            if not np.isfinite(value): raise ValueError(f"Metric '{name}' returned a non-finite bootstrap estimate.")
            samples[name].append(value)
    alpha = (1-ci)/2
    return pl.DataFrame([{"metric": name, "point_estimate": points[name], "ci_lower": float(np.quantile(samples[name], alpha)), "ci_upper": float(np.quantile(samples[name], 1-alpha)), "std_error": float(np.std(samples[name], ddof=0))} for name in fns])
