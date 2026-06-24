"""Model evaluation metrics: Gini, equal-weight lift, double lift, MSE, MAE, VIF, bootstrap."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import polars as pl

# np.trapz was renamed to np.trapezoid in numpy 2.0 and removed in 2.4
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ── Gini coefficient ──────────────────────────────────────────────────────────

def gini_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> float:
    """
    Compute the (optionally normalised) Gini coefficient.

    Sorts by ``y_pred`` descending and builds the Lorenz curve of actual losses
    vs cumulative exposure.

    Parameters
    ----------
    normalize : bool
        Divide by the oracle Gini (sort by ``y_true``) to get a value in
        [0, 1].  A score of 1.0 means the model perfectly ranks risks.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)

    def _gini(order_by: np.ndarray) -> float:
        idx = np.argsort(-order_by)
        ys, ws = y_true[idx], w[idx]
        cum_w = np.concatenate([[0.0], np.cumsum(ws) / ws.sum()])
        total_loss = (ys * ws).sum()
        if total_loss == 0:
            return 0.0
        cum_loss = np.concatenate([[0.0], np.cumsum(ys * ws) / total_loss])
        return float(2.0 * _trapezoid(cum_loss, cum_w) - 1.0)

    model_gini = _gini(y_pred)
    if normalize:
        perfect = _gini(y_true)
        return model_gini / perfect if abs(perfect) > 1e-12 else 0.0
    return model_gini


# ── Equal-weight bucket assignment ────────────────────────────────────────────

def _equal_weight_buckets(
    pred: np.ndarray,
    weights: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """
    Assign bucket indices ``0 … n_buckets-1`` so each bucket contains
    approximately equal total weight (exposure).

    Observations are first sorted by ``pred`` ascending.  Cumulative weight
    is then split into ``n_buckets`` equal segments.

    Parameters
    ----------
    pred : np.ndarray
        Model predictions (used for ordering).
    weights : np.ndarray
        Sample weights (must be positive).
    n_buckets : int

    Returns
    -------
    np.ndarray of int, same length as *pred*.
    """
    order = np.argsort(pred)
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1]
    target = total_w / n_buckets

    bucket_sorted = np.zeros(len(pred), dtype=int)
    current_bucket = 0
    threshold = target

    for i, cw in enumerate(cum_w):
        bucket_sorted[i] = current_bucket
        # Advance bucket when we've accumulated enough weight,
        # but don't exceed the last bucket index
        if cw >= threshold and current_bucket < n_buckets - 1:
            current_bucket += 1
            threshold = target * (current_bucket + 1)

    # Map back to original row order
    result = np.empty(len(pred), dtype=int)
    result[order] = bucket_sorted
    return result


# ── Lift table ────────────────────────────────────────────────────────────────

def lift_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_buckets: int = 10,
) -> pl.DataFrame:
    """
    Build a lift table with equal-weight buckets sorted by ``y_pred``.

    Each bucket contains approximately equal total exposure (weight).

    Returns
    -------
    pl.DataFrame
        Columns: ``bucket``, ``actual``, ``predicted``, ``exposure``, ``lift``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)

    buckets = _equal_weight_buckets(y_pred, w, n_buckets)
    overall_mean = float(np.average(y_true, weights=w)) if w.sum() > 0 else 0.0

    rows = []
    for b in range(n_buckets):
        mask = buckets == b
        if not mask.any():
            continue
        yw, ww = y_true[mask], w[mask]
        wb = float(ww.sum())
        actual = float(np.average(yw, weights=ww)) if wb > 0 else 0.0
        predicted = float(np.average(y_pred[mask], weights=ww)) if wb > 0 else 0.0
        rows.append({
            "bucket": b + 1,
            "actual": actual,
            "predicted": predicted,
            "exposure": wb,
            "lift": actual / overall_mean if overall_mean != 0 else float("nan"),
        })

    return pl.DataFrame(rows)


# ── Double lift table ─────────────────────────────────────────────────────────

def double_lift_table(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_buckets: int = 10,
) -> pl.DataFrame:
    """
    Build a double-lift table comparing two models using equal-weight buckets.

    Observations are sorted by the ratio ``pred1 / pred2`` and binned into
    ``n_buckets`` equal-weight groups.

    Returns
    -------
    pl.DataFrame
        Columns: ``bucket``, ``actual``, ``model1``, ``model2``,
        ``ratio_mean``, ``exposure``.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)

    safe_pred2 = np.where(np.abs(pred2) < 1e-12, 1e-12, pred2)
    ratio = pred1 / safe_pred2

    buckets = _equal_weight_buckets(ratio, w, n_buckets)

    rows = []
    for b in range(n_buckets):
        mask = buckets == b
        if not mask.any():
            continue
        ww = w[mask]
        wb = float(ww.sum())
        rows.append({
            "bucket": b + 1,
            "actual": float(np.average(y_true[mask], weights=ww)) if wb > 0 else 0.0,
            "model1": float(np.average(pred1[mask], weights=ww)) if wb > 0 else 0.0,
            "model2": float(np.average(pred2[mask], weights=ww)) if wb > 0 else 0.0,
            "ratio_mean": float(np.average(ratio[mask], weights=ww)) if wb > 0 else 0.0,
            "exposure": wb,
        })

    return pl.DataFrame(rows)


# ── Summary metrics ───────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    version_name: str = "model",
) -> pl.DataFrame:
    """
    Compute MSE, RMSE, MAE, Gini (raw and normalised).

    Returns
    -------
    pl.DataFrame
        Columns: ``metric``, ``<version_name>``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)

    resid = y_true - y_pred
    mse = float(np.average(resid ** 2, weights=w))

    metrics = {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": float(np.average(np.abs(resid), weights=w)),
        "gini": gini_coefficient(y_true, y_pred, w, normalize=False),
        "gini_norm": gini_coefficient(y_true, y_pred, w, normalize=True),
    }
    return pl.DataFrame({"metric": list(metrics.keys()), version_name: list(metrics.values())})


def double_lift_score(
    dl_table: pl.DataFrame,
    deviation: str = "absolute",
) -> float:
    """
    Compute the double-lift score from a double-lift table.

    Measures which model's bucket means are closer to actual across the full
    equal-weight, ratio-sorted axis.

    Parameters
    ----------
    deviation : {'absolute', 'relative'}
        ``'absolute'`` (default) — score = Σ(|m1 − a| − |m2 − a|) per bucket.
        ``'relative'`` — score = Σ(|a/m1 − 1| − |a/m2 − 1|) per bucket.
        Relative is scale-free: a miss of 0.05 against actual 0.50 counts the
        same as 0.10 against 1.00.

    * **Negative** → model1 is closer to actual (model1 wins).
    * **Positive** → model2 is closer to actual (model2 wins).
    * **Zero** → tie.
    """
    m1 = dl_table["model1"].to_numpy()
    m2 = dl_table["model2"].to_numpy()
    a = dl_table["actual"].to_numpy()

    if deviation == "relative":
        safe_m1 = np.where(np.abs(m1) < 1e-12, 1e-12, m1)
        safe_m2 = np.where(np.abs(m2) < 1e-12, 1e-12, m2)
        return float((np.abs(a / safe_m1 - 1.0) - np.abs(a / safe_m2 - 1.0)).sum())

    return float((np.abs(m1 - a) - np.abs(m2 - a)).sum())


def compare_metrics(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    weights: Optional[np.ndarray] = None,
    name1: str = "model1",
    name2: str = "model2",
    dl_score: Optional[float] = None,
    deviation: str = "absolute",
) -> pl.DataFrame:
    """
    Return side-by-side metrics for two model versions with a ``winner`` column.

    Parameters
    ----------
    dl_score : float, optional
        Pre-computed :func:`double_lift_score`.  When supplied, a
        ``double_lift_score`` row is appended; the value is shown in both model
        columns (it is a single comparison number, not per-model) and the
        winner is determined by its sign.
    deviation : {'absolute', 'relative'}
        Forwarded to :func:`double_lift_score` when computing winner annotation.
    """
    m1 = compute_metrics(y_true, pred1, weights, name1)
    m2 = compute_metrics(y_true, pred2, weights, name2)
    merged = m1.join(m2, on="metric")

    # ── winner annotation ────────────────────────────────────────────────────
    _lower_better = {"mse", "rmse", "mae"}
    _higher_better = {"gini", "gini_norm"}

    winners: list[str] = []
    for row in merged.iter_rows(named=True):
        metric = row["metric"]
        v1, v2 = row[name1], row[name2]
        if metric in _lower_better:
            winners.append(name1 if v1 < v2 else (name2 if v2 < v1 else "tie"))
        elif metric in _higher_better:
            winners.append(name1 if v1 > v2 else (name2 if v2 > v1 else "tie"))
        else:
            winners.append("")

    merged = merged.with_columns(pl.Series("winner", winners))

    # ── optional double-lift score row ───────────────────────────────────────
    if dl_score is not None:
        dl_winner = name1 if dl_score < 0 else (name2 if dl_score > 0 else "tie")
        dl_row = pl.DataFrame({
            "metric": ["double_lift_score"],
            name1: [dl_score],
            name2: [dl_score],
            "winner": [dl_winner],
        })
        merged = pl.concat([merged, dl_row])

    return merged


# ── Variance Inflation Factor ────────────────────────────────────────────────

def vif_table(design_matrix: pl.DataFrame) -> pl.DataFrame:
    """
    Compute Variance Inflation Factor for each column in a design matrix.

    VIF measures multicollinearity: VIF = 1 / (1 - R^2) where R^2 comes
    from regressing each feature on all others via OLS.

    Parameters
    ----------
    design_matrix : pl.DataFrame
        Numeric columns only (post-preprocessing, dummy-encoded).

    Returns
    -------
    pl.DataFrame
        Columns: ``variable``, ``vif``, sorted descending.
    """
    X = design_matrix.to_numpy().astype(float)
    n_features = X.shape[1]
    col_names = design_matrix.columns

    vifs = []
    for i in range(n_features):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        # Add intercept
        X_aug = np.column_stack([np.ones(len(y_i)), X_others])
        # OLS: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_aug, y_i, rcond=None)[0]
            y_hat = X_aug @ beta
            ss_res = np.sum((y_i - y_hat) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            vif_val = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        except np.linalg.LinAlgError:
            vif_val = float("inf")
        vifs.append({"variable": col_names[i], "vif": vif_val})

    return pl.DataFrame(vifs).sort("vif", descending=True)


# ── Bootstrap confidence intervals ──────────────────────────────────────────

def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    metric_fns: Optional[Dict[str, Callable]] = None,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Bootstrap confidence intervals on model performance metrics.

    Parameters
    ----------
    metric_fns : dict of {name: callable}, optional
        Each callable has signature ``fn(y_true, y_pred, weights) -> float``.
        Defaults to Gini (normalised) and negative MSE.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    pl.DataFrame
        Columns: ``metric``, ``point_estimate``, ``ci_lower``, ``ci_upper``,
        ``std_error``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    rng = np.random.RandomState(random_state)

    if metric_fns is None:
        metric_fns = {
            "gini_norm": lambda yt, yp, wt: gini_coefficient(yt, yp, wt, normalize=True),
            "mse": lambda yt, yp, wt: -float(np.average((yt - yp) ** 2, weights=wt)),
        }

    alpha = (1 - ci) / 2
    n = len(y_true)

    # Point estimates
    points = {name: fn(y_true, y_pred, w) for name, fn in metric_fns.items()}

    # Bootstrap
    boot_results: Dict[str, list] = {name: [] for name in metric_fns}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt_b, yp_b, w_b = y_true[idx], y_pred[idx], w[idx]
        for name, fn in metric_fns.items():
            boot_results[name].append(fn(yt_b, yp_b, w_b))

    rows = []
    for name in metric_fns:
        samples = np.array(boot_results[name])
        rows.append({
            "metric": name,
            "point_estimate": points[name],
            "ci_lower": float(np.quantile(samples, alpha)),
            "ci_upper": float(np.quantile(samples, 1 - alpha)),
            "std_error": float(np.std(samples)),
        })

    return pl.DataFrame(rows)
