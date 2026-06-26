"""Model evaluation metrics: Gini, equal-weight lift, double lift, MSE, MAE, VIF, bootstrap."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union
from xml.parsers.expat import errors

import numpy as np
import polars as pl


# ── Gini coefficient ──────────────────────────────────────────────────────────

def _gini(y_pred: np.ndarray, y_true: np.ndarray, w: np.ndarray) -> float:
    idx = np.argsort(-y_pred)
    ys, ws = y_true[idx], w[idx]
    cum_w = np.concatenate([[0.0], np.cumsum(ws) / ws.sum()])
    total_loss = (ys * ws).sum()
    if total_loss == 0:
        return 0.0
    cum_loss = np.concatenate([[0.0], np.cumsum(ys * ws) / total_loss])
    return float(2.0 * np.trapezoid(cum_loss, cum_w) - 1.0)

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
    w = np.ones(len(y_true)) if weights is None else weights

    model_gini = _gini(y_pred, y_true, w)
    perfect = _gini(y_true, y_true, w) if normalize else 1.0

    return model_gini / perfect if perfect > 1e-12 else 0.0

# ── Lift table ────────────────────────────────────────────────────────────────

def _weighted_relativity(vals1: pl.Series, weights: pl.Series, vals2: Optional[pl.Series] = None) -> pl.Series:
    """
    Compute relativity to the weighted mean for one or two columns in a summary DataFrame.
    """
    vals1_mean = vals1.sum() / weights.sum()
    rels1 = (vals1 / weights) / vals1_mean if vals1_mean > 1e-12 else np.zeros_like(vals1)

    rels2 = None
    if vals2 is not None:
        vals2_mean = vals2.sum() / weights.sum()
        rels2 = (vals2 / weights) / vals2_mean if vals2_mean > 1e-12 else np.zeros_like(vals2)

    return rels1, rels2

def lift_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_buckets: int = 10
) -> pl.DataFrame:
    """
    Build a lift table with equal-weight buckets sorted by ``y_pred``.

    Each bucket contains approximately equal total exposure (weight).

    Returns
    -------
    pl.DataFrame
        Columns: ``bucket``, ``actual``, ``predicted``, ``exposure``, ``lift``.
    """
    w = np.ones(len(y_true)) if weights is None else weights

    bks = np.quantile(y_pred, q=np.linspace(0, 1, n_buckets + 1), weights=w, method='inverted_cdf')
    bks = sorted(set(dict.fromkeys(bks)))[1:-1]  # Exclude min and max

    lift_tab = pl.DataFrame(
        {
            "predicted_ratio": y_pred,
            "exposure": w,
            'actual': y_true * w,
            'predicted': y_pred * w
        }
    )
    lift_tab = lift_tab.with_columns([
        pl.col('predicted_ratio').cut(bks, labels=[str(x) for x in range(1, n_buckets + 1)], left_closed=True).cast(pl.String).cast(pl.Int8).alias("bucket")
    ])

    
    lift_tab_agg = lift_tab.group_by('bucket').agg(pl.exclude('predicted_ratio').sum()).filter(pl.col('bucket').is_not_null()).sort("bucket", descending=False)

    pred_rels, act_rels = _weighted_relativity(lift_tab_agg['predicted'], lift_tab_agg['exposure'], lift_tab_agg['actual'])

    lift_tab_agg = lift_tab_agg.with_columns(pred_rels.alias("predicted"), act_rels.alias("actual"))

    return lift_tab_agg.select(pl.col('bucket', 'actual', 'predicted', 'exposure'))

def lift_rmse(lift_tab: pl.DataFrame) -> float:
    """
    Compute the RMSE of predicted vs actual loss amounts across lift table buckets.

    This is a single-number summary of how well the model's predicted risk
    matches actual risk across the distribution of predictions.
    """
    total_loss_ratio = np.average(lift_tab["actual"], weights=lift_tab["exposure"])
    preds_for_rmse = lift_tab["predicted"] * total_loss_ratio * lift_tab["exposure"]

    residuals = preds_for_rmse - (lift_tab["actual"] * lift_tab["exposure"]) 
    return float(np.sqrt(np.average(residuals ** 2, weights=lift_tab["exposure"])))

def lift_range(lift_tab: pl.DataFrame) -> float:
    """
    Compute the range of predicted loss ratio relativities across lift table buckets.

    This is a single-number summary of how much the model differentiates risk
    across the distribution of predictions.  A higher range indicates better
    discrimination.
    """
    return float(lift_tab['actual'][lift_tab['bucket'].arg_max()] / lift_tab['predicted'][lift_tab['bucket'].arg_max()])

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
    w = np.ones(len(y_true)) if weights is None else weights

    pred_ratio = pred2 / pred1  # Sort ratio
    bks = np.quantile(pred_ratio, q=np.linspace(0, 1, n_buckets + 1), weights=w, method='inverted_cdf')

    # Remove duplicates while preserving order of breaks
    bks = sorted(set(dict.fromkeys(bks)))[1:-1]  # Exclude min and max

    dl_tab = pl.DataFrame(
        {
            "Actual_Loss": y_true * w,
            "Pred1_Loss": pred1 * w,
            "Pred2_Loss": pred2 * w,
            "Ratio": pred_ratio,
            "weight": w,
        }
    )
    dl_tab = dl_tab.with_columns([
        pl.col('Ratio').cut(bks, labels=[str(x) for x in range(1, n_buckets + 1)], left_closed=True).cast(pl.String).cast(pl.Int8).alias("bucket")
    ])

    dl_agg = (dl_tab.group_by("bucket").agg(pl.exclude('Ratio').sum())).filter(pl.col('bucket').is_not_null()).sort("bucket", descending=False)

    pred1_rels, pred2_rels = _weighted_relativity(dl_agg["Pred1_Loss"], dl_agg["weight"], dl_agg["Pred2_Loss"])
    act_rels, _ = _weighted_relativity(dl_agg["Actual_Loss"], dl_agg["weight"])

    dl_agg = dl_agg.with_columns([
        pred1_rels.alias('model1'),
        pred2_rels.alias('model2'),
        act_rels.alias("actual"),
    ])

    return dl_agg.select(pl.col('bucket', 'weight', 'actual', 'model1', 'model2'))

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
    w = np.ones(len(y_true)) if weights is None else weights

    resid = w * (y_true - y_pred)
    mse = np.average(resid ** 2, weights=w)

    lift_tab = lift_table(y_true, y_pred, weights, n_buckets=20)
    lift_range_val = lift_range(lift_tab)
    lift_rmse_val = lift_rmse(lift_tab)

    metrics = {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": np.average(np.abs(resid), weights=w),
        "gini": gini_coefficient(y_true, y_pred, w, normalize=False),
        "gini_norm": gini_coefficient(y_true, y_pred, w, normalize=True),
        'lift_range': lift_range_val,
        'lift_rmse': lift_rmse_val
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
    _lower_better = {"mse", "rmse", "mae", 'lift_rmse'}
    _higher_better = {"gini", "gini_norm", 'lift_range'}

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
    w = np.ones(len(y_true)) if weights is None else weights
    rng = np.random.RandomState(random_state)

    if metric_fns is None:
        metric_fns = {
            "gini_norm": lambda yt, yp, wt: gini_coefficient(yt, yp, wt, normalize=True),
            "mse": lambda yt, yp, wt: -np.average((yt - yp) ** 2, weights=wt),
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
