"""Visualisation utilities (polars backend, seaborn/matplotlib output)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import seaborn as sns

from .metrics import double_lift_table, gini_coefficient, lift_table
from .variable import MISSING_SENTINEL, Preprocessor, _is_str_or_cat, compute_quantile_bin_edges, make_bin_labels

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_FIG: Tuple[int, int] = (10, 5)

# Apply seaborn theme globally so all charts share a consistent aesthetic.
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)


# ── Binning helper for plots ──────────────────────────────────────────────────

def _bin_for_plot(s: pl.Series, n_bins: int = 10) -> pl.Series:
    """
    Bin a continuous pl.Series into quantile-based string labels.

    Null values and the :data:`~variable.MISSING_SENTINEL` value
    (``-999999999``) are labelled ``'Missing'``.  Labels use the same
    ``{letter}_[lo, hi)`` / ``{letter}_lo+`` format as the preprocessor so
    that unregistered variables are displayed consistently.
    """
    arr = s.cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy(allow_copy=True)
    is_missing = np.isnan(arr) | np.isclose(arr, MISSING_SENTINEL, rtol=0, atol=1.0)
    valid = arr[~is_missing]

    if len(valid) == 0:
        return pl.Series(["Missing"] * len(arr))

    full_edges = compute_quantile_bin_edges(valid, n_bins)
    breaks = full_edges[1:-1]
    all_labels = make_bin_labels(breaks)
    s_float = pl.Series(s.name, arr).set(pl.Series(is_missing), None)
    labeled = s_float.cut(
        list(breaks), labels=all_labels, left_closed=True
    ).cast(pl.Utf8).fill_null("Missing")
    return labeled.rename(s.name + "_bin")


def _resolve_level(
    col: str,
    X: pl.DataFrame,
    preprocessor: Optional[Preprocessor],
    n_bins: int,
) -> pl.Series:
    """
    Determine the level series for *col* in *X*.

    If *preprocessor* has a config for *col*, delegate to
    ``preprocessor.get_level_labels`` which handles binned numeric,
    categorical (with custom remap), and multi-input derived variables —
    all using the same labels as ``relativities_table``.

    Otherwise falls back to direct cast for categorical / low-cardinality
    columns and ``_bin_for_plot`` for continuous variables.
    """
    if preprocessor is not None and col in preprocessor.configs:
        p = preprocessor._params.get(col, {})
        if "bin_edges" in p or p.get("is_categorical"):
            return preprocessor.get_level_labels(col, X)
        # Continuous non-binned (including polynomial) — bin on the fly
    s = X[col]
    if _is_str_or_cat(s.dtype) or s.n_unique() <= 20:
        return s.cast(pl.Utf8).fill_null("Missing")
    return _bin_for_plot(s, n_bins)


def _sort_labels(labels: list) -> list:
    """
    Sort bin/category labels numerically by their lower bound.

    Handles the ``{LETTER(S)}_{range}`` format produced by ``make_bin_labels``
    (including multi-letter prefixes like ``AA_``) and puts ``'Missing'`` last.
    Falls back to lexicographic sort if no numeric value can be extracted.
    """
    import re as _re

    def _key(x: str) -> float:
        if x in ("Missing", "__MISSING__"):
            return float("inf")
        # Strip leading letter prefix produced by _bin_letter (A_, AA_, etc.)
        body = _re.sub(r"^[A-Z]+_", "", x)
        # Extract first numeric value from the lower bound
        m = _re.search(r"-?[\d.]+", body.split(",")[0])
        return float(m.group()) if m else float("inf")

    try:
        return sorted(labels, key=_key)
    except Exception:
        return sorted(labels)


# ── Combo bar + line chart ────────────────────────────────────────────────────

def _bar_with_line(
    ax_bar: plt.Axes,
    x_labels: list,
    bar_vals: np.ndarray,
    line_vals: np.ndarray,
    bar_label: str = "Exposure",
    line_label: str = "Relativity",
    bar_color: str = "#9ecae1",
    line_color: str = "#e6550d",
    rotation: int = 45,
    ref_line: Optional[float] = None,
) -> plt.Axes:
    """Seaborn bar + twinx line chart. Returns the secondary (line) Axes."""
    x = np.arange(len(x_labels))
    sns.barplot(x=x_labels, y=bar_vals, ax=ax_bar, color=bar_color, alpha=0.8,
                label=bar_label, zorder=2, errorbar=None)
    ax_bar.set_ylabel(bar_label, fontsize=9)
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.setp(ax_bar.get_xticklabels(), rotation=rotation, ha="right", fontsize=8)

    ax2 = ax_bar.twinx()
    # Use numeric x positions so the line aligns with seaborn's categorical bars.
    ax2.plot(x, line_vals, color=line_color, marker="o", linewidth=2,
             markersize=5, label=line_label, zorder=3)
    if ref_line is not None:
        ax2.axhline(ref_line, color="black", linewidth=1.0, linestyle="--",
                    alpha=0.55, zorder=2)
    ax2.set_ylabel(line_label, fontsize=9, color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)

    h1, l1 = ax_bar.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_bar.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    return ax2


# ── Aggregate helper ──────────────────────────────────────────────────────────

def _weighted_agg(
    level: pl.Series,
    y: pl.Series,
    weight: pl.Series,
    extra_series: Optional[Dict[str, pl.Series]] = None,
) -> pl.DataFrame:
    """
    Compute weighted mean of *y* (and optionally other series) by *level*.

    Returns a sorted DataFrame with columns:
    ``_level``, ``mean_y``, ``exposure``, plus one ``mean_<k>`` per extra.
    """
    df = pl.DataFrame({"_level": level, "_y": y, "_w": weight})
    agg_exprs = [
        ((pl.col("_y") * pl.col("_w")).sum() / pl.col("_w").sum()).alias("mean_y"),
        pl.col("_w").sum().alias("exposure"),
    ]

    if extra_series:
        for k, s in extra_series.items():
            df = df.with_columns(s.alias(f"_extra_{k}"))
            agg_exprs.append(
                ((pl.col(f"_extra_{k}") * pl.col("_w")).sum() / pl.col("_w").sum()
                 ).alias(f"mean_{k}")
            )

    summary = df.group_by("_level").agg(agg_exprs)

    # Sort by natural order of labels
    labels = _sort_labels(summary["_level"].to_list())
    label_order = pl.DataFrame({"_level": labels, "_order": list(range(len(labels)))})
    summary = summary.join(label_order, on="_level").sort("_order").drop("_order")
    return summary


# ── Univariate plot ───────────────────────────────────────────────────────────

def univariate_plot(
    X: pl.DataFrame,
    y: pl.Series,
    col: str,
    weights: Optional[pl.Series] = None,
    n_bins: int = 10,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    preprocessor: Optional[Preprocessor] = None,
) -> plt.Figure:
    """
    Univariate view of ``col`` vs the target ``y``.

    Continuous variables are binned into ``n_bins`` quantile buckets.
    Shows exposure (bar) and the weighted mean loss ratio as a **relativity
    to the overall weighted mean** (line). A value of 1.0 means the level
    is exactly at the portfolio average.

    Pass *preprocessor* to use fitted bin edges/labels (consistent with
    ``relativities_table``) instead of re-binning the raw column.
    """
    w = weights if weights is not None else pl.Series("w", np.ones(len(X)))
    level = _resolve_level(col, X, preprocessor, n_bins)

    summary = _weighted_agg(level, y, w)

    # Convert to relativities: each level mean / overall weighted mean.
    overall_mean = float((y.cast(pl.Float64) * w).sum() / w.sum())
    rel_vals = summary["mean_y"].to_numpy() / overall_mean

    fig, ax = plt.subplots(figsize=figsize or _DEFAULT_FIG)
    ax.set_title(title or f"Univariate: {col}", fontsize=12, fontweight="bold")
    _bar_with_line(
        ax,
        x_labels=summary["_level"].to_list(),
        bar_vals=summary["exposure"].to_numpy(),
        line_vals=rel_vals,
        bar_label="Exposure",
        line_label="Relativity to Mean",
        ref_line=1.0,
    )
    fig.tight_layout()
    return fig


# ── Actual vs Expected chart ──────────────────────────────────────────────────

def ae_chart(
    X: pl.DataFrame,
    y: pl.Series,
    col: str,
    predictions: np.ndarray,
    weights: Optional[pl.Series] = None,
    n_bins: int = 10,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    version_name: str = "Model",
    preprocessor: Optional[Preprocessor] = None,
) -> plt.Figure:
    """
    Actual vs Expected chart for ``col`` using the supplied model predictions.

    Both actual and predicted lines are expressed as **relativities to their
    respective overall weighted means**, so 1.0 always represents the portfolio
    average for each series. This makes vertical differences between the two
    lines directly interpretable as model mis-specification.

    Works for variables not in the model.  Continuous variables are binned.
    Pass *preprocessor* to use fitted bin edges/labels instead of re-binning.
    """
    w = weights if weights is not None else pl.Series("w", np.ones(len(X)))
    w_np = w.to_numpy().astype(float)
    pred_s = pl.Series("_pred", predictions)
    level = _resolve_level(col, X, preprocessor, n_bins)

    summary = _weighted_agg(level, y, w, extra_series={"pred": pred_s})

    # Convert both series to relativities vs their respective portfolio means.
    overall_actual = float((y.cast(pl.Float64) * w).sum() / w.sum())
    overall_pred = float((predictions * w_np).sum() / w_np.sum())
    actual_rel = summary["mean_y"].to_numpy() / overall_actual
    pred_rel = summary["mean_pred"].to_numpy() / overall_pred

    x_labels = summary["_level"].to_list()
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=figsize or _DEFAULT_FIG)
    ax.set_title(title or f"Actual vs Expected: {col}", fontsize=12, fontweight="bold")

    sns.barplot(x=x_labels, y=summary["exposure"].to_numpy(), ax=ax,
                color="#9ecae1", alpha=0.7, label="Exposure", zorder=2, errorbar=None)
    ax.set_ylabel("Exposure", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, actual_rel, color="#e6550d", marker="o",
             linewidth=2, markersize=6, label="Actual", zorder=3)
    ax2.plot(x, pred_rel, color="#31a354", marker="s",
             linewidth=2, markersize=6, linestyle="--",
             label=f"Expected ({version_name})", zorder=3)
    ax2.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55, zorder=2)
    ax2.set_ylabel("Relativity to Mean", fontsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ── Residual signal chart ─────────────────────────────────────────────────────

def residual_chart(
    X: pl.DataFrame,
    y: pl.Series,
    col: str,
    predictions: np.ndarray,
    weights: Optional[pl.Series] = None,
    n_bins: int = 10,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    version_name: str = "Model",
    preprocessor: Optional[Preprocessor] = None,
) -> plt.Figure:
    """
    Residual signal chart: ``mean_actual / mean_predicted`` per variable level.

    For each level (or quantile bin) of ``col`` the chart shows the ratio of
    the weighted-mean actual loss ratio to the weighted-mean model prediction.
    A value above 1.0 means the model is under-predicting for that group;
    below 1.0 means over-predicting.  Exposure is shown as bars on a
    secondary y-axis.

    This is one step beyond an A/E chart: rather than plotting actual and
    predicted side-by-side, it plots their ratio so residual signal is
    immediately visible as deviations from the reference line at 1.0.

    Parameters
    ----------
    col : str
        Variable to slice by.  Does not need to be a model predictor.
    predictions : np.ndarray
        Model predictions aligned row-for-row with *X* and *y*.
    n_bins : int
        Number of quantile bins for continuous variables.
    version_name : str
        Label shown in the chart title.
    preprocessor : Preprocessor, optional
        When supplied, bin edges fitted for *col* are used for label
        assignment instead of re-binning the raw column.
    """
    w = weights if weights is not None else pl.Series("w", np.ones(len(X)))
    pred_s = pl.Series("_pred", predictions)
    level = _resolve_level(col, X, preprocessor, n_bins)

    summary = _weighted_agg(level, y, w, extra_series={"pred": pred_s})

    # Residual = mean_actual / mean_predicted; guard against zero predicted.
    # This ratio is already a relativity-style metric — bases cancel, no
    # further normalisation needed.
    mean_y = summary["mean_y"].to_numpy()
    mean_pred = summary["mean_pred"].to_numpy()
    residual = np.where(np.abs(mean_pred) > 1e-12, mean_y / mean_pred, np.nan)
    exposure = summary["exposure"].to_numpy()
    x_labels = summary["_level"].to_list()
    x = np.arange(len(x_labels))

    fig, ax_bar = plt.subplots(figsize=figsize or _DEFAULT_FIG)
    ax_bar.set_title(
        title or f"Residual Signal: {col}  |  {version_name}",
        fontsize=12, fontweight="bold",
    )

    # Exposure bars (primary axis)
    sns.barplot(x=x_labels, y=exposure, ax=ax_bar,
                color="#9ecae1", alpha=0.7, label="Exposure", zorder=2, errorbar=None)
    ax_bar.set_ylabel("Exposure", fontsize=9)
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # Residual ratio line (secondary axis)
    ax2 = ax_bar.twinx()
    ax2.plot(x, residual, color="#e6550d", marker="o", linewidth=2,
             markersize=6, label="Actual / Predicted", zorder=3)
    ax2.axhline(1.0, color="black", linewidth=1.0, linestyle="--",
                alpha=0.55, label="1.0 reference", zorder=2)
    ax2.set_ylabel("Actual / Predicted", fontsize=9, color="#e6550d")
    ax2.tick_params(axis="y", labelcolor="#e6550d")

    h1, l1 = ax_bar.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_bar.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


# ── Double lift chart ─────────────────────────────────────────────────────────

def double_lift_chart(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_buckets: int = 10,
    name1: str = "Model 1",
    name2: str = "Model 2",
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Double lift chart comparing two models using equal-weight buckets.

    Sorted by pred1/pred2 ratio; each bucket has equal total exposure.
    """
    tbl = double_lift_table(y_true, pred1, pred2, weights=weights, n_buckets=n_buckets)
    x = np.arange(len(tbl))
    xlabels = [str(b) for b in tbl["bucket"].to_list()]

    fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 5))

    ax = axes[0]
    ax.plot(x, tbl["actual"].to_numpy(), color="#e6550d", marker="o",
            linewidth=2, markersize=6, label="Actual")
    ax.plot(x, tbl["model1"].to_numpy(), color="#3182bd", marker="s",
            linewidth=2, markersize=5, linestyle="--", label=name1)
    ax.plot(x, tbl["model2"].to_numpy(), color="#31a354", marker="^",
            linewidth=2, markersize=5, linestyle=":", label=name2)
    ax.set_title("Actual vs Both Models", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Bucket (sorted by {name1}/{name2} ratio, equal weight)")
    ax.set_ylabel("Weighted Mean Loss Ratio")
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(x, tbl["exposure"].to_numpy(), color="#9ecae1", alpha=0.8)
    ax2.set_title("Bucket Exposure", fontsize=11, fontweight="bold")
    ax2.set_xlabel(f"Bucket (sorted by {name1}/{name2} ratio)")
    ax2.set_ylabel("Exposure")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Double Lift: {name1} vs {name2}", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── Lorenz / Gini chart ───────────────────────────────────────────────────────

def lorenz_chart(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    weights: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Lorenz curve for one or more model versions.

    Parameters
    ----------
    predictions_dict : dict
        ``{version_name: prediction_array}``
    """
    y_true = np.asarray(y_true, dtype=float)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=figsize or (7, 6))
    for i, (name, preds) in enumerate(predictions_dict.items()):
        preds = np.asarray(preds, dtype=float)
        idx = np.argsort(-preds)
        ys, ws = y_true[idx], w[idx]
        cum_w = np.concatenate([[0.0], np.cumsum(ws) / ws.sum()])
        total_loss = (ys * ws).sum()
        cum_l = np.concatenate([[0.0], np.cumsum(ys * ws) / (total_loss or 1)])
        gini = gini_coefficient(y_true, preds, w, normalize=True)
        ax.plot(cum_w, cum_l, color=colors[i % 10], linewidth=2,
                label=f"{name} (Gini={gini:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random")
    ax.set_xlabel("Cumulative Exposure Fraction", fontsize=10)
    ax.set_ylabel("Cumulative Loss Fraction", fontsize=10)
    ax.set_title("Lorenz Curve / Gini Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── Coefficient plot ──────────────────────────────────────────────────────────

def coefficient_plot(
    coef_df: pl.DataFrame,
    version_name: str = "Model",
    top_n: int = 30,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of model coefficients sorted by absolute value.

    Parameters
    ----------
    coef_df : pl.DataFrame
        Two columns: ``'feature'`` and ``'coefficient'``.
    """
    df = (
        coef_df
        .filter(pl.col("feature") != "intercept")
        .with_columns(pl.col("coefficient").abs().alias("_abs"))
        .sort("_abs", descending=False)
        .tail(top_n)
    )

    features = df["feature"].to_list()
    values = df["coefficient"].to_numpy()
    colors = ["#e6550d" if v > 0 else "#3182bd" for v in values]

    fig, ax = plt.subplots(figsize=figsize or (9, max(4, len(features) * 0.35)))
    ax.barh(features, values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value", fontsize=10)
    ax.set_title(
        f"Coefficients — {version_name} (top {len(features)} by |coef|)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ── CV stability plot ─────────────────────────────────────────────────────────

def cv_stability_plot(
    stability_df: pl.DataFrame,
    top_n: int = 20,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Box plot of coefficient values across CV folds.

    Parameters
    ----------
    stability_df : pl.DataFrame
        Output of :func:`~elastic_net_tool.model.fit_cv_stability`.
        Must contain a ``'fold'`` column with numeric fold labels.
    """
    # Only use fold rows (exclude geomean / std / cv_pct summary rows)
    fold_rows = stability_df.filter(
        ~pl.col("fold").is_in(["geomean", "std", "cv_pct"])
    )
    if fold_rows.is_empty():
        raise ValueError("stability_df contains no fold rows.")

    feat_cols = [c for c in fold_rows.columns if c != "fold"]
    means_abs = {
        c: float(fold_rows[c].cast(pl.Float64).abs().mean() or 0.0)
        for c in feat_cols
    }
    top_features = sorted(means_abs, key=means_abs.get, reverse=True)[:top_n]  # type: ignore[arg-type]

    data = [fold_rows[f].cast(pl.Float64).drop_nulls().to_numpy() for f in top_features]

    fig, ax = plt.subplots(figsize=figsize or (max(8, len(top_features) * 0.6), 5))
    # tick labels are applied via set_xticklabels below (boxplot's labels=
    # kwarg was removed in matplotlib 3.11)
    ax.boxplot(
        data, patch_artist=True,
        boxprops=dict(facecolor="#9ecae1", alpha=0.8),
        medianprops=dict(color="#e6550d", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient Value", fontsize=10)
    ax.set_title(
        f"CV Coefficient Stability — top {len(top_features)} features",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Metrics bar chart ─────────────────────────────────────────────────────────

def metrics_bar_chart(
    metrics_df: pl.DataFrame,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Bar chart comparing key metrics across model versions.

    Parameters
    ----------
    metrics_df : pl.DataFrame
        Output of :func:`~elastic_net_tool.metrics.compare_metrics`.
        First column is ``'metric'``.
    """
    metric_col = metrics_df.columns[0]
    version_cols = [c for c in metrics_df.columns[1:] if c != "winner"]
    show = [m for m in ["rmse", "mae", "gini_norm"]
            if m in metrics_df[metric_col].to_list()]

    n = len(show)
    if n == 0:
        raise ValueError("No recognised metrics found in metrics_df.")

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, n, figsize=figsize or (4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, show):
        row = metrics_df.filter(pl.col(metric_col) == metric)
        vals = [float(row[v][0]) for v in version_cols]
        bars = ax.bar(list(version_cols), vals, color=colors[: len(version_cols)], alpha=0.85)
        ax.set_title(metric.upper(), fontsize=10, fontweight="bold")
        ymax = max(vals) * 1.2 if max(vals) > 0 else 1.0
        ax.set_ylim(0, ymax)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
            )
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Comparison Metrics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Interaction heatmap ──────────────────────────────────────────────────────

def interaction_heatmap(
    ranking_df: pl.DataFrame,
    top_n: int = 15,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Symmetric matrix heatmap of H-statistics from :func:`interaction_ranking`.

    Parameters
    ----------
    ranking_df : pl.DataFrame
        Output of ``discovery.interaction_ranking()``.
        Columns: ``var1``, ``var2``, ``h_statistic``.
    top_n : int
        Show only the top ``top_n`` variables (by max H-statistic involvement).
    """
    # Collect all variables involved in top pairs
    df = ranking_df.head(top_n * (top_n - 1) // 2)
    all_vars = sorted(
        set(df["var1"].to_list()) | set(df["var2"].to_list())
    )
    if len(all_vars) > top_n:
        all_vars = all_vars[:top_n]

    n = len(all_vars)
    var_idx = {v: i for i, v in enumerate(all_vars)}
    matrix = np.zeros((n, n))

    for row in df.iter_rows(named=True):
        v1, v2, h = row["var1"], row["var2"], row["h_statistic"]
        if v1 in var_idx and v2 in var_idx:
            i, j = var_idx[v1], var_idx[v2]
            matrix[i, j] = h
            matrix[j, i] = h

    fig, ax = plt.subplots(figsize=figsize or (max(8, n * 0.6), max(6, n * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_vars, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_vars, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, label="H-statistic", shrink=0.8)
    ax.set_title("Interaction Strength (H-statistic)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 2D partial dependence plot ───────────────────────────────────────────────

def pd_plot_2d(
    pd_data: pl.DataFrame,
    var1_name: str,
    var2_name: str,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Contour/heatmap of 2D partial dependence.

    Parameters
    ----------
    pd_data : pl.DataFrame
        Output of ``discovery.partial_dependence_2d()``.
        Columns: ``var1_value``, ``var2_value``, ``pd_value``.
    """
    v1 = np.sort(pd_data["var1_value"].unique().to_numpy())
    v2 = np.sort(pd_data["var2_value"].unique().to_numpy())

    # Build grid matrix
    grid = np.full((len(v2), len(v1)), np.nan)
    v1_idx = {v: i for i, v in enumerate(v1)}
    v2_idx = {v: i for i, v in enumerate(v2)}

    for row in pd_data.iter_rows(named=True):
        i = v1_idx.get(row["var1_value"])
        j = v2_idx.get(row["var2_value"])
        if i is not None and j is not None:
            grid[j, i] = row["pd_value"]

    fig, ax = plt.subplots(figsize=figsize or (9, 7))
    im = ax.contourf(v1, v2, grid, levels=20, cmap="RdYlBu_r")
    ax.contour(v1, v2, grid, levels=20, colors="black", linewidths=0.3, alpha=0.3)
    fig.colorbar(im, ax=ax, label="Partial Dependence", shrink=0.8)
    ax.set_xlabel(var1_name, fontsize=10)
    ax.set_ylabel(var2_name, fontsize=10)
    ax.set_title(f"2D Partial Dependence: {var1_name} x {var2_name}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Permutation importance plot ──────────────────────────────────────────────

def importance_plot(
    importance_df: pl.DataFrame,
    top_n: int = 20,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of permutation importance with error bars.

    Parameters
    ----------
    importance_df : pl.DataFrame
        Output of ``discovery.permutation_importance()``.
        Columns: ``variable``, ``importance_mean``, ``importance_std``.
    """
    df = importance_df.head(top_n).sort("importance_mean", descending=False)
    variables = df["variable"].to_list()
    means = df["importance_mean"].to_numpy()
    stds = df["importance_std"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize or (9, max(4, len(variables) * 0.35)))
    ax.barh(variables, means, xerr=stds, color="#3182bd", alpha=0.85,
            capsize=3, ecolor="#333333")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Importance (metric degradation when shuffled)", fontsize=10)
    ax.set_title(title or f"Permutation Importance — top {len(variables)}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Residual heatmap (2D A/E) ───────────────────────────────────────────────

def residual_heatmap(
    X: pl.DataFrame,
    y: pl.Series,
    col1: str,
    col2: str,
    predictions: np.ndarray,
    weights: Optional[pl.Series] = None,
    preprocessor: Optional[Preprocessor] = None,
    n_bins: int = 8,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, pl.DataFrame]:
    """
    2D residual heatmap: actual/expected ratio across two variable dimensions.

    Each cell shows the weighted mean A/E ratio and exposure volume.
    Cells deviating from 1.0 with meaningful exposure reveal interactions.

    Parameters
    ----------
    col1, col2 : str
        Variables for the two axes.
    predictions : np.ndarray
        Model predictions aligned with rows of *X*.
    preprocessor : Preprocessor, optional
        Use fitted bin edges for level assignment.
    n_bins : int
        Quantile bins if preprocessor not available.

    Returns
    -------
    tuple of (Figure, pl.DataFrame)
        The heatmap figure and underlying data table.
    """
    w = weights if weights is not None else pl.Series("w", np.ones(len(X)))
    pred_s = pl.Series("_pred", predictions)

    level1 = _resolve_level(col1, X, preprocessor, n_bins)
    level2 = _resolve_level(col2, X, preprocessor, n_bins)

    # Build working DataFrame
    work = pl.DataFrame({
        "_level1": level1,
        "_level2": level2,
        "_actual": y,
        "_pred": pred_s,
        "_w": w,
    })

    # Compute weighted A/E per cell
    summary = work.group_by(["_level1", "_level2"]).agg([
        ((pl.col("_actual") * pl.col("_w")).sum() / pl.col("_w").sum()).alias("actual_mean"),
        ((pl.col("_pred") * pl.col("_w")).sum() / pl.col("_w").sum()).alias("pred_mean"),
        pl.col("_w").sum().alias("exposure"),
    ]).with_columns(
        (pl.col("actual_mean") / pl.col("pred_mean")).alias("ae_ratio"),
    )

    # Build matrix
    labels1 = _sort_labels(summary["_level1"].unique().to_list())
    labels2 = _sort_labels(summary["_level2"].unique().to_list())
    idx1 = {v: i for i, v in enumerate(labels1)}
    idx2 = {v: i for i, v in enumerate(labels2)}

    ae_matrix = np.full((len(labels2), len(labels1)), np.nan)
    exp_matrix = np.full((len(labels2), len(labels1)), 0.0)

    for row in summary.iter_rows(named=True):
        i = idx1.get(row["_level1"])
        j = idx2.get(row["_level2"])
        if i is not None and j is not None:
            ae_matrix[j, i] = row["ae_ratio"]
            exp_matrix[j, i] = row["exposure"]

    # Plot
    fig, ax = plt.subplots(figsize=figsize or (max(8, len(labels1) * 0.9),
                                                max(6, len(labels2) * 0.8)))

    # Build custom annotation matrix: "A/E\n(exposure)" per cell.
    annot_matrix = np.empty((len(labels2), len(labels1)), dtype=object)
    for i in range(len(labels2)):
        for j in range(len(labels1)):
            if not np.isnan(ae_matrix[i, j]):
                annot_matrix[i, j] = f"{ae_matrix[i, j]:.2f}\n({exp_matrix[i, j]:,.0f})"
            else:
                annot_matrix[i, j] = ""

    valid_vals = ae_matrix[~np.isnan(ae_matrix)]
    vmin = max(0.5, valid_vals.min() - 0.05) if len(valid_vals) > 0 else 0.5
    vmax = min(2.0, valid_vals.max() + 0.05) if len(valid_vals) > 0 else 1.5

    sns.heatmap(
        ae_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="RdBu_r",
        center=1.0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=labels1,
        yticklabels=labels2,
        linewidths=0.4,
        linecolor="lightgray",
        annot_kws={"size": 7},
        cbar_kws={"label": "A/E Ratio (1.0 = perfect)", "shrink": 0.8},
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel(col1, fontsize=10)
    ax.set_ylabel(col2, fontsize=10)
    ax.set_title(title or f"2D Residual Heatmap: {col1} x {col2}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    # Return clean table
    result_df = summary.rename({"_level1": col1, "_level2": col2})
    return fig, result_df


# ── Regularization path plot ────────────────────────────────────────────────

def regularization_path_plot(
    path_df: pl.DataFrame,
    top_n: int = 20,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot coefficient evolution across regularization strengths.

    Parameters
    ----------
    path_df : pl.DataFrame
        Columns: ``alpha``, ``variable``, ``coefficient``.
    top_n : int
        Show only the top variables (by max absolute coefficient).
    """
    # Find top variables by max absolute coefficient across all alphas
    max_abs = (
        path_df
        .group_by("variable")
        .agg(pl.col("coefficient").abs().max().alias("max_abs"))
        .sort("max_abs", descending=True)
        .head(top_n)
    )
    top_vars = set(max_abs["variable"].to_list())
    df = path_df.filter(pl.col("variable").is_in(top_vars))

    fig, ax = plt.subplots(figsize=figsize or (11, 6))
    colors = plt.cm.tab20.colors  # type: ignore[attr-defined]

    for i, var in enumerate(sorted(top_vars)):
        var_df = df.filter(pl.col("variable") == var).sort("alpha")
        ax.plot(
            var_df["alpha"].to_numpy(),
            var_df["coefficient"].to_numpy(),
            color=colors[i % 20],
            linewidth=1.5,
            label=var,
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Alpha (regularization strength)", fontsize=10)
    ax.set_ylabel("Coefficient", fontsize=10)
    ax.set_title("Regularization Path", fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── Overfitting monitor plot ────────────────────────────────────────────────

def overfitting_plot(
    monitor_df: pl.DataFrame,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Train vs CV metric across model versions.

    Parameters
    ----------
    monitor_df : pl.DataFrame
        Columns: ``step``, ``n_variables``, ``variables_added``,
        ``train_metric``, ``cv_metric``, ``gap``.
    """
    steps = monitor_df["step"].to_numpy()
    train = monitor_df["train_metric"].to_numpy()
    cv = monitor_df["cv_metric"].to_numpy()
    labels = monitor_df["variables_added"].to_list()

    fig, ax = plt.subplots(figsize=figsize or (10, 5))
    ax.plot(steps, train, color="#3182bd", marker="o", linewidth=2,
            markersize=6, label="Train")
    ax.plot(steps, cv, color="#e6550d", marker="s", linewidth=2,
            markersize=6, label="CV")
    ax.fill_between(steps, train, cv, alpha=0.15, color="#e6550d",
                     label="Gap (overfitting)")

    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Variables Added", fontsize=10)
    ax.set_ylabel("Metric", fontsize=10)
    ax.set_title("Overfitting Monitor: Train vs CV", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── Bootstrap CI plot ────────────────────────────────────────────────────────

def bootstrap_ci_plot(
    bootstrap_df: pl.DataFrame,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Point estimates with confidence interval error bars.

    Parameters
    ----------
    bootstrap_df : pl.DataFrame
        Columns: ``metric``, ``point_estimate``, ``ci_lower``, ``ci_upper``.
    """
    metrics = bootstrap_df["metric"].to_list()
    points = bootstrap_df["point_estimate"].to_numpy()
    lowers = bootstrap_df["ci_lower"].to_numpy()
    uppers = bootstrap_df["ci_upper"].to_numpy()

    errors = np.array([points - lowers, uppers - points])

    fig, ax = plt.subplots(figsize=figsize or (8, max(3, len(metrics) * 0.5)))
    ax.barh(metrics, points, xerr=errors, color="#3182bd", alpha=0.85,
            capsize=5, ecolor="#333333")
    ax.set_xlabel("Value", fontsize=10)
    ax.set_title(title or "Bootstrap Confidence Intervals", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Relativities CI plot ────────────────────────────────────────────────────

def relativities_ci_plot(
    relativities_df: pl.DataFrame,
    variable: str,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Relativity per level with bootstrap confidence interval error bars.

    Parameters
    ----------
    relativities_df : pl.DataFrame
        Columns: ``variable``, ``level``, ``relativity``, ``ci_lower``, ``ci_upper``.
    variable : str
        Variable to plot.
    """
    df = relativities_df.filter(pl.col("variable") == variable)
    levels = df["level"].to_list()
    rels = df["relativity"].to_numpy()
    lowers = df["ci_lower"].to_numpy()
    uppers = df["ci_upper"].to_numpy()

    errors = np.array([rels - lowers, uppers - rels])

    fig, ax = plt.subplots(figsize=figsize or (9, max(4, len(levels) * 0.4)))
    x = np.arange(len(levels))
    ax.bar(x, rels, yerr=errors, color="#3182bd", alpha=0.85,
           capsize=4, ecolor="#333333")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="Base (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relativity (exp(coefficient))", fontsize=10)
    ax.set_title(f"Bootstrap Relativity CIs — {variable}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
