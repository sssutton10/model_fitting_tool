"""
bin_suggestor.py
================
Utilities for suggesting breakpoints for continuous variables.

Four strategies are available:

* ``"quantile"``    – equal-weight (exposure-weighted) quantile splits
* ``"equal_width"`` – equal-width splits spanning [min, max]
* ``"optbin"``      – OptimalBinning via MILP/CP-SAT (``optbinning`` package)
* ``"gbm"``         – Most-used split thresholds from a single-feature GBM
                      (LightGBM when installed, sklearn GBM as fallback)

None of these functions modify any variable configuration; they only print
and return suggested breakpoints for the user to decide on.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

from .variable import MISSING_SENTINEL


# ── Internal helpers ──────────────────────────────────────────────────────────

_COLORS = ["#e6550d", "#31a354", "#3182bd", "#756bb1"]
_LINESTYLES = ["--", "-.", ":", "-"]


def _drop_sentinel(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Remove rows where the *first* array equals MISSING_SENTINEL."""
    mask = arrays[0] != MISSING_SENTINEL
    return tuple(a[mask] for a in arrays)


def _weighted_quantiles(
    arr: np.ndarray, w: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Compute weighted quantiles of *arr* at fractions *quantiles*."""
    idx = np.argsort(arr)
    arr_s, w_s = arr[idx], w[idx]
    cum_w = np.cumsum(w_s) / w_s.sum()
    return np.interp(quantiles, cum_w, arr_s)


def _print_splits(method: str, col: str, splits: List[float]) -> None:
    """Pretty-print a single method's splits."""
    width = 58
    print(f"\n  +- {method}")
    print(f"  |  Variable   : {col}")
    if splits:
        body = ", ".join(f"{s:.6g}" for s in splits)
        print(f"  |  Splits ({len(splits):2d}) : [{body}]")
    else:
        print("  |  Splits     : (none found)")
    print(f"  +{'-' * width}")


# ── Method 1: Equal-weight quantile ──────────────────────────────────────────

def suggest_bins_quantile(
    col: str,
    X: pl.DataFrame,
    n_bins: int = 10,
    weights: Optional[pl.Series] = None,
    verbose: bool = True,
) -> List[float]:
    """
    Equal-weight (exposure-weighted) quantile breakpoints.

    Splits are placed at the interior boundaries of ``n_bins`` groups, each
    containing approximately the same total weight.  Rows with the missing
    sentinel value are excluded before computing quantiles.

    Parameters
    ----------
    n_bins : int
        Target number of bins → produces ``n_bins - 1`` interior split points.
    weights : pl.Series, optional
        Exposure or frequency weight for each row.
    """
    arr = X[col].to_numpy().astype(float)
    w = (
        weights.to_numpy().astype(float)
        if weights is not None
        else np.ones(len(arr))
    )
    arr, w = _drop_sentinel(arr, w)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    raw = _weighted_quantiles(arr, w, quantiles)

    # Deduplicate while preserving order
    seen: set = set()
    splits: List[float] = []
    for v in raw:
        v = float(v)
        if v not in seen:
            seen.add(v)
            splits.append(v)

    if verbose:
        _print_splits("Equal-weight quantile", col, splits)
    return splits


# ── Method 2: Equal-width ────────────────────────────────────────────────────

def suggest_bins_equal_width(
    col: str,
    X: pl.DataFrame,
    n_bins: int = 10,
    verbose: bool = True,
) -> List[float]:
    """
    Equal-width breakpoints spanning ``[min, max]``.

    Simple and fast but ignores the distribution of exposure across the range.
    Useful as a naive baseline to compare other methods against.

    Parameters
    ----------
    n_bins : int
        Target number of bins → produces ``n_bins - 1`` interior split points.
    """
    arr = X[col].to_numpy().astype(float)
    arr = arr[arr != MISSING_SENTINEL]
    lo, hi = float(arr.min()), float(arr.max())
    splits = [float(v) for v in np.linspace(lo, hi, n_bins + 1)[1:-1]]

    if verbose:
        _print_splits("Equal-width", col, splits)
    return splits


# ── Method 3: OptimalBinning ─────────────────────────────────────────────────

def suggest_bins_optbin(
    col: str,
    X: pl.DataFrame,
    y: pl.Series,
    weights: Optional[pl.Series] = None,
    verbose: bool = True,
    **optbin_kwargs: Any,
) -> List[float]:
    """
    Optimal breakpoints via ``optbinning``.

    Uses a MILP / CP-SAT solver to find splits that maximise the statistical
    divergence between adjacent bins while enforcing monotonicity and minimum
    bin-size constraints.  A binary target (≤ 2 unique values) uses
    ``OptimalBinning``; a continuous target (the usual case for loss ratios)
    uses ``ContinuousOptimalBinning``.

    Extra keyword arguments are forwarded directly to the binning class
    ``__init__`` (e.g. ``max_n_bins``, ``min_bin_size``, ``monotonic_trend``).

    Requires
    --------
    ``pip install optbinning``
    """
    try:
        from optbinning import ContinuousOptimalBinning, OptimalBinning
    except ImportError as exc:
        raise ImportError(
            "The 'optbinning' package is required for this method.\n"
            "Install it with:  pip install optbinning"
        ) from exc

    arr = X[col].to_numpy().astype(float)
    y_arr = y.to_numpy().astype(float)
    w = weights.to_numpy().astype(float) if weights is not None else None

    if w is not None:
        arr, y_arr, w = _drop_sentinel(arr, y_arr, w)
    else:
        arr, y_arr = _drop_sentinel(arr, y_arr)

    kwargs: Dict[str, Any] = {"name": col, "dtype": "numerical"}
    kwargs.update(optbin_kwargs)

    binning_cls = OptimalBinning if np.unique(y_arr).size <= 2 else ContinuousOptimalBinning
    ob = binning_cls(**kwargs)
    ob.fit(arr, y_arr, sample_weight=w)

    splits = sorted(
        float(s) for s in ob.splits if np.isfinite(float(s))
    )

    if verbose:
        _print_splits("OptimalBinning (optbinning)", col, splits)
    return splits


# ── Method 4: GBM splits ─────────────────────────────────────────────────────

def suggest_bins_gbm(
    col: str,
    X: pl.DataFrame,
    y: pl.Series,
    weights: Optional[pl.Series] = None,
    n_estimators: int = 100,
    max_depth: int = 3,
    max_splits: int = 20,
    verbose: bool = True,
    **gbm_kwargs: Any,
) -> List[float]:
    """
    Breakpoints extracted from a single-feature GBM.

    A shallow gradient-boosted regressor is fit using only ``col`` as a
    predictor.  Every unique split threshold is collected across the entire
    ensemble, and the ``max_splits`` thresholds that appear *most frequently*
    across all trees are returned — frequency is used as a proxy for
    importance, so the most influential decision boundaries surface first.

    **LightGBM** is used when available;
    ``sklearn.ensemble.GradientBoostingRegressor`` is the fallback.

    Extra keyword arguments are forwarded to the GBM constructor (e.g.
    ``learning_rate``, ``subsample``).

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum depth of each individual tree.
    max_splits : int
        Maximum number of split thresholds to return, selected by frequency.
    """
    arr = X[col].to_numpy().astype(float)
    y_arr = y.to_numpy().astype(float)
    w = weights.to_numpy().astype(float) if weights is not None else None

    if w is not None:
        arr, y_arr, w = _drop_sentinel(arr, y_arr, w)
    else:
        arr, y_arr = _drop_sentinel(arr, y_arr)

    arr2d = arr.reshape(-1, 1)
    counter: Counter = Counter()

    # ── LightGBM ─────────────────────────────────────────────────────────────
    try:
        import lightgbm as lgb

        params: Dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": max(2, 2 ** max_depth - 1),
            "verbose": -1,
            "n_jobs": 1,
        }
        params.update(gbm_kwargs)

        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(arr2d, y_arr, sample_weight=w, feature_name=[col])

        trees_df = mdl.booster_.trees_to_dataframe()
        for val in trees_df["threshold"].dropna():
            try:
                counter[float(val)] += 1
            except (ValueError, TypeError):
                pass

        backend = "LightGBM"

    # ── sklearn GBM ──────────────────────────────────────────────────────────
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor

        params = {"n_estimators": n_estimators, "max_depth": max_depth}
        params.update(gbm_kwargs)

        mdl = GradientBoostingRegressor(**params)
        mdl.fit(arr2d, y_arr, sample_weight=w)

        for stage in mdl.estimators_:
            for est in stage:
                t = est.tree_
                for i in range(t.node_count):
                    if t.children_left[i] != -1:        # internal node
                        counter[float(t.threshold[i])] += 1

        backend = "sklearn GBM"

    # Return the max_splits most-frequently-used thresholds, sorted
    top = counter.most_common(max_splits)
    splits = sorted(threshold for threshold, _ in top)

    if verbose:
        _print_splits(f"GBM ({backend})", col, splits)
    return splits


# ── Combined entry point ──────────────────────────────────────────────────────

def suggest_bins(
    col: str,
    X: pl.DataFrame,
    y: pl.Series,
    weights: Optional[pl.Series] = None,
    methods: Sequence[str] = ("quantile", "equal_width", "optbin", "gbm"),
    n_bins: int = 10,
    max_splits: int = 20,
    show_plot: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    **method_kwargs: Any,
) -> Dict[str, List[float]]:
    """
    Run multiple bin-suggestion strategies and display all results.

    Prints each method's splits as they are computed, then (optionally) shows
    a weighted histogram of ``col`` with every method's split points overlaid
    as colour-coded vertical lines for a quick visual comparison.

    Parameters
    ----------
    col : str
        Continuous variable to analyse.  Does not need to be in the model.
    methods : sequence of str
        Any subset of ``"quantile"``, ``"equal_width"``, ``"optbin"``,
        ``"gbm"``.  Defaults to running all four.
    n_bins : int
        Target bin count for the ``"quantile"`` and ``"equal_width"`` methods.
    max_splits : int
        Maximum thresholds returned by the ``"gbm"`` method.
    show_plot : bool
        If ``True``, display the distribution plot after all methods run.
    method_kwargs
        Forward kwargs to individual methods by passing ``quantile_kwargs``,
        ``equal_width_kwargs``, ``optbin_kwargs``, or ``gbm_kwargs`` as dicts.

        Example::

            tool.suggest_bins(
                'age',
                methods=['optbin', 'gbm'],
                optbin_kwargs={'max_n_bins': 8, 'monotonic_trend': 'auto'},
                gbm_kwargs={'learning_rate': 0.05},
            )

    Returns
    -------
    dict[str, list[float]]
        Maps method name → sorted list of split points.
    """
    print(f"\n{'=' * 62}")
    print(f"  Bin suggestions for : '{col}'")
    print(f"  Methods             : {list(methods)}")
    print(f"{'=' * 62}")

    results: Dict[str, List[float]] = {}

    _dispatch: Dict[str, Any] = {
        "quantile": lambda: suggest_bins_quantile(
            col, X,
            n_bins=n_bins,
            weights=weights,
            **method_kwargs.get("quantile_kwargs", {}),
        ),
        "equal_width": lambda: suggest_bins_equal_width(
            col, X,
            n_bins=n_bins,
            **method_kwargs.get("equal_width_kwargs", {}),
        ),
        "optbin": lambda: suggest_bins_optbin(
            col, X, y,
            weights=weights,
            **method_kwargs.get("optbin_kwargs", {}),
        ),
        "gbm": lambda: suggest_bins_gbm(
            col, X, y,
            weights=weights,
            max_splits=max_splits,
            **method_kwargs.get("gbm_kwargs", {}),
        ),
    }

    for method in methods:
        if method not in _dispatch:
            print(f"\n  [WARNING] Unknown method '{method}' — skipping.")
            continue
        try:
            results[method] = _dispatch[method]()
        except ImportError as exc:
            print(f"\n  [SKIPPED] {method}: {exc}")
        except Exception as exc:
            print(f"\n  [ERROR] {method}: {exc}")

    print(f"\n{'=' * 62}\n")

    if show_plot and results:
        fig = _plot_suggestions(col, X, results, weights=weights, figsize=figsize)
        plt.show()

    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def _plot_suggestions(
    col: str,
    X: pl.DataFrame,
    splits_dict: Dict[str, List[float]],
    weights: Optional[pl.Series] = None,
    n_hist_bins: int = 60,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Weighted histogram with each method's splits overlaid as vertical lines.
    """
    arr = X[col].to_numpy().astype(float)
    w = (
        weights.to_numpy().astype(float)
        if weights is not None
        else np.ones(len(arr))
    )
    mask = arr != MISSING_SENTINEL
    arr, w = arr[mask], w[mask]

    fig, ax = plt.subplots(figsize=figsize or (13, 5))
    ax.set_title(
        f"Bin suggestions — '{col}'",
        fontsize=12, fontweight="bold",
    )

    # Weighted histogram
    counts, edges = np.histogram(arr, bins=n_hist_bins, weights=w)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_w = edges[1] - edges[0]
    ax.bar(centers, counts, width=bar_w, color="#9ecae1", alpha=0.55,
           label="Exposure", zorder=1)
    ax.set_ylabel("Exposure", fontsize=9)
    ax.set_xlabel(col, fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )

    # One set of vertical lines per method, colour-coded
    for i, (method, splits) in enumerate(splits_dict.items()):
        color = _COLORS[i % len(_COLORS)]
        ls = _LINESTYLES[i % len(_LINESTYLES)]
        for j, s in enumerate(splits):
            ax.axvline(
                s,
                color=color,
                linestyle=ls,
                linewidth=1.5,
                alpha=0.9,
                label=f"{method} ({len(splits)} splits)" if j == 0 else None,
                zorder=3,
            )

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
