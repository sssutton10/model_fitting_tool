"""
discovery.py
============
Shadow GBM-based feature discovery for GLM variable selection.

Uses LightGBM as a *diagnostic lens* — never as the final model. Provides:

* Interaction ranking via Friedman's H-statistic
* Tree co-occurrence interaction ranking (fast, covers categoricals)
* SHAP importance, dependence, and interaction ranking
* Permutation importance (model-agnostic)
* Residual-based GBM to find signal the GLM is missing
* 2D partial dependence for interaction visualisation
* Categorical level grouping suggestions
* Monotonicity cost test
* Boruta-style feature selection
"""

from __future__ import annotations

import re as _re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from .variable import MISSING_SENTINEL, _NUMERIC_DTYPES, _is_str_or_cat


# ── Internal helpers ──────────────────────────────────────────────────────────

def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Call model.predict(X), suppressing the sklearn 'X does not have valid
    feature names' warning that fires when a model was fitted with named
    features but predict receives a plain numpy array.
    """
    try:
        return model.predict(X, validate_features=False)
    except TypeError:
        return model.predict(X)


def _to_numpy(s: Union[pl.Series, np.ndarray]) -> np.ndarray:
    if isinstance(s, pl.Series):
        return s.to_numpy().astype(float, copy=False)
    return np.asarray(s, dtype=float)


def _subsample_rows(
    X: np.ndarray,
    sample_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return a row-subsample of *X* (no copy when already small enough)."""
    if len(X) > sample_size:
        return X[rng.choice(len(X), sample_size, replace=False)]
    return X


# ── One-hot encoding helper ─────────────────────────────────────────────────

def _encode_features(
    df: pl.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Build a numeric matrix from *feature_cols*, one-hot encoding categoricals.

    Returns
    -------
    X : np.ndarray
        2-D float matrix, rows = observations.
    encoded_names : list of str
        Column name for each column in *X*.  Numeric columns keep their
        original name; categorical dummies are ``"col_level"``.
    col_index_map : dict
        Maps each *original* column name to the list of column indices in
        *X* that belong to it.  For a numeric column, this is a single-
        element list; for a categorical, it lists all its dummy columns.
    """
    arrays: List[np.ndarray] = []
    encoded_names: List[str] = []
    col_index_map: Dict[str, List[int]] = {}
    pos = 0

    for col in feature_cols:
        s = df[col]
        if _is_str_or_cat(s.dtype):
            filled = s.cast(pl.Utf8, strict=False).fill_null("__MISSING__")
            levels = sorted(filled.unique().to_list())
            indices = []
            for lvl in levels:
                dummy = (filled == lvl).cast(pl.Float64).to_numpy()
                arrays.append(dummy)
                encoded_names.append(f"{col}_{lvl}")
                indices.append(pos)
                pos += 1
            col_index_map[col] = indices
        else:
            arr = s.cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy(allow_copy=True)
            arrays.append(arr)
            encoded_names.append(col)
            col_index_map[col] = [pos]
            pos += 1

    X = np.column_stack(arrays) if arrays else np.empty((len(df), 0))
    return X, encoded_names, col_index_map


def _encode_features_native(
    df: pl.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, List[int]], List[str]]:
    """
    Build a numeric matrix for LightGBM's native categorical support.

    Categoricals are encoded as integer codes (0-based); LightGBM splits on
    optimal level partitions rather than individual dummies, which captures
    grouping interactions one-hot cannot.  Numeric nulls become NaN
    (LightGBM treats NaN as a separate missing branch).

    Returns
    -------
    X : np.ndarray
    encoded_names : list of str  (one name per original feature col)
    col_index_map : dict  (each col maps to its single index)
    cat_feature_names : list of str  (names of categorical columns to pass to LightGBM)
    """
    arrays: List[np.ndarray] = []
    cat_feature_names: List[str] = []
    col_index_map: Dict[str, List[int]] = {}

    for i, col in enumerate(feature_cols):
        s = df[col]
        if _is_str_or_cat(s.dtype):
            filled = s.cast(pl.Utf8, strict=False).fill_null("__MISSING__")
            levels = sorted(filled.unique().to_list())
            level_map = {lvl: j for j, lvl in enumerate(levels)}
            codes = np.array([level_map[v] for v in filled.to_list()], dtype=float)
            arrays.append(codes)
            cat_feature_names.append(col)
        else:
            arr = s.cast(pl.Float64, strict=False).to_numpy(allow_copy=True).astype(float)
            arrays.append(arr)
        col_index_map[col] = [i]

    X = np.column_stack(arrays) if arrays else np.empty((len(df), 0))
    return X, list(feature_cols), col_index_map, cat_feature_names


# ── Shadow GBM fitting ────────────────────────────────────────────────────────

def fit_shadow_gbm(
    df: pl.DataFrame,
    target_col: str,
    weight_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    family: str = "tweedie",
    tweedie_power: float = 1.5,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    use_categorical: bool = False,
    **lgb_params: Any,
) -> Any:
    """
    Fit a LightGBM regressor on raw features for diagnostic purposes.

    Parameters
    ----------
    df : pl.DataFrame
    target_col : str
    weight_col : str, optional
    feature_cols : list of str, optional
        Defaults to all numeric and string/categorical columns except target
        and weight.
    family : str
        ``'tweedie'`` (default) or ``'regression'``.
    tweedie_power : float
    use_categorical : bool
        When ``True``, categoricals are passed as integer codes with LightGBM's
        native categorical handling (optimal partition splits).  When ``False``
        (default), categoricals are one-hot encoded.  Native mode finds level-
        grouping interactions one-hot cannot; one-hot mode is compatible with
        all downstream functions including SHAP dependence.

    Returns
    -------
    lgb.LGBMRegressor with attached attributes:

        - ``_shadow_feature_cols`` — original column names (pre-encoding)
        - ``_shadow_encoded_names`` — column names after encoding
        - ``_shadow_col_index_map`` — maps original col → indices in encoded matrix
        - ``_shadow_use_categorical`` — bool, whether native categorical was used
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for shadow GBM discovery.\n"
            "Install it with:  pip install lightgbm"
        ) from exc

    if feature_cols is None:
        exclude = {target_col}
        if weight_col:
            exclude.add(weight_col)
        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and (df[c].dtype in _NUMERIC_DTYPES or _is_str_or_cat(df[c].dtype))
        ]

    y = df[target_col].to_numpy().astype(float)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None

    params: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_leaves": max(2, 2 ** max_depth - 1),
        "verbose": -1,
        "n_jobs": -1,
    }
    if family == "tweedie":
        params["objective"] = "tweedie"
        params["tweedie_variance_power"] = tweedie_power
    params.update(lgb_params)

    model = lgb.LGBMRegressor(**params)

    if use_categorical:
        X, encoded_names, col_index_map, cat_names = _encode_features_native(df, feature_cols)
        model.fit(X, y, sample_weight=w, feature_name=encoded_names,
                  categorical_feature=cat_names)
        model._shadow_use_categorical = True
        model._shadow_cat_feature_names = cat_names
    else:
        X, encoded_names, col_index_map = _encode_features(df, feature_cols)
        model.fit(X, y, sample_weight=w, feature_name=encoded_names)
        model._shadow_use_categorical = False

    model._shadow_feature_cols = feature_cols
    model._shadow_encoded_names = encoded_names
    model._shadow_col_index_map = col_index_map
    return model


# ── Prepare matrix helper (reused by downstream functions) ───────────────────

def _prepare_X(
    model: Any,
    df: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, List[int]], List[str]]:
    """
    Build the encoded matrix for *df* using the encoding stored on *model*.

    Aligns column layout to the training encoding so that scoring on a
    subset of the training data (or a holdout) never shifts column indices.
    Absent levels produce zero columns; extra levels in df are dropped.

    Returns (X, encoded_names, col_index_map, feature_cols).
    """
    if feature_cols is None:
        feature_cols = getattr(model, "_shadow_feature_cols", None)
    if feature_cols is None:
        raise ValueError("feature_cols must be provided or model must have _shadow_feature_cols")

    col_index_map = getattr(model, "_shadow_col_index_map", None)
    encoded_names = getattr(model, "_shadow_encoded_names", None)
    use_categorical = getattr(model, "_shadow_use_categorical", False)

    if col_index_map is not None and encoded_names is not None:
        if use_categorical:
            X, _, _, _ = _encode_features_native(df, feature_cols)
        else:
            # Re-encode df, then align columns to the training layout by name.
            X_raw, new_names, _ = _encode_features(df, feature_cols)
            new_name_to_idx = {n: i for i, n in enumerate(new_names)}
            n = len(df)
            X = np.zeros((n, len(encoded_names)), dtype=float)
            for i, name in enumerate(encoded_names):
                if name in new_name_to_idx:
                    X[:, i] = X_raw[:, new_name_to_idx[name]]
                # else: absent level (unseen in df) stays 0
    else:
        # Legacy path: all numeric, no encoding
        X = df.select(feature_cols).to_numpy().astype(float)
        encoded_names = list(feature_cols)
        col_index_map = {c: [i] for i, c in enumerate(feature_cols)}

    return X, encoded_names, col_index_map, feature_cols


# ── Permutation importance ────────────────────────────────────────────────────

def permutation_importance(
    model: Any,
    df: pl.DataFrame,
    target_col: str,
    weight_col: Optional[str] = None,
    metric_fn: Optional[Callable] = None,
    n_repeats: int = 5,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Model-agnostic permutation importance.

    Shuffles each feature, measures degradation in ``metric_fn``.
    For categorical variables, all dummy columns are shuffled together
    (preserving the one-hot structure).

    Parameters
    ----------
    model
        Any object with a ``.predict(X)`` method.
    metric_fn : callable, optional
        ``fn(y_true, y_pred, weights) -> float``.  Higher = better.
        Defaults to negative MSE.
    feature_cols : list of str, optional
        Original column names.  For shadow GBM models, defaults to
        ``model._shadow_feature_cols``.

    Returns
    -------
    pl.DataFrame
        Columns: ``variable``, ``importance_mean``, ``importance_std``.
        One row per *original* column (not per dummy).
    """
    X, encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)

    if metric_fn is None:
        def metric_fn(y_true, y_pred, w):
            resid = y_true - y_pred
            if w is not None:
                return -float(np.average(resid ** 2, weights=w))
            return -float(np.mean(resid ** 2))

    y = df[target_col].to_numpy().astype(float)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None

    rng = np.random.RandomState(random_state)
    baseline = metric_fn(y, _predict(model, X), w)

    results = []
    for col_name in feature_cols:
        indices = col_index_map[col_name]
        orig_cols = {idx: X[:, idx].copy() for idx in indices}
        scores = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            for idx in indices:
                X[:, idx] = X[perm, idx]
            score = metric_fn(y, _predict(model, X), w)
            scores.append(baseline - score)
            for idx, orig in orig_cols.items():
                X[:, idx] = orig
        results.append({
            "variable": col_name,
            "importance_mean": float(np.mean(scores)),
            "importance_std": float(np.std(scores)),
        })

    return pl.DataFrame(results).sort("importance_mean", descending=True)


# ── 1D Partial dependence (internal helper) ──────────────────────────────────

def _partial_dependence_1d(
    model: Any,
    X: np.ndarray,
    feature_idx: int,
    grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute 1D partial dependence at grid points for a single feature."""
    pd_values = np.zeros(len(grid))
    orig = X[:, feature_idx].copy()
    for i, val in enumerate(grid):
        X[:, feature_idx] = val
        preds = _predict(model, X)
        pd_values[i] = np.average(preds, weights=weights) if weights is not None else preds.mean()
    X[:, feature_idx] = orig
    return pd_values


# ── 2D Partial dependence ────────────────────────────────────────────────────

def partial_dependence_2d(
    model: Any,
    df: pl.DataFrame,
    var1: str,
    var2: str,
    feature_cols: Optional[List[str]] = None,
    weight_col: Optional[str] = None,
    grid_resolution: int = 20,
    sample_size: int = 500,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    2D partial dependence for a pair of *numeric* variables.

    Categorical variables are not supported for PDP (the grid concept
    doesn't apply).  They are included in the model matrix but held
    constant while the two numeric variables are varied.

    Parameters
    ----------
    model
        Fitted model with ``.predict(X)``.
    var1, var2 : str
        Original column names (must be numeric).
    weight_col : str, optional
        Exposure weight column for weighted PDP means.
    grid_resolution : int
        Number of grid points per variable (default 20).
    sample_size : int
        Subsample size for speed (default 500).

    Returns
    -------
    pl.DataFrame
        Columns: ``var1_value``, ``var2_value``, ``pd_value``.
    """
    X, encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)

    if len(col_index_map[var1]) != 1:
        raise ValueError(f"'{var1}' is categorical — PDP requires numeric variables.")
    if len(col_index_map[var2]) != 1:
        raise ValueError(f"'{var2}' is categorical — PDP requires numeric variables.")

    idx1 = col_index_map[var1][0]
    idx2 = col_index_map[var2][0]

    rng = np.random.RandomState(random_state)
    X_sample = _subsample_rows(X, sample_size, rng)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None
    if w is not None and len(X) > sample_size:
        # Subsample weights with the same random indices
        idx_sample = rng.choice(len(X), sample_size, replace=False) if len(X) > sample_size else np.arange(len(X))
        w = w[idx_sample]

    grid1 = np.unique(np.quantile(X_sample[:, idx1], np.linspace(0, 1, grid_resolution)))
    grid2 = np.unique(np.quantile(X_sample[:, idx2], np.linspace(0, 1, grid_resolution)))

    orig1 = X_sample[:, idx1].copy()
    orig2 = X_sample[:, idx2].copy()
    rows = []
    for v1 in grid1:
        X_sample[:, idx1] = v1
        for v2 in grid2:
            X_sample[:, idx2] = v2
            preds = _predict(model, X_sample)
            pd_val = float(np.average(preds, weights=w) if w is not None else preds.mean())
            rows.append({"var1_value": float(v1), "var2_value": float(v2), "pd_value": pd_val})
    X_sample[:, idx1] = orig1
    X_sample[:, idx2] = orig2

    return pl.DataFrame(rows)


# ── Interaction ranking (H-statistic) ────────────────────────────────────────

def interaction_ranking(
    model: Any,
    df: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
    weight_col: Optional[str] = None,
    top_n: int = 20,
    grid_resolution: int = 15,
    sample_size: int = 300,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Rank variable pairs by Friedman's H-statistic.

    The H-statistic measures what fraction of a variable pair's joint
    PDP variance comes from their interaction (beyond individual main effects).

    Pre-screens to top ``top_n`` variables by aggregated feature importance
    before computing pairwise H-statistics.  Only *numeric* variables are
    included in the H-statistic computation.

    Parameters
    ----------
    weight_col : str, optional
        Exposure weight column.  When provided, PDP means are exposure-weighted.
    top_n : int
    grid_resolution : int
    sample_size : int

    Returns
    -------
    pl.DataFrame
        Columns: ``var1``, ``var2``, ``h_statistic``.
    """
    X, encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)

    rng = np.random.RandomState(random_state)
    sample_idx = (
        rng.choice(len(X), sample_size, replace=False) if len(X) > sample_size
        else np.arange(len(X))
    )
    X_sample = X[sample_idx]
    w = df[weight_col].to_numpy().astype(float)[sample_idx] if weight_col else None

    try:
        raw_importances = model.feature_importances_
    except AttributeError:
        raw_importances = np.ones(len(encoded_names))

    var_importance: Dict[str, float] = {}
    for col_name in feature_cols:
        indices = col_index_map[col_name]
        var_importance[col_name] = sum(float(raw_importances[i]) for i in indices)

    numeric_vars = [c for c in feature_cols if len(col_index_map[c]) == 1]
    numeric_vars_sorted = sorted(numeric_vars, key=lambda c: var_importance.get(c, 0), reverse=True)
    top_vars = numeric_vars_sorted[:top_n]

    pdp_1d: Dict[str, np.ndarray] = {}
    grids: Dict[str, np.ndarray] = {}
    for var in top_vars:
        idx = col_index_map[var][0]
        grid = np.unique(np.quantile(X_sample[:, idx], np.linspace(0, 1, grid_resolution)))
        grids[var] = grid
        pdp_1d[var] = _partial_dependence_1d(model, X_sample, idx, grid, weights=w)

    rows = []
    for i_pos, var_i in enumerate(top_vars):
        idx_i = col_index_map[var_i][0]
        for j_pos in range(i_pos + 1, len(top_vars)):
            var_j = top_vars[j_pos]
            idx_j = col_index_map[var_j][0]

            grid_i = grids[var_i]
            grid_j = grids[var_j]
            orig_i = X_sample[:, idx_i].copy()
            orig_j = X_sample[:, idx_j].copy()
            joint_pd = np.zeros((len(grid_i), len(grid_j)))
            for gi, vi in enumerate(grid_i):
                X_sample[:, idx_i] = vi
                for gj, vj in enumerate(grid_j):
                    X_sample[:, idx_j] = vj
                    preds = _predict(model, X_sample)
                    joint_pd[gi, gj] = np.average(preds, weights=w) if w is not None else preds.mean()
            X_sample[:, idx_i] = orig_i
            X_sample[:, idx_j] = orig_j

            mean_joint = joint_pd.mean()
            interaction = (
                joint_pd
                - pdp_1d[var_i][:, np.newaxis]
                - pdp_1d[var_j][np.newaxis, :]
                + mean_joint
            )

            var_interaction = float(np.var(interaction))
            var_joint = float(np.var(joint_pd))
            h_stat = var_interaction / var_joint if var_joint > 1e-12 else 0.0

            rows.append({"var1": var_i, "var2": var_j, "h_statistic": h_stat})

    return pl.DataFrame(rows).sort("h_statistic", descending=True)


# ── Residual GBM ─────────────────────────────────────────────────────────────

def residual_gbm(
    df: pl.DataFrame,
    residuals: Union[np.ndarray, pl.Series],
    feature_cols: List[str],
    weight_col: Optional[str] = None,
    top_n: int = 10,
    n_estimators: int = 100,
    max_depth: int = 3,
    **lgb_params: Any,
) -> pl.DataFrame:
    """
    Fit a GBM on GLM residuals to find missing signal.

    Categorical columns are one-hot encoded automatically.

    Parameters
    ----------
    residuals : array-like
        Actual / predicted ratios from the current GLM.
    feature_cols : list of str
        Raw feature columns to search for signal in (numeric and/or
        categorical).
    top_n : int
        Number of top *original* features to return.

    Returns
    -------
    pl.DataFrame
        Columns: ``variable``, ``importance``, ``top_split_value``.
        One row per original column (importances summed over dummies).
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for residual GBM.\n"
            "Install it with:  pip install lightgbm"
        ) from exc

    residuals = _to_numpy(residuals)
    X, encoded_names, col_index_map = _encode_features(df, feature_cols)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None

    params: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "num_leaves": max(2, 2 ** max_depth - 1),
        "verbose": -1,
        "n_jobs": -1,
    }
    params.update(lgb_params)

    model = lgb.LGBMRegressor(**params)
    model.fit(X, residuals, sample_weight=w, feature_name=encoded_names)

    raw_importances = model.feature_importances_
    trees_df = model.booster_.trees_to_dataframe()
    split_groups = (
        trees_df.dropna(subset=["threshold"])
        .groupby("split_feature")["threshold"]
        .apply(list)
        .to_dict()
    )

    rows = []
    for col_name in feature_cols:
        indices = col_index_map[col_name]
        total_importance = sum(float(raw_importances[i]) for i in indices)

        top_split = float("nan")
        if len(indices) == 1:
            thresholds = split_groups.get(encoded_names[indices[0]], [])
            if thresholds:
                top_split = float(Counter(thresholds).most_common(1)[0][0])

        rows.append({
            "variable": col_name,
            "importance": total_importance,
            "top_split_value": top_split,
        })

    return pl.DataFrame(rows).sort("importance", descending=True).head(top_n)


# ── SHAP-based discovery ──────────────────────────────────────────────────────

def shap_importance(
    model: Any,
    df: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
    sample_size: int = 500,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    SHAP-based feature importance using TreeExplainer.

    More stable than permutation importance — no repeated shuffling needed.
    Mean |SHAP value| is summed over all dummy columns and reported per
    original variable.

    Requires
    --------
    ``pip install shap``

    Returns
    -------
    pl.DataFrame
        Columns: ``variable``, ``importance_mean``, ``importance_std``.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required: pip install shap") from exc

    X, encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)
    rng = np.random.RandomState(random_state)
    X_sample = _subsample_rows(X, sample_size, rng)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)   # (n, n_features)

    rows = []
    for col_name in feature_cols:
        indices = col_index_map[col_name]
        # Sum |SHAP| over all dummy columns for this variable, per row
        col_shap = np.abs(shap_values[:, indices]).sum(axis=1)
        rows.append({
            "variable": col_name,
            "importance_mean": float(col_shap.mean()),
            "importance_std": float(col_shap.std()),
        })

    return pl.DataFrame(rows).sort("importance_mean", descending=True)


def shap_dependence(
    model: Any,
    df: pl.DataFrame,
    var: str,
    color_var: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    sample_size: int = 500,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    SHAP dependence data for a single variable.

    The scatter of SHAP value vs. raw feature value directly shows the
    transform shape (log-like, hinge, plateau) and where breakpoints belong.
    Pass to ``plots.shap_dependence_plot`` for visualisation.

    Parameters
    ----------
    var : str
        Variable to plot (must be numeric — single column in col_index_map).
    color_var : str, optional
        A second numeric variable to colour the scatter points by, revealing
        interaction structure.

    Requires
    --------
    ``pip install shap``

    Returns
    -------
    pl.DataFrame
        Columns: ``{var}``, ``shap_value`` [, ``{color_var}``].
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required: pip install shap") from exc

    X, encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)
    rng = np.random.RandomState(random_state)
    sample_idx = (
        rng.choice(len(X), sample_size, replace=False)
        if len(X) > sample_size else np.arange(len(X))
    )
    X_sample = X[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    indices = col_index_map[var]
    var_shap = shap_values[:, indices].sum(axis=1)

    if len(indices) == 1:
        var_vals = X_sample[:, indices[0]]
    else:
        # Categorical: encode as level index (which dummy is active)
        var_vals = np.argmax(X_sample[:, indices], axis=1).astype(float)

    data: Dict[str, list] = {var: var_vals.tolist(), "shap_value": var_shap.tolist()}

    if color_var is not None and color_var in col_index_map:
        c_indices = col_index_map[color_var]
        if len(c_indices) == 1:
            data[color_var] = X_sample[:, c_indices[0]].tolist()

    return pl.DataFrame(data)


def shap_interaction_ranking(
    model: Any,
    df: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
    sample_size: int = 200,
    random_state: int = 42,
    top_n: int = 20,
) -> pl.DataFrame:
    """
    Rank variable pairs by SHAP interaction values.

    Replaces the Friedman-H PDP loop: exact TreeSHAP interactions cover
    both numeric and categorical variables and are cheaper than the O(grid²)
    PDP computation.

    Requires
    --------
    ``pip install shap``  (>= 0.40 recommended)

    Returns
    -------
    pl.DataFrame
        Columns: ``var1``, ``var2``, ``interaction_strength``.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required: pip install shap") from exc

    X, encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)
    rng = np.random.RandomState(random_state)
    X_sample = _subsample_rows(X, sample_size, rng)

    explainer = shap.TreeExplainer(model)
    # shap_interaction_values: (n, n_features, n_features)
    interaction_values = explainer.shap_interaction_values(X_sample)

    rows = []
    for i_pos, var_i in enumerate(feature_cols):
        for j_pos in range(i_pos + 1, len(feature_cols)):
            var_j = feature_cols[j_pos]
            idx_i = col_index_map[var_i]
            idx_j = col_index_map[var_j]
            strength = 0.0
            for ii in idx_i:
                for jj in idx_j:
                    strength += float(np.abs(interaction_values[:, ii, jj]).mean())
            rows.append({
                "var1": var_i,
                "var2": var_j,
                "interaction_strength": strength,
            })

    return pl.DataFrame(rows).sort("interaction_strength", descending=True).head(top_n)


# ── Tree co-occurrence interaction ranking ────────────────────────────────────

def tree_interaction_cooccurrence(
    model: Any,
    top_n: int = 20,
) -> pl.DataFrame:
    """
    Rank variable pairs by co-occurrence within the same tree, weighted by
    split gain.

    Cheaper than H-statistic and covers categorical variables (reported at
    the original variable level, not per dummy).  Use as a fast pre-screen
    to identify candidate pairs before running the full H-statistic or SHAP
    interaction analysis.

    Returns
    -------
    pl.DataFrame
        Columns: ``var1``, ``var2``, ``cooccurrence_score``.
    """
    try:
        trees_df = model.booster_.trees_to_dataframe()
    except AttributeError:
        raise ValueError(
            "tree_interaction_cooccurrence requires a LightGBM model with booster_. "
            "Call fit_shadow_gbm() first."
        )

    encoded_names: List[str] = getattr(model, "_shadow_encoded_names", [])
    col_index_map: Dict[str, List[int]] = getattr(model, "_shadow_col_index_map", {})

    # Map encoded feature name → original variable name
    encoded_to_orig: Dict[str, str] = {}
    for orig_col, idxs in col_index_map.items():
        for idx in idxs:
            if idx < len(encoded_names):
                encoded_to_orig[encoded_names[idx]] = orig_col

    internal = trees_df.dropna(subset=["split_feature"])
    pair_scores: Dict[Tuple[str, str], float] = defaultdict(float)

    for tree_id, group in internal.groupby("tree_index"):
        orig_feats = [encoded_to_orig.get(f, f) for f in group["split_feature"].tolist()]
        gains = (
            group["split_gain"].tolist()
            if "split_gain" in group.columns
            else [1.0] * len(orig_feats)
        )

        var_gain: Dict[str, float] = defaultdict(float)
        for f, g in zip(orig_feats, gains):
            if g is not None and not (isinstance(g, float) and g != g):
                var_gain[f] = max(var_gain[f], float(g))

        unique_vars = list(var_gain.keys())
        for i in range(len(unique_vars)):
            for j in range(i + 1, len(unique_vars)):
                vi, vj = unique_vars[i], unique_vars[j]
                key = (min(vi, vj), max(vi, vj))
                pair_scores[key] += (var_gain[vi] * var_gain[vj]) ** 0.5

    rows = [
        {"var1": k[0], "var2": k[1], "cooccurrence_score": float(v)}
        for k, v in pair_scores.items()
    ]
    if not rows:
        return pl.DataFrame({"var1": [], "var2": [], "cooccurrence_score": []})
    return pl.DataFrame(rows).sort("cooccurrence_score", descending=True).head(top_n)


# ── Categorical level grouping ────────────────────────────────────────────────

def suggest_category_groups(
    col: str,
    df: pl.DataFrame,
    y: Union[pl.Series, np.ndarray],
    weights: Optional[Union[pl.Series, np.ndarray]] = None,
    max_groups: int = 10,
    min_exposure_pct: float = 0.01,
    verbose: bool = True,
) -> Tuple[Dict[str, str], pl.DataFrame]:
    """
    Suggest groupings for a high-cardinality categorical variable.

    Computes the exposure-weighted mean target per level, sorts levels by
    mean target, then greedily merges:
    1. Any level below ``min_exposure_pct`` of total exposure is merged with
       its nearest neighbour (by mean target).
    2. Adjacent groups (by mean target) are merged until at most
       ``max_groups`` groups remain.

    This mirrors the manual actuary workflow of banding categories by
    observed relativity.

    Parameters
    ----------
    col : str
        Categorical column to group.
    df : pl.DataFrame
    y : array-like
        Target (e.g. loss ratio).
    weights : array-like, optional
        Exposure weights.
    max_groups : int
        Maximum number of output groups.
    min_exposure_pct : float
        Minimum exposure fraction before a level is merged (default 0.01 = 1%).

    Returns
    -------
    level_to_group : dict[str, str]
        Maps each original level to its suggested group label (``G01``, ``G02``…).
    summary : pl.DataFrame
        Columns: ``group``, ``levels``, ``exposure``, ``mean_target``.
    """
    s = df[col].cast(pl.Utf8, strict=False).fill_null("__MISSING__")
    y_arr = _to_numpy(y)
    w_arr = (
        _to_numpy(weights)
        if weights is not None
        else np.ones(len(df))
    )
    total_w = float(w_arr.sum())
    if total_w == 0:
        total_w = 1.0

    # Compute per-level stats
    level_stats = []
    for lvl in s.unique().to_list():
        mask = (s == lvl).to_numpy()
        lw = w_arr[mask]
        ly = y_arr[mask]
        w_sum = float(lw.sum())
        mean_y = float(np.average(ly, weights=lw)) if w_sum > 0 else 0.0
        level_stats.append({"level": lvl, "mean_target": mean_y, "exposure": w_sum})

    # Start with sorted groups
    groups = sorted(
        [{"levels": [r["level"]], "mean": r["mean_target"], "exposure": r["exposure"]}
         for r in level_stats],
        key=lambda g: g["mean"],
    )

    def _merge(i: int, j: int) -> None:
        gi, gj = groups[i], groups[j]
        merged_exp = gi["exposure"] + gj["exposure"]
        merged_mean = (
            (gi["mean"] * gi["exposure"] + gj["mean"] * gj["exposure"]) / merged_exp
            if merged_exp > 0 else 0.0
        )
        merged = {
            "levels": gi["levels"] + gj["levels"],
            "mean": merged_mean,
            "exposure": merged_exp,
        }
        lo, hi = min(i, j), max(i, j)
        groups.pop(hi)
        groups.pop(lo)
        groups.insert(lo, merged)

    # Phase 1: merge tiny levels
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i, g in enumerate(groups):
            if g["exposure"] / total_w < min_exposure_pct:
                if i == 0:
                    neighbor = 1
                elif i == len(groups) - 1:
                    neighbor = len(groups) - 2
                else:
                    neighbor = (
                        i - 1
                        if abs(groups[i - 1]["mean"] - g["mean"])
                        <= abs(groups[i + 1]["mean"] - g["mean"])
                        else i + 1
                    )
                _merge(i, neighbor)
                # Re-sort after merge to keep groups ordered by mean
                groups.sort(key=lambda g2: g2["mean"])
                changed = True
                break

    # Phase 2: merge to max_groups
    while len(groups) > max_groups:
        diffs = [
            abs(groups[k + 1]["mean"] - groups[k]["mean"])
            for k in range(len(groups) - 1)
        ]
        idx = int(np.argmin(diffs))
        _merge(idx, idx + 1)

    # Build output
    level_to_group: Dict[str, str] = {}
    summary_rows = []
    for gi, g in enumerate(groups, 1):
        label = f"G{gi:02d}"
        for lvl in g["levels"]:
            level_to_group[lvl] = label
        summary_rows.append({
            "group": label,
            "levels": str(g["levels"]),
            "exposure": g["exposure"],
            "mean_target": g["mean"],
        })

    summary = pl.DataFrame(summary_rows).sort("mean_target")

    if verbose:
        print(f"\n  Category groups for '{col}' ({len(groups)} groups):")
        for row in summary.iter_rows(named=True):
            print(
                f"    {row['group']}: mean={row['mean_target']:.4f}  "
                f"exp={row['exposure']:,.0f}  levels={row['levels']}"
            )

    return level_to_group, summary


# ── Monotonicity cost test ────────────────────────────────────────────────────

def monotonicity_test(
    df: pl.DataFrame,
    target_col: str,
    var: str,
    weight_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    **lgb_params: Any,
) -> Dict[str, Any]:
    """
    Measure the RMSE cost of enforcing a monotone constraint on ``var``.

    Fits two additional LightGBM models — one with ``monotone_constraints=+1``
    and one with ``-1`` on ``var`` — and reports the RMSE increase versus an
    unconstrained baseline.  A small cost (< ~1%) means you can safely enforce
    the constraint in the GLM binning without sacrificing lift.

    Parameters
    ----------
    var : str
        Variable to test (must be in feature_cols or numeric columns of df).
    verbose : bool
        Print results table.

    Returns
    -------
    dict with keys:
        ``unconstrained_rmse``, ``constrained_rmse_pos``, ``constrained_rmse_neg``,
        ``cost_pos``, ``cost_neg``, ``recommended`` ('increasing', 'decreasing',
        or 'no_constraint').
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("LightGBM is required: pip install lightgbm") from exc

    if feature_cols is None:
        exclude = {target_col}
        if weight_col:
            exclude.add(weight_col)
        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and (df[c].dtype in _NUMERIC_DTYPES or _is_str_or_cat(df[c].dtype))
        ]

    if var not in feature_cols:
        raise ValueError(f"'{var}' not in feature_cols.")

    X, encoded_names, col_index_map = _encode_features(df, feature_cols)
    y = df[target_col].to_numpy().astype(float)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None
    n_features = X.shape[1]

    base_params: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": random_state,
    }
    base_params.update(lgb_params)

    def _rmse(mdl: Any) -> float:
        pred = mdl.predict(X)
        r = y - pred
        return float(np.sqrt(np.average(r ** 2, weights=w) if w is not None else np.mean(r ** 2)))

    mdl_free = lgb.LGBMRegressor(**base_params)
    mdl_free.fit(X, y, sample_weight=w, feature_name=encoded_names)
    rmse_free = _rmse(mdl_free)

    var_indices = col_index_map[var]
    constrained_rmses: Dict[str, float] = {}
    for direction, label in [(1, "pos"), (-1, "neg")]:
        constraints = [0] * n_features
        for idx in var_indices:
            constraints[idx] = direction
        params_c = dict(base_params, monotone_constraints=constraints)
        mdl_c = lgb.LGBMRegressor(**params_c)
        mdl_c.fit(X, y, sample_weight=w, feature_name=encoded_names)
        constrained_rmses[label] = _rmse(mdl_c)

    cost_pos = constrained_rmses["pos"] - rmse_free
    cost_neg = constrained_rmses["neg"] - rmse_free
    threshold = rmse_free * 0.01   # 1% RMSE increase is "trivial"

    if cost_pos <= cost_neg and cost_pos <= threshold:
        recommended = "increasing"
    elif cost_neg < cost_pos and cost_neg <= threshold:
        recommended = "decreasing"
    else:
        recommended = "no_constraint"

    if verbose:
        print(f"\n  Monotonicity test — '{var}'")
        print(f"    Unconstrained RMSE : {rmse_free:.6f}")
        print(f"    Monotone (+1) RMSE : {constrained_rmses['pos']:.6f}  cost = {cost_pos / rmse_free:+.2%}")
        print(f"    Monotone (-1) RMSE : {constrained_rmses['neg']:.6f}  cost = {cost_neg / rmse_free:+.2%}")
        print(f"    Recommendation     : {recommended}")

    return {
        "unconstrained_rmse": rmse_free,
        "constrained_rmse_pos": constrained_rmses["pos"],
        "constrained_rmse_neg": constrained_rmses["neg"],
        "cost_pos": cost_pos,
        "cost_neg": cost_neg,
        "recommended": recommended,
    }


# ── Boruta-style feature selection ────────────────────────────────────────────

def boruta_select(
    df: pl.DataFrame,
    target_col: str,
    weight_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    n_estimators: int = 100,
    n_iterations: int = 20,
    threshold: float = 0.05,
    random_state: int = 42,
    **lgb_params: Any,
) -> pl.DataFrame:
    """
    Boruta-style feature selection using shadow (shuffled) features.

    In each iteration, a shuffled copy of every feature is appended to the
    matrix and LightGBM is fit.  A real feature "wins" that iteration if its
    importance exceeds the *maximum* importance of any shadow feature.
    Features that win in at least ``1 - threshold`` of iterations are selected.

    This provides an objective importance threshold rather than requiring the
    user to eyeball the permutation importance bar chart.

    Requires
    --------
    ``pip install lightgbm``

    Parameters
    ----------
    n_iterations : int
        Number of shadow-feature iterations (default 20).
    threshold : float
        Maximum allowed fraction of losing iterations for a feature to be
        selected (default 0.05 → must win 95 % of iterations).

    Returns
    -------
    pl.DataFrame
        Columns: ``variable``, ``pass_rate``, ``selected``.
        Sorted by ``pass_rate`` descending.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("LightGBM is required: pip install lightgbm") from exc

    if feature_cols is None:
        exclude = {target_col}
        if weight_col:
            exclude.add(weight_col)
        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and (df[c].dtype in _NUMERIC_DTYPES or _is_str_or_cat(df[c].dtype))
        ]

    X, encoded_names, col_index_map = _encode_features(df, feature_cols)
    y = df[target_col].to_numpy().astype(float)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None
    n_real = X.shape[1]

    rng = np.random.RandomState(random_state)
    hit_counts: Dict[str, int] = {c: 0 for c in feature_cols}

    params: Dict[str, Any] = {"n_estimators": n_estimators, "verbose": -1, "n_jobs": -1}
    params.update(lgb_params)

    for _ in range(n_iterations):
        shadow = np.column_stack([rng.permutation(X[:, j]) for j in range(n_real)])
        X_aug = np.column_stack([X, shadow])
        shadow_names = [f"__shadow_{n}" for n in encoded_names]
        aug_names = encoded_names + shadow_names

        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_aug, y, sample_weight=w, feature_name=aug_names)

        imp = mdl.feature_importances_
        real_imp = imp[:n_real]
        max_shadow = imp[n_real:].max()

        for col_name in feature_cols:
            indices = col_index_map[col_name]
            col_imp = sum(float(real_imp[i]) for i in indices)
            if col_imp > max_shadow:
                hit_counts[col_name] += 1

    rows = [
        {
            "variable": col_name,
            "pass_rate": hit_counts[col_name] / n_iterations,
            "selected": hit_counts[col_name] / n_iterations >= (1.0 - threshold),
        }
        for col_name in feature_cols
    ]
    return pl.DataFrame(rows).sort("pass_rate", descending=True)
