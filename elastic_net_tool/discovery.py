"""
discovery.py
============
Shadow GBM-based feature discovery for GLM variable selection.

Uses LightGBM as a *diagnostic lens* — never as the final model. Provides:

* Interaction ranking via Friedman's H-statistic
* Permutation importance (model-agnostic)
* Residual-based GBM to find signal the GLM is missing
* 2D partial dependence for interaction visualisation
"""

from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import polars as pl

from .model import _build_preprocessor
from .plots import _resolve_level
from .variable import _NUMERIC_DTYPES, MISSING_SENTINEL, _is_str_or_cat

# ── Internal helpers ──────────────────────────────────────────────────────────


# Modified LGBMRegressor that can be used with polars dataframes, to be consistent with the rest of the codebase
class LGBRegressorPolars(lgb.LGBMRegressor):
    @property
    def feature_names_in_(self):
        return self._feature_name

    @feature_names_in_.setter
    def feature_names_in_(self, x):
        self._feature_name = x


def _predict(model: Any, X: pl.DataFrame) -> np.ndarray:
    """
    Call model.predict(X), suppressing the sklearn 'X does not have valid
    feature names' warning that fires when a model was fitted with named
    features but predict receives a plain numpy array.

    Passes ``validate_features=False`` for sklearn-compatible models that
    accept that keyword; falls back to a plain call for others.
    """
    try:
        return model.predict(X.to_arrow(), validate_features=False)
    except TypeError:
        return model.predict(X.to_arrow())


def _to_numpy(s: pl.Series | np.ndarray) -> np.ndarray:
    if isinstance(s, pl.Series):
        return s.to_numpy().astype(float, copy=False)
    return np.asarray(s, dtype=float)


def _subsample_rows(
    X: pl.DataFrame,
    sample_size: int,
    rng: np.random.RandomState,
) -> pl.DataFrame:
    """Return a row-subsample of *X* (no copy when already small enough)."""
    if len(X) > sample_size:
        return X[rng.choice(len(X), sample_size, replace=False)]
    return X


# ── One-hot encoding helper ─────────────────────────────────────────────────


def _materialize_discovery_columns(
    data: pl.DataFrame,
    feature_columns: list[str],
    variable_configs: dict[str, Any] | None,
    weights: np.ndarray | None,
) -> pl.DataFrame:
    """Materialize configured derived variables for discovery models."""
    variable_configs = variable_configs or {}
    calculated_columns = []
    materialized = data.clone()
    for col in feature_columns:
        assert col in materialized.columns or col in variable_configs, (
            f"Feature column '{col}' not found in DataFrame or variable_configs."
        )
        if col in variable_configs and col not in materialized.columns:
            calculated_columns.append(col)
    if calculated_columns:
        preprocessor = _build_preprocessor(
            calculated_columns,
            materialized,
            variable_configs,
        )
        preprocessor.fit(materialized, weights=weights)
        for col in calculated_columns:
            values = _resolve_level(col, materialized, preprocessor, 10)
            materialized = materialized.with_columns(pl.Series(col, values))
    return materialized


def _encoded_discovery_data(
    data: pl.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pl.DataFrame:
    """Encode numeric and categorical discovery features."""
    numeric_data = {
        column: data[column]
        .cast(pl.Float32, strict=False)
        .fill_null(0.0)
        .shrink_dtype()
        for column in numeric_columns
    }
    dummy_data: dict[str, pl.Series] = {}
    if categorical_columns:
        categorical = data.select(categorical_columns).with_columns(
            [
                pl.col(column)
                .cast(pl.Utf8, strict=False)
                .fill_null("__MISSING__")
                .alias(column)
                for column in categorical_columns
            ]
        )
        dummies = categorical.to_dummies()
        dummy_data = {name: dummies[name] for name in dummies.columns}
    all_data = {**numeric_data, **dummy_data}
    return pl.DataFrame(all_data) if all_data else pl.DataFrame()


def _sanitize_discovery_columns(data: pl.DataFrame) -> pl.DataFrame:
    """Remove JSON-special characters rejected by LightGBM."""
    translation = str.maketrans({character: "_" for character in r'[]{}":,\/'})
    raw_columns = list(data.columns)
    clean_columns = [column.translate(translation) for column in raw_columns]
    if clean_columns == raw_columns:
        return data
    return data.rename(dict(zip(raw_columns, clean_columns)))


def _discovery_column_indices(
    encoded_names: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, list[int]]:
    """Map original variables to positions in the encoded feature matrix."""
    positions = {name: index for index, name in enumerate(encoded_names)}
    mapping = {column: [positions[column]] for column in numeric_columns}
    for column in categorical_columns:
        mapping[column] = [
            positions[name] for name in encoded_names if name.startswith(f"{column}_")
        ]
    return mapping


def _encode_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    variable_configs: dict[str, Any] | None = {},
    weights: np.ndarray | None = None,
) -> tuple[pl.DataFrame, list[str], dict[str, list[int]]]:
    df_enc = _materialize_discovery_columns(
        df,
        feature_cols,
        variable_configs,
        weights,
    )

    numeric_cols = [
        col for col in feature_cols if not _is_str_or_cat(df_enc[col].dtype)
    ]
    cat_cols = [col for col in feature_cols if _is_str_or_cat(df_enc[col].dtype)]
    X = _sanitize_discovery_columns(
        _encoded_discovery_data(df_enc, numeric_cols, cat_cols)
    )
    encoded_names = list(X.columns)
    return (
        X,
        encoded_names,
        _discovery_column_indices(
            encoded_names,
            numeric_cols,
            cat_cols,
        ),
    )


# ── Shadow GBM fitting ────────────────────────────────────────────────────────


def fit_shadow_gbm(
    df: pl.DataFrame,
    target_col: str,
    weight_col: str | None = None,
    offset_col: str | None = None,
    feature_cols: list[str] | None = None,
    family: str = "tweedie",
    tweedie_power: float = 1.5,
    variable_configs: dict[str, Any] | None = None,
    n_estimators: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.01,
    **lgb_params: Any,
) -> Any:
    """
    Fit a LightGBM regressor on raw features for diagnostic purposes.

    Categorical columns are one-hot encoded automatically.

    Parameters
    ----------
    df : pl.DataFrame
        Source data.
    target_col : str
        Target column (e.g. loss_ratio).
    weight_col : str, optional
        Exposure weight column.
    feature_cols : list of str, optional
        Columns to use as features.  Defaults to all numeric *and*
        string/categorical columns except target and weight.
    family : str
        ``'tweedie'`` (default) or ``'regression'``.
    tweedie_power : float
        Tweedie variance power (default 1.5, typical for insurance).

    Returns
    -------
    lgb.LGBMRegressor
        Fitted LightGBM model.  Also carries:

        - ``_shadow_feature_cols`` — original column names (pre-encoding)
        - ``_shadow_encoded_names`` — column names after one-hot encoding
        - ``_shadow_col_index_map`` — maps original col → indices in encoded matrix
    """
    if feature_cols is None:
        exclude = {target_col}
        if weight_col:
            exclude.add(weight_col)
        feature_cols = [
            c
            for c in df.columns
            if c not in exclude
            and (df[c].dtype in _NUMERIC_DTYPES or _is_str_or_cat(df[c].dtype))
        ]

    X, encoded_names, col_index_map = _encode_features(
        df, feature_cols, variable_configs
    )

    X = X.with_columns(
        pl.col(
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ).cast(pl.Float64)
    )
    X = X.with_columns(
        pl.col(pl.Float32, pl.Float64).replace(MISSING_SENTINEL, float("nan"))
    )

    y = df[target_col].to_numpy().astype(float)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None
    if offset_col is not None and w is not None:
        w = w * np.exp(df[offset_col].to_numpy().astype(float))

    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_leaves": max(2, 2**max_depth - 1),
        "verbose": -1,
        "n_jobs": -1,
    }
    if family == "tweedie":
        params["objective"] = "tweedie"
        params["tweedie_variance_power"] = tweedie_power
    params.update(lgb_params)

    model = LGBRegressorPolars(**params)
    model.fit(X.to_arrow(), y, sample_weight=w, feature_name=encoded_names)

    # Stash encoding info for downstream functions
    model._shadow_feature_cols = feature_cols
    model._shadow_encoded_names = encoded_names
    model._shadow_col_index_map = col_index_map
    model._variable_configs = variable_configs or {}
    model._weight_col = weight_col

    return model


# ── Prepare matrix helper (reused by downstream functions) ───────────────────


def _prepare_X(
    model: Any,
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str], dict[str, list[int]], list[str]]:
    """
    Build the encoded matrix for *df* using the encoding stored on *model*.

    Returns (X, encoded_names, col_index_map, feature_cols).
    """
    if feature_cols is None:
        feature_cols = getattr(model, "_shadow_feature_cols", None)
    if feature_cols is None:
        raise ValueError(
            "feature_cols must be provided or model must have _shadow_feature_cols"
        )

    col_index_map = getattr(model, "_shadow_col_index_map", None)
    encoded_names = getattr(model, "_shadow_encoded_names", None)
    variable_configs = getattr(model, "_variable_configs", None)
    weight_col = getattr(model, "_weight_col", None)

    if weight_col is not None:
        weights = df[weight_col].to_numpy().astype(float)
    else:
        weights = None

    if col_index_map is not None and encoded_names is not None:
        X, _, _ = _encode_features(df, feature_cols, variable_configs, weights)
    else:
        # Legacy path: all numeric, no encoding needed
        X = df.select(feature_cols)
        encoded_names = list(feature_cols)
        col_index_map = {c: [i] for i, c in enumerate(feature_cols)}

    return X, encoded_names, col_index_map, feature_cols


# ── Permutation importance ────────────────────────────────────────────────────


def permutation_importance(
    model: Any,
    df: pl.DataFrame,
    target_col: str,
    weight_col: str | None = None,
    metric_fn: Callable | None = None,
    n_repeats: int = 5,
    feature_cols: list[str] | None = None,
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
    X, _, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)

    if metric_fn is None:

        def metric_fn(y_true, y_pred, w):
            resid = y_true - y_pred
            if w is not None:
                return -float(np.average(resid**2, weights=w))
            return -float(np.mean(resid**2))

    y = df[target_col].to_numpy().astype(float)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None

    baseline = metric_fn(y, _predict(model, X), w)

    results = []
    for col_name in feature_cols:
        indices = col_index_map[col_name]
        orig_cols = {idx: X[:, idx].clone() for idx in indices}
        scores = []
        for _ in range(n_repeats):
            # Use the same permutation for all dummy columns of this variable
            # to preserve one-hot validity (each row maps to exactly one level).
            perm = X.clone().sample(fraction=1.0, seed=random_state, shuffle=True)
            for idx in indices:
                X = X.with_columns(perm.to_series(idx).alias(X.columns[idx]))
            score = metric_fn(y, _predict(model, X), w)
            scores.append(baseline - score)
            for idx, orig in orig_cols.items():
                X = X.with_columns(orig.alias(X.columns[idx]))
        results.append(
            {
                "variable": col_name,
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )

    return pl.DataFrame(results).sort("importance_mean", descending=True)


# ── 1D Partial dependence (internal helper) ──────────────────────────────────


def _partial_dependence_1d(
    model: Any,
    X: pl.DataFrame,
    feature_idx: int,
    grid: np.ndarray,
) -> np.ndarray:
    """Compute 1D partial dependence at grid points for a single feature."""
    pd_values = np.zeros(len(grid))
    orig = X[:, feature_idx].clone()
    for i, val in enumerate(grid):
        X = X.with_columns(pl.lit(val).alias(X.columns[feature_idx]))
        pd_values[i] = _predict(model, X).mean()
    X = X.with_columns(orig.alias(X.columns[feature_idx]))
    return pd_values


# ── 2D Partial dependence ────────────────────────────────────────────────────


def partial_dependence_2d(
    model: Any,
    df: pl.DataFrame,
    var1: str,
    var2: str,
    feature_cols: list[str] | None = None,
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
    grid_resolution : int
        Number of grid points per variable (default 20).
    sample_size : int
        Subsample size for speed (default 500).

    Returns
    -------
    pl.DataFrame
        Columns: ``var1_value``, ``var2_value``, ``pd_value``.
    """
    X, _encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)

    # PDP only makes sense for single-index (numeric) columns
    if len(col_index_map[var1]) != 1:
        raise ValueError(f"'{var1}' is categorical — PDP requires numeric variables.")
    if len(col_index_map[var2]) != 1:
        raise ValueError(f"'{var2}' is categorical — PDP requires numeric variables.")

    idx1 = col_index_map[var1][0]
    idx2 = col_index_map[var2][0]

    # Subsample for speed
    rng = np.random.RandomState(random_state)
    X_sample = _subsample_rows(X, sample_size, rng)

    # Grid at quantiles for better coverage
    grid1 = np.unique(
        np.quantile(X_sample[:, idx1], np.linspace(0, 1, grid_resolution))
    )
    grid2 = np.unique(
        np.quantile(X_sample[:, idx2], np.linspace(0, 1, grid_resolution))
    )

    orig1 = X_sample[:, idx1].clone()
    orig2 = X_sample[:, idx2].clone()
    rows = []
    for v1 in grid1:
        X_sample = X_sample.with_columns(pl.lit(v1).alias(X_sample.columns[idx1]))
        for v2 in grid2:
            X_sample = X_sample.with_columns(pl.lit(v2).alias(X_sample.columns[idx2]))
            rows.append(
                {
                    "var1_value": float(v1),
                    "var2_value": float(v2),
                    "pd_value": float(_predict(model, X_sample).mean()),
                }
            )
    X_sample = X_sample.with_columns(orig1.alias(X_sample.columns[idx1]))
    X_sample = X_sample.with_columns(orig2.alias(X_sample.columns[idx2]))

    return pl.DataFrame(rows)


# ── Interaction ranking (H-statistic) ────────────────────────────────────────


def interaction_ranking(
    model: Any,
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
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
    included in the H-statistic computation (the PDP grid doesn't apply
    to one-hot dummies), but categorical variables contribute to the model
    and improve the GBM's ability to detect interactions.

    Parameters
    ----------
    top_n : int
        Number of top *numeric* variables (by importance) to consider.
    grid_resolution : int
        Grid points per variable for PDP computation.
    sample_size : int
        Subsample size for speed.

    Returns
    -------
    pl.DataFrame
        Columns: ``var1``, ``var2``, ``h_statistic``.
    """
    X, encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)

    # Subsample for speed
    rng = np.random.RandomState(random_state)
    X_sample = _subsample_rows(X, sample_size, rng)

    # Aggregate importance per original variable (sum over dummies)
    try:
        raw_importances = model.feature_importances_
    except AttributeError:
        raw_importances = np.ones(len(encoded_names))

    var_importance: dict[str, float] = {}
    for col_name in feature_cols:
        indices = col_index_map[col_name]
        var_importance[col_name] = sum(float(raw_importances[i]) for i in indices)

    # Only keep numeric variables for H-statistic (single-index in col_index_map)
    numeric_vars = [c for c in feature_cols if len(col_index_map[c]) == 1]
    numeric_vars_sorted = sorted(
        numeric_vars, key=lambda c: var_importance.get(c, 0), reverse=True
    )
    top_vars = numeric_vars_sorted[:top_n]

    # Compute 1D PDPs for each top variable
    pdp_1d: dict[str, np.ndarray] = {}
    grids: dict[str, np.ndarray] = {}
    for var in top_vars:
        idx = col_index_map[var][0]
        grid = np.unique(
            np.quantile(X_sample[:, idx], np.linspace(0, 1, grid_resolution))
        )
        grids[var] = grid
        pdp_1d[var] = _partial_dependence_1d(model, X_sample, idx, grid)

    # Compute H-statistic for each pair
    rows = []
    for i_pos, var_i in enumerate(top_vars):
        idx_i = col_index_map[var_i][0]
        for j_pos in range(i_pos + 1, len(top_vars)):
            var_j = top_vars[j_pos]
            idx_j = col_index_map[var_j][0]

            grid_i = grids[var_i]
            grid_j = grids[var_j]
            orig_i = X_sample[:, idx_i].clone()
            orig_j = X_sample[:, idx_j].clone()
            joint_pd = np.zeros((len(grid_i), len(grid_j)))
            for gi, vi in enumerate(grid_i):
                X_sample = X_sample.with_columns(
                    pl.lit(vi).alias(X_sample.columns[idx_i])
                )
                for gj, vj in enumerate(grid_j):
                    X_sample = X_sample.with_columns(
                        pl.lit(vj).alias(X_sample.columns[idx_j])
                    )
                    joint_pd[gi, gj] = _predict(model, X_sample).mean()
            X_sample = X_sample.with_columns(orig_i.alias(X_sample.columns[idx_i]))
            X_sample = X_sample.with_columns(orig_j.alias(X_sample.columns[idx_j]))

            mean_joint = joint_pd.mean()
            pd_i_expanded = pdp_1d[var_i][:, np.newaxis]
            pd_j_expanded = pdp_1d[var_j][np.newaxis, :]
            interaction = joint_pd - pd_i_expanded - pd_j_expanded + mean_joint

            var_interaction = float(np.var(interaction))
            var_joint = float(np.var(joint_pd))

            h_stat = var_interaction / var_joint if var_joint > 1e-12 else 0.0

            rows.append(
                {
                    "var1": var_i,
                    "var2": var_j,
                    "h_statistic": h_stat,
                }
            )

    return pl.DataFrame(rows).sort("h_statistic", descending=True)


# ── Residual GBM ─────────────────────────────────────────────────────────────


def residual_gbm(
    df: pl.DataFrame,
    residuals: np.ndarray | pl.Series,
    feature_cols: list[str],
    weight_col: str | None = None,
    offset_col: str | None = None,
    top_n: int = 10,
    variable_configs: dict[str, Any] | None = None,
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
    spec = importlib.util.find_spec("lightgbm")
    if spec is None:
        raise ImportError(
            "LightGBM is required for residual GBM.\n"
            "Install it with:  pip install lightgbm"
        )

    residuals = _to_numpy(residuals)

    X, encoded_names, col_index_map = _encode_features(
        df, feature_cols, variable_configs
    )
    X = X.with_columns(
        pl.col(
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ).cast(pl.Float64)
    )
    X = X.with_columns(
        pl.col(pl.Float32, pl.Float64).replace(MISSING_SENTINEL, float("nan"))
    )

    w = df[weight_col].to_numpy().astype(float) if weight_col else None
    if offset_col is not None and w is not None:
        w = w * np.exp(df[offset_col].to_numpy().astype(float))

    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "num_leaves": max(2, 2**max_depth - 1),
        "verbose": -1,
        "n_jobs": -1,
    }

    params.update(lgb_params)

    model = LGBRegressorPolars(**params)
    model.fit(X.to_arrow(), residuals, sample_weight=w, feature_name=encoded_names)

    # Aggregate importance per original variable
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

        # Find top split: only meaningful for numeric (single-index) columns
        top_split = float("nan")
        if len(indices) == 1:
            use_name = (
                col_name if col_name in model.feature_name_ else f"Column_{indices[0]}"
            )
            thresholds = split_groups.get(use_name, [])
            if thresholds:
                top_split = float(Counter(thresholds).most_common(1)[0][0])

        rows.append(
            {
                "variable": col_name,
                "importance": total_importance,
                "top_split_value": round(top_split, 3),
            }
        )

    result = pl.DataFrame(rows).sort("importance", descending=True).head(top_n)
    return result


# ── SHAP-based discovery ──────────────────────────────────────────────────────


def _normalize_shap_values(shap_values: Any) -> np.ndarray:
    """
    Coerce a TreeExplainer ``shap_values`` result to a dense 2-D
    ``(n, n_features)`` ndarray.

    SHAP can return several layouts that all break the
    ``[:, indices].sum(axis=1)`` idiom used here:

    * a list of ``(n, n_features)`` arrays for multi-output models;
    * a 3-D ``(n, n_features, n_outputs)`` array for single-output regression
      (trailing output axis) in recent versions;
    * a SciPy sparse matrix or ``np.matrix`` — for these, ``sum(axis=1)``
      stays 2-D (the classic ``np.matrix`` gotcha), leaving ``var_shap`` a
      matrix that polars refuses to build a column from.

    Densify and collapse to a single output so the result is a plain 2-D
    ``np.ndarray``.
    """
    if isinstance(shap_values, list):
        # Multi-output: list of (n, n_features) arrays — take the first output.
        shap_values = shap_values[0]
    # SciPy sparse matrices expose .toarray(); densify before np.asarray so we
    # don't end up with a 0-d object array wrapping the sparse matrix.
    if hasattr(shap_values, "toarray"):
        shap_values = shap_values.toarray()
    # np.asarray downcasts np.matrix to a plain ndarray (so sum collapses dims).
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        # (n, n_features, n_outputs) — take the first output.
        shap_values = shap_values[:, :, 0]
    return shap_values


def shap_importance(
    model: Any,
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
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

    X, _encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)
    rng = np.random.RandomState(random_state)
    X_sample = _subsample_rows(X, sample_size, rng)

    explainer = shap.TreeExplainer(model)
    shap_values = _normalize_shap_values(
        explainer.shap_values(X_sample.to_numpy())
    )  # (n, n_features)

    rows = []
    for col_name in feature_cols:
        indices = col_index_map[col_name]
        # Sum |SHAP| over all dummy columns for this variable, per row
        col_shap = np.abs(shap_values[:, indices]).sum(axis=1)
        rows.append(
            {
                "variable": col_name,
                "importance_mean": float(col_shap.mean()),
                "importance_std": float(col_shap.std()),
            }
        )

    return pl.DataFrame(rows).sort("importance_mean", descending=True)


def shap_dependence(
    model: Any,
    df: pl.DataFrame,
    var: str,
    color_var: str | None = None,
    feature_cols: list[str] | None = None,
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

    X, _encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)
    rng = np.random.RandomState(random_state)
    sample_idx = (
        rng.choice(len(X), sample_size, replace=False)
        if len(X) > sample_size
        else np.arange(len(X))
    )
    X_sample = X[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = _normalize_shap_values(explainer.shap_values(X_sample.to_numpy()))

    indices = col_index_map[var]
    var_shap = shap_values[:, indices].sum(axis=1)

    if len(indices) == 1:
        var_vals = X_sample[:, indices[0]].to_numpy()
    else:
        # Categorical: encode as level index (which dummy is active)
        var_vals = np.argmax(X_sample[:, indices], axis=1).astype(float)

    data: dict[str, list] = {var: var_vals.tolist(), "shap_value": var_shap.tolist()}

    if color_var is not None and color_var in col_index_map:
        c_indices = col_index_map[color_var]
        if len(c_indices) == 1:
            data[color_var] = X_sample[:, c_indices[0]].tolist()

    return pl.DataFrame(data)


def shap_interaction_ranking(
    model: Any,
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
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

    X, _encoded_names, col_index_map, feature_cols = _prepare_X(model, df, feature_cols)
    rng = np.random.RandomState(random_state)
    X_sample = _subsample_rows(X, sample_size, rng)

    explainer = shap.TreeExplainer(model)
    # shap_interaction_values: (n, n_features, n_features)
    interaction_values = explainer.shap_interaction_values(X_sample.to_numpy())

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
            rows.append(
                {
                    "var1": var_i,
                    "var2": var_j,
                    "interaction_strength": strength,
                }
            )

    return pl.DataFrame(rows).sort("interaction_strength", descending=True).head(top_n)


# ── Tree co-occurrence interaction ranking ────────────────────────────────────


def _encoded_to_original_map(model: Any) -> dict[str, str]:
    """Map encoded feature names back to original variables."""
    encoded_names: list[str] = getattr(model, "_shadow_encoded_names", [])
    column_indices: dict[str, list[int]] = getattr(
        model,
        "_shadow_col_index_map",
        {},
    )
    return {
        encoded_names[index]: original
        for original, indices in column_indices.items()
        for index in indices
        if index < len(encoded_names)
    }


def _tree_variable_gains(
    tree: Any,
    encoded_to_original: dict[str, str],
) -> dict[str, float]:
    """Return the maximum split gain for each original variable in one tree."""
    features = [
        encoded_to_original.get(feature, feature)
        for feature in tree["split_feature"].tolist()
    ]
    gains = (
        tree["split_gain"].tolist()
        if "split_gain" in tree.columns
        else [1.0] * len(features)
    )
    variable_gains: dict[str, float] = defaultdict(float)
    for feature, gain in zip(features, gains):
        if gain is not None and not (isinstance(gain, float) and gain != gain):
            variable_gains[feature] = max(variable_gains[feature], float(gain))
    return variable_gains


def _add_tree_pair_scores(
    pair_scores: dict[tuple[str, str], float],
    variable_gains: dict[str, float],
) -> None:
    """Accumulate geometric split-gain scores for every pair in one tree."""
    variables = list(variable_gains)
    for first_index, first in enumerate(variables):
        for second in variables[first_index + 1 :]:
            pair = (min(first, second), max(first, second))
            pair_scores[pair] += (variable_gains[first] * variable_gains[second]) ** 0.5


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

    internal = trees_df.dropna(subset=["split_feature"])
    encoded_to_original = _encoded_to_original_map(model)
    pair_scores: dict[tuple[str, str], float] = defaultdict(float)
    for _tree_id, tree in internal.groupby("tree_index"):
        _add_tree_pair_scores(
            pair_scores,
            _tree_variable_gains(tree, encoded_to_original),
        )

    rows = [
        {"var1": k[0], "var2": k[1], "cooccurrence_score": float(v)}
        for k, v in pair_scores.items()
    ]
    if not rows:
        return pl.DataFrame({"var1": [], "var2": [], "cooccurrence_score": []})
    return pl.DataFrame(rows).sort("cooccurrence_score", descending=True).head(top_n)


# ── Categorical level grouping ────────────────────────────────────────────────


def _initial_category_groups(
    levels: pl.Series,
    target: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    """Build one exposure/target record per original category."""
    groups = []
    for level in levels.unique().to_list():
        mask = (levels == level).to_numpy()
        level_weights = weights[mask]
        weight_sum = float(level_weights.sum())
        groups.append(
            {
                "levels": [level],
                "mean": (
                    float(np.average(target[mask], weights=level_weights))
                    if weight_sum > 0
                    else 0.0
                ),
                "exposure": weight_sum,
            }
        )
    return sorted(groups, key=lambda group: group["mean"])


def _merge_category_groups(
    groups: list[dict[str, Any]],
    first_index: int,
    second_index: int,
) -> None:
    """Merge two category groups in place using their exposure-weighted mean."""
    first, second = groups[first_index], groups[second_index]
    exposure = first["exposure"] + second["exposure"]
    mean = (
        (first["mean"] * first["exposure"] + second["mean"] * second["exposure"])
        / exposure
        if exposure > 0
        else 0.0
    )
    lower, higher = min(first_index, second_index), max(first_index, second_index)
    groups.pop(higher)
    groups.pop(lower)
    groups.insert(
        lower,
        {
            "levels": first["levels"] + second["levels"],
            "mean": mean,
            "exposure": exposure,
        },
    )


def _merge_small_category_groups(
    groups: list[dict[str, Any]],
    total_exposure: float,
    minimum_exposure_percent: float,
) -> None:
    """Merge low-exposure categories with their nearest mean neighbor."""
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for index, group in enumerate(groups):
            if group["exposure"] / total_exposure >= minimum_exposure_percent:
                continue
            neighbors = [
                neighbor
                for neighbor in (index - 1, index + 1)
                if 0 <= neighbor < len(groups)
            ]
            nearest = min(
                neighbors,
                key=lambda neighbor: abs(groups[neighbor]["mean"] - group["mean"]),
            )
            _merge_category_groups(groups, index, nearest)
            groups.sort(key=lambda candidate: candidate["mean"])
            changed = True
            break


def _limit_category_groups(
    groups: list[dict[str, Any]],
    maximum_groups: int,
) -> None:
    """Merge the closest adjacent groups until the requested limit is met."""
    while len(groups) > maximum_groups:
        differences = [
            abs(groups[index + 1]["mean"] - groups[index]["mean"])
            for index in range(len(groups) - 1)
        ]
        closest = int(np.argmin(differences))
        _merge_category_groups(groups, closest, closest + 1)


def _category_group_outputs(
    groups: list[dict[str, Any]],
) -> tuple[dict[str, str], pl.DataFrame]:
    """Build the public category mapping and summary table."""
    mapping: dict[str, str] = {}
    rows = []
    for index, group in enumerate(groups, 1):
        label = f"G{index:02d}"
        mapping.update((level, label) for level in group["levels"])
        rows.append(
            {
                "group": label,
                "levels": str(group["levels"]),
                "exposure": group["exposure"],
                "mean_target": group["mean"],
            }
        )
    return mapping, pl.DataFrame(rows).sort("mean_target")


def suggest_category_groups(
    col: str,
    df: pl.DataFrame,
    y: pl.Series | np.ndarray | None,
    weights: pl.Series | np.ndarray | None = None,
    max_groups: int = 10,
    min_exposure_pct: float = 0.01,
    verbose: bool = True,
) -> tuple[dict[str, str], pl.DataFrame]:
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
    w_arr = _to_numpy(weights) if weights is not None else np.ones(len(df))
    total_w = max(float(w_arr.sum()), 1.0)
    groups = _initial_category_groups(s, y_arr, w_arr)
    _merge_small_category_groups(groups, total_w, min_exposure_pct)
    _limit_category_groups(groups, max_groups)
    level_to_group, summary = _category_group_outputs(groups)

    if verbose:
        print(f"\n  Category groups for '{col}' ({len(groups)} groups):")
        for row in summary.iter_rows(named=True):
            print(
                f"    {row['group']}: mean={row['mean_target']:.4f}  exp={row['exposure']:,.0f}  levels={row['levels']}"
            )

    return level_to_group, summary


# ── Monotonicity cost test ────────────────────────────────────────────────────


def _default_feature_columns(
    data: pl.DataFrame,
    target_column: str,
    weight_column: str | None,
) -> list[str]:
    """Select numeric and categorical modeling columns, excluding target/weight."""
    excluded = {target_column, weight_column} if weight_column else {target_column}
    return [
        column
        for column in data.columns
        if column not in excluded
        and (
            data[column].dtype in _NUMERIC_DTYPES or _is_str_or_cat(data[column].dtype)
        )
    ]


def _fit_lgb_rmse(
    data: pl.DataFrame,
    target: np.ndarray,
    weights: np.ndarray | None,
    feature_names: list[str],
    parameters: dict[str, Any],
    constraints: list[int] | None = None,
) -> float:
    """Fit one LightGBM model and return its in-sample weighted RMSE."""
    resolved_parameters = dict(parameters)
    if constraints is not None:
        resolved_parameters["monotone_constraints"] = constraints
    model = LGBRegressorPolars(**resolved_parameters)
    model.fit(data, target, sample_weight=weights, feature_name=feature_names)
    residuals = target - model.predict(data)
    mean_squared_error = (
        np.average(residuals**2, weights=weights)
        if weights is not None
        else np.mean(residuals**2)
    )
    return float(np.sqrt(mean_squared_error))


def _monotonicity_recommendation(
    positive_cost: float,
    negative_cost: float,
    threshold: float,
) -> str:
    """Choose the lowest-cost viable monotonic direction."""
    if positive_cost <= negative_cost and positive_cost <= threshold:
        return "increasing"
    if negative_cost < positive_cost and negative_cost <= threshold:
        return "decreasing"
    return "no_constraint"


def monotonicity_test(
    df: pl.DataFrame,
    target_col: str,
    var: str,
    weight_col: str | None = None,
    feature_cols: list[str] | None = None,
    n_estimators: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    variable_configs: dict[str, Any] | None = {},
    **lgb_params: Any,
) -> dict[str, Any]:
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
    if feature_cols is None:
        feature_cols = _default_feature_columns(df, target_col, weight_col)

    if var not in feature_cols:
        raise ValueError(f"'{var}' not in feature_cols.")

    X, encoded_names, col_index_map = _encode_features(
        df, feature_cols, variable_configs
    )
    y = df[target_col].to_numpy().astype(float)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None

    base_params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": random_state,
        **lgb_params,
    }

    rmse_free = _fit_lgb_rmse(X, y, w, encoded_names, base_params)

    n_features, var_indices = X.shape[1], col_index_map[var]
    constrained_rmses: dict[str, float] = {}
    for direction, label in [(1, "pos"), (-1, "neg")]:
        constraints = [0] * n_features
        for idx in var_indices:
            constraints[idx] = direction
        constrained_rmses[label] = _fit_lgb_rmse(
            X,
            y,
            w,
            encoded_names,
            base_params,
            constraints,
        )

    cost_pos = constrained_rmses["pos"] - rmse_free
    cost_neg = constrained_rmses["neg"] - rmse_free
    threshold = rmse_free * 0.01  # 1% RMSE increase is "trivial"

    recommended = _monotonicity_recommendation(cost_pos, cost_neg, threshold)

    if verbose:
        print(f"\n  Monotonicity test — '{var}'")
        print(f"    Unconstrained RMSE : {rmse_free:.6f}")
        print(
            f"    Monotone (+1) RMSE : {constrained_rmses['pos']:.6f}  cost = {cost_pos / rmse_free:+.2%}"
        )
        print(
            f"    Monotone (-1) RMSE : {constrained_rmses['neg']:.6f}  cost = {cost_neg / rmse_free:+.2%}"
        )
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


def _boruta_hit_counts(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    encoded_names: list[str],
    feature_columns: list[str],
    column_indices: dict[str, list[int]],
    parameters: dict[str, Any],
    iterations: int,
    random_state: int,
) -> dict[str, int]:
    """Run repeated shadow-feature fits and count wins per original variable."""
    real_feature_count = features.shape[1]
    random = np.random.RandomState(random_state)
    hit_counts = {column: 0 for column in feature_columns}
    shadow = np.empty_like(features)
    augmented = np.empty(
        (features.shape[0], real_feature_count * 2),
        dtype=features.dtype,
    )
    augmented[:, :real_feature_count] = features
    augmented_names = encoded_names + [f"_shadow_{name}" for name in encoded_names]
    for _ in range(iterations):
        for index in range(real_feature_count):
            shadow[:, index] = random.permutation(features[:, index])
        augmented[:, real_feature_count:] = shadow
        model = LGBRegressorPolars(**parameters)
        model.fit(
            augmented,
            target,
            sample_weight=weights,
            feature_name=augmented_names,
        )
        importance = model.feature_importances_
        real_importance = importance[:real_feature_count]
        maximum_shadow = np.percentile(importance[real_feature_count:], 95)
        for column in feature_columns:
            column_importance = sum(
                float(real_importance[index]) for index in column_indices[column]
            )
            if column_importance > maximum_shadow:
                hit_counts[column] += 1
    return hit_counts


def boruta_select(
    df: pl.DataFrame,
    target_col: str,
    weight_col: str | None = None,
    feature_cols: list[str] | None = None,
    n_estimators: int = 100,
    n_iterations: int = 20,
    threshold: float = 0.05,
    random_state: int = 42,
    variable_configs: dict[str, Any] | None = {},
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
    spec = importlib.util.find_spec("lightgbm")
    if spec is None:
        raise ImportError(
            "LightGBM is required for residual GBM.\n"
            "Install it with:  pip install lightgbm"
        )

    if feature_cols is None:
        feature_cols = _default_feature_columns(df, target_col, weight_col)

    X, encoded_names, col_index_map = _encode_features(
        df, feature_cols, variable_configs
    )
    y = df[target_col].to_numpy().astype(float)
    w = df[weight_col].to_numpy().astype(float) if weight_col else None
    X_np = X.to_numpy()
    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "verbose": -1,
        "n_jobs": -1,
        "importance_type": "gain",
    }
    params.update(lgb_params)
    hit_counts = _boruta_hit_counts(
        X_np,
        y,
        w,
        encoded_names,
        feature_cols,
        col_index_map,
        params,
        n_iterations,
        random_state,
    )

    rows = [
        {
            "variable": col_name,
            "pass_rate": hit_counts[col_name] / n_iterations,
            "selected": hit_counts[col_name] / n_iterations >= (1.0 - threshold),
        }
        for col_name in feature_cols
    ]
    return pl.DataFrame(rows).sort("pass_rate", descending=True)
