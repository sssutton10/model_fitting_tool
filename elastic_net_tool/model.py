"""Model fitting using glum for elastic net GLMs (polars backend)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import polars as pl

try:
    from glum import (
        GeneralizedLinearRegressor,
        GeneralizedLinearRegressorCV,
        TweedieDistribution,
        LogLink
    )
except ImportError as e:
    raise ImportError("glum is required: pip install glum") from e

from .variable import Preprocessor, VariableConfig, default_config, make_bin_labels

_ZERO_THRESHOLD = 1e-10


def _factor_dict(ft_v: pl.DataFrame) -> Dict[str, float]:
    return dict(zip(ft_v["Level"].to_list(), ft_v["Factor"].to_list()))


def _apply_factors(
    level_arr: np.ndarray,
    fdict: Dict[str, float],
    V: str,
    missing_factor: float,
) -> np.ndarray:
    factors = np.array([fdict.get(lv, float("nan")) for lv in level_arr])
    nan_mask = np.isnan(factors)
    if nan_mask.any():
        unseen = list(np.unique(level_arr[nan_mask]))
        print(
            f"  [WARN] {V}: {int(nan_mask.sum())} row(s) have unseen levels "
            f"{unseen} -> factor {missing_factor}"
        )
        factors[nan_mask] = missing_factor
    return factors


def _resolve_level_arr(
    V: str,
    X: pl.DataFrame,
    Xt: Optional[pl.DataFrame],
    cols_set: set,
    n: int,
    p: Optional[Dict[str, Any]],
) -> np.ndarray:
    if p is None:
        return np.array(X[V].cast(pl.String).to_list(), dtype=object)

    if p.get("is_categorical") and p.get("encoding") == "onehot":
        dropped = p.get("dropped_category", "")
        level_arr = np.full(n, dropped, dtype=object)
        for cat in p["categories"]:
            feat = f"{V}_{cat}"
            if feat in cols_set:
                level_arr[Xt[feat].to_numpy().astype(bool)] = str(cat)
        return level_arr

    if "bin_edges" in p:
        dropped_bin = p.get("dropped_bin", 0)
        all_labels = p.get("bin_labels")
        base_lbl = all_labels[dropped_bin]
        level_arr = np.full(n, base_lbl, dtype=object)
        missing_feat = f"{V}_missing"
        if missing_feat in cols_set:
            level_arr[Xt[missing_feat].to_numpy().astype(bool)] = "Missing"
        for i, label in enumerate(all_labels):
            if i == dropped_bin:
                continue
            feat = f"{V}_{label}"
            if feat in cols_set:
                level_arr[Xt[feat].to_numpy().astype(bool)] = label
        return level_arr

    # Pure continuous or unrecognised encoding — fall back to direct lookup
    return np.array(X[V].cast(pl.String).to_list(), dtype=object)


# ── Factor model version (Excel-based) ───────────────────────────────────────

@dataclass
class FactorModelVersion:
    """
    Factor-table model version loaded from an Excel workbook.

    Predictions are computed by looking up each row's level for every variable
    in the factor table and multiplying the matched factors together.

    Variables covered by a fitted :class:`Preprocessor` (via ``preprocessor_vars``)
    use the same bin / category label strings as
    :meth:`ModelingTool.relativities_table`.  All other variables are resolved
    by direct string match on the raw column value (suitable for pre-banded or
    categorical columns).

    Attributes
    ----------
    name : str
    variables : list of str
        All variables in the factor table (excluding any ``"intercept"`` row).
    factor_table : pl.DataFrame
        Columns ``Variable``, ``Level``, ``Factor``.
    preprocessor : Preprocessor, optional
        Fitted preprocessor for level-string resolution of numeric/binned and
        one-hot categorical variables.  ``None`` in standalone mode.
    preprocessor_vars : list of str
        Subset of *variables* whose levels are resolved via the preprocessor.
    train_predictions : np.ndarray
    alpha, l1_ratio : always ``None`` (stubs for ``list_versions`` compatibility)
    feature_names : always ``[]``
    coefficients : always empty DataFrame
    """

    name: str
    variables: List[str]
    factor_table: pl.DataFrame
    preprocessor: Optional[Any]       # Optional[Preprocessor]
    preprocessor_vars: List[str]
    train_predictions: np.ndarray
    offset_col: Optional[str] = None

    # Stubs — keep list_versions / compare_models happy
    alpha: Optional[float] = None
    l1_ratio: Optional[float] = None
    feature_names: List[str] = field(default_factory=list)
    coefficients: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame({"feature": [], "coefficient": []})
    )
    fit_info: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pl.DataFrame, missing_factor: float = 1.0, offset: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Score *X* by factor-table lookup and return a numpy array of predictions.

        For each variable the method determines each row's level string, looks
        up the corresponding factor, and multiplies across all variables.

        Parameters
        ----------
        X : pl.DataFrame
            Data to score.  Must contain every variable not covered by the
            preprocessor as a raw column.
        missing_factor : float
            Factor applied to any level not found in the table (default 1.0).
            A warning is printed naming the variable and unseen levels.
        """
        Xt = self.preprocessor.transform(X) if (self.preprocessor and self.preprocessor_vars) else None
        cols_set = set(Xt.columns) if Xt is not None else set()
        n = len(X)
        product = np.ones(n, dtype=float)

        if offset is not None:
            product *= offset

        factor_by_var = {
            grp[0]: df.select(["Level", "Factor"])
            for grp, df in self.factor_table.group_by("Variable")
        }

        for V in self.variables:
            p = self.preprocessor._params.get(V) if (V in self.preprocessor_vars and Xt is not None) else None
            level_arr = _resolve_level_arr(V, X, Xt, cols_set, n, p)
            fdict = _factor_dict(factor_by_var[V]) if V in factor_by_var else {}
            product *= _apply_factors(level_arr, fdict, V, missing_factor)

        icept = factor_by_var.get("intercept")
        if icept is not None and len(icept) > 0:
            product *= float(icept["Factor"][0])

        return product


# ── Model version ─────────────────────────────────────────────────────────────

@dataclass
class ModelVersion:
    """
    Container for a single fitted model version.

    Attributes
    ----------
    name : str
    variables : list of str
    preprocessor : Preprocessor
    glm : GeneralizedLinearRegressor
    feature_names : list of str
    coefficients : pl.Series  (index-named via schema; includes 'intercept')
    alpha, l1_ratio : float
    family, link : str or glum distribution
    train_predictions : np.ndarray  (aligned with training data rows)
    fit_info : dict
    """

    name: str
    variables: List[str]
    preprocessor: Preprocessor
    glm: Any
    feature_names: List[str]
    coefficients: pl.DataFrame       # columns: ['feature', 'coefficient']
    alpha: float
    l1_ratio: float
    family: Any
    link: str
    train_predictions: np.ndarray
    fit_info: Dict[str, Any] = field(default_factory=dict)
    cv_stability: Optional[pl.DataFrame] = None
    tweedie_power: Optional[float] = 1.50
    gradient_tol: Optional[float] = None

    def predict(self, X: pl.DataFrame, offset:Optional[np.ndarray] = None) -> np.ndarray:
        """Transform *X* through the preprocessor and return model predictions."""
        Xt = self.preprocessor.transform(X).to_numpy().astype(float)
        return self.glm.predict(Xt, offset=offset)

    # def coefficient_table(self) -> pl.DataFrame:
    #     """Return coefficients sorted by descending absolute value."""
    #     return (
    #         self.coefficients
    #         .with_columns(pl.col("coefficient").abs().alias("_abs"))
    #         .sort("_abs", descending=True)
    #         .drop("_abs")
    #     )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_family(family: Any, tweedie_power: float = 1.5) -> Any:
    if family is None:
        return TweedieDistribution(power=tweedie_power)
    if isinstance(family, str):
        name = family.lower()
        if name == "tweedie":
            return TweedieDistribution(power=tweedie_power)
        return name
    return family


def _build_preprocessor(
    variables: List[str],
    X: pl.DataFrame,
    configs: Optional[Dict[str, VariableConfig]] = None,
) -> Preprocessor:
    """Build (unfitted) Preprocessor, filling absent variables with defaults."""
    cfgs = []
    for col in variables:
        if col in configs:
            cfgs.append(configs[col])
        else:
            # For derived/multi-input variables, col might not be in X
            if col in X.columns:
                cfgs.append(default_config(col, X[col]))
            else:
                raise KeyError(
                    f"Variable '{col}' is not in the DataFrame and has no "
                    "VariableConfig registered. Add it with tool.add_variable()."
                )
    return Preprocessor(cfgs)


def _extract_coefficients(
    glm: GeneralizedLinearRegressor, feature_names: List[str]
) -> pl.DataFrame:
    features = ["intercept"] + feature_names
    values = [float(glm.intercept_)] + [float(v) for v in glm.coef_]
    return pl.DataFrame({"feature": features, "coefficient": values})


def _build_glm(
    family: Any,
    link: str,
    alpha: float,
    l1_ratio: float,
    fit_intercept: bool,
    max_iter: int,
    gradient_tol: Optional[float] = None
) -> GeneralizedLinearRegressor:
    return GeneralizedLinearRegressor(
        family=family, link=link,
        alpha=alpha, l1_ratio=l1_ratio,
        fit_intercept=fit_intercept, max_iter=max_iter, scale_predictors=True,
        gradient_tol=gradient_tol
    )


def _geometric_mean_signed(values: np.ndarray) -> float:
    """
    Signed geometric mean of *values*.

    For each fold coefficient:
    - Take geometric mean of the absolute values of nonzero entries.
    - Multiply by the majority sign across folds.
    - Return 0 if all values are effectively zero.
    """
    if len(values) == 0:
        return 0.0
    abs_vals = np.abs(values)
    mask = abs_vals > _ZERO_THRESHOLD
    nonzero = abs_vals[mask]
    if len(nonzero) == 0:
        return 0.0
    geo_abs = float(np.exp(np.mean(np.log(nonzero))))
    signs = np.sign(values[mask])
    majority_sign = 1.0 if np.sum(signs) >= 0 else -1.0
    return geo_abs * majority_sign


# ── Model fitting ─────────────────────────────────────────────────────────────

def fit_model(
    X: pl.DataFrame,
    y: np.ndarray,
    variables: List[str],
    version_name: str,
    configs: Optional[Dict[str, VariableConfig]] = {},
    weights: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    family: Any = None,
    link: Any = LogLink(),
    tweedie_power: float = 1.5,
    preprocessor: Optional[Preprocessor] = None,
    alpha: Optional[float] = None,
    l1_ratio: Union[float, List[float]] = 0.5,
    use_cv: bool = True,
    cv: Any = 5,
    max_iter: int = 1000,
    fit_intercept: bool = True,
    drop_reference: str = "max_weight",
    gradient_tol: Optional[float] = None
) -> ModelVersion:
    """
    Fit an elastic net GLM and return a :class:`ModelVersion`.

    Parameters
    ----------
    X : pl.DataFrame
    y : pl.Series
        Target (loss ratio).
    variables : list of str
        Predictor column names (or derived variable names with registered configs).
    version_name : str
    configs : dict
        Registered :class:`VariableConfig` objects.
    weights : pl.Series, optional
        Sample weights (exposure).
    family : str or glum distribution, optional
        Accepted strings: ``"tweedie"`` (default), ``"poisson"``, ``"gamma"``.
        Defaults to ``TweedieDistribution(power=tweedie_power)``.
    alpha : float, optional
        Fixed regularisation strength.  Ignored when ``use_cv=True``.
        Set to ``0`` for an unpenalised GLM.
    l1_ratio : float or list of float
        Elastic-net mixing.  List triggers CV search.
    use_cv : bool
        Cross-validate to select best ``alpha`` (and optionally ``l1_ratio``).
    cv : int or sklearn CV splitter
        Fold specification passed directly to ``GeneralizedLinearRegressorCV``.
        An ``int`` triggers stratified k-fold; a ``PredefinedSplit`` (or any
        other sklearn splitter) uses the provided fold assignments.
    alphas : np.ndarray, optional
        Custom alpha grid for CV.
    """
    family = _resolve_family(family, tweedie_power)

    prep = _build_preprocessor(variables, X, configs) if preprocessor is None else preprocessor

    if not prep._fitted:
        fit_weights = weights if drop_reference == "max_weight" else None
        prep.fit(X, weights=fit_weights)
    Xt = prep.transform(X).to_numpy().astype(float)
    feature_names = prep.get_feature_names()

    link = LogLink() if link is None else link

    if use_cv:
        cv_l1 = l1_ratio if isinstance(l1_ratio, list) else [l1_ratio]
        
        glm_cv = GeneralizedLinearRegressorCV(
            family=family,
            link=link,
            l1_ratio=cv_l1,
            cv=cv,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            scale_predictors=True,
            gradient_tol=gradient_tol
        )
        glm_cv.fit(Xt, y, sample_weight=weights, offset=offset)
        best_alpha = float(glm_cv.alpha_)
        best_l1 = float(glm_cv.l1_ratio_)

        glm = _build_glm(family, link, best_alpha, best_l1, fit_intercept, max_iter, gradient_tol)
        glm.fit(Xt, y, sample_weight=weights, offset=offset)
        cv_label = cv if isinstance(cv, int) else type(cv).__name__
        fit_info: Dict[str, Any] = {
            "cv_folds": cv_label,
            "cv_l1_ratio_grid": cv_l1,
        }
    else:
        best_alpha = alpha if alpha is not None else 0.0
        best_l1 = (l1_ratio if not isinstance(l1_ratio, list) else l1_ratio[0])
        glm = _build_glm(family, link, best_alpha, best_l1, fit_intercept, max_iter, gradient_tol)
        glm.fit(Xt, y, sample_weight=weights, offset=offset)
        fit_info = {}

    fit_info['Fit_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coef_df = _extract_coefficients(glm, feature_names)
    preds = glm.predict(Xt, offset=offset)

    return ModelVersion(
        name=version_name,
        variables=list(variables),
        preprocessor=prep,
        glm=glm,
        feature_names=feature_names,
        coefficients=coef_df,
        alpha=best_alpha,
        l1_ratio=best_l1,
        family=family,
        link=link,
        train_predictions=preds,
        fit_info=fit_info,
        tweedie_power=tweedie_power,
        gradient_tol=gradient_tol
    )


# ── CV stability ──────────────────────────────────────────────────────────────

def fit_cv_stability(
    X: pl.DataFrame,
    y: np.ndarray,
    variables: List[str],
    configs: Dict[str, VariableConfig],
    fold_col: str,
    weights: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    family: Any = None,
    link: str = "log",
    tweedie_power: float = 1.5,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    gradient_tol: Optional[float] = None,
    fit_intercept: bool = True,
    drop_reference: str = "max_weight",
) -> pl.DataFrame:
    """
    Evaluate coefficient stability by fitting the model on each CV fold.

    Each fold value in ``fold_col`` is used as the *test* fold; the model is
    trained on all other observations.  The preprocessing is fitted once on
    the full dataset (shared across folds) so that feature names are
    consistent.

    Parameters
    ----------
    fold_col : str
        Column in *X* whose values identify the test fold for each row
        (e.g. ``1``, ``2``, … ``5``).  Each unique value becomes one fold.
    family : str or glum distribution, optional
        Accepted strings: ``"tweedie"`` (default), ``"poisson"``, ``"gamma"``.
        Defaults to ``TweedieDistribution(power=tweedie_power)``.
    alpha, l1_ratio : float
        Fixed hyperparameters used for all fold fits.

    Returns
    -------
    pl.DataFrame
        Rows = one per fold (labelled by fold value) + three summary rows:
        ``'geomean'``, ``'std'``, ``'cv_pct'``.
        Columns = ``'fold'`` + one per coefficient (intercept first).
    """
    family = _resolve_family(family, tweedie_power)

    # Fit a reference preprocessor on the full dataset to fix feature names
    # (preprocessing params shared across folds — no data leakage for cap/bin
    # values, which is standard in insurance CV practice)
    feat_vars = [v for v in variables if v != fold_col]
    X_feats = X.drop(fold_col) if fold_col in X.columns else X

    ref_prep = _build_preprocessor(feat_vars, X_feats, configs)
    fit_weights = weights if drop_reference == "max_weight" else None
    ref_prep.fit(X_feats, weights=fit_weights)
    feature_names = ref_prep.get_feature_names()

    folds = sorted(X[fold_col].unique().to_list())
    records: List[Dict[str, Any]] = []

    Xt_full = ref_prep.transform(X_feats).to_numpy().astype(float)

    for fold_val in folds:
        train_mask = (X[fold_col] != fold_val).to_numpy()

        Xt = Xt_full[train_mask]
        y_train = y[train_mask]
        w_train = weights[train_mask] if weights is not None else None
        offset_train = offset[train_mask] if offset is not None else None

        glm = _build_glm(family, link, alpha, l1_ratio, fit_intercept, max_iter, gradient_tol)
        glm.fit(Xt, y_train, sample_weight=w_train, offset=offset_train)

        row: Dict[str, Any] = {"fold": f"fold_{str(fold_val)}", "intercept": float(glm.intercept_)}
        for name, val in zip(feature_names, glm.coef_):
            row[name] = float(val)
        records.append(row)

    stability = pl.DataFrame(records)
    fold_numeric = stability.drop("fold")
    fold_matrix = fold_numeric.to_numpy()

    geomean_row: Dict[str, Any] = {"fold": "geomean"}
    std_row: Dict[str, Any] = {"fold": "std"}
    cv_row: Dict[str, Any] = {"fold": "cv_pct"}

    for col, vals in zip(fold_numeric.columns, fold_matrix.T):
        gm = _geometric_mean_signed(vals)
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv_pct = abs(sd / gm) * 100 if abs(gm) > _ZERO_THRESHOLD else float("nan")
        geomean_row[col] = gm
        std_row[col] = sd
        cv_row[col] = cv_pct

    summary = pl.DataFrame([geomean_row, std_row, cv_row])
    summary = summary.select(stability.columns)

    return pl.concat([stability, summary])
