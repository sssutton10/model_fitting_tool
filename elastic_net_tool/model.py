"""Model fitting using glum for elastic net GLMs (polars backend)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

try:
    from glum import (
        GeneralizedLinearRegressor,
        GeneralizedLinearRegressorCV,
        LogLink,
        TweedieDistribution,
    )
except ImportError as e:
    raise ImportError("glum is required: pip install glum") from e

from .variable import (
    FittedBinnedNumericParams,
    FittedCategoricalParams,
    FittedParams,
    Preprocessor,
    VariableConfig,
    default_config,
)

_ZERO_THRESHOLD = 1e-10


def _factor_dict(ft_v: pl.DataFrame) -> dict[str, float]:
    return dict(zip(ft_v["Level"].to_list(), ft_v["Factor"].to_list()))


def _apply_factors(
    level_arr: np.ndarray,
    fdict: dict[str, float],
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
    Xt: pl.DataFrame | None,
    cols_set: set,
    n: int,
    p: FittedParams | Mapping[str, Any] | None,
) -> np.ndarray:
    if p is None:
        return np.array(X[V].cast(pl.String).to_list(), dtype=object)

    if isinstance(p, Mapping):
        if p.get("is_categorical") and p.get("encoding") == "onehot":
            p = FittedCategoricalParams(
                True,
                p.get("impute_val"),
                "onehot",
                tuple(p.get("categories", ())),
                p.get("dropped_category"),
            )
        elif "bin_edges" in p:
            p = FittedBinnedNumericParams(
                False,
                p.get("impute_val"),
                bin_edges=np.asarray(p.get("bin_edges", ()), dtype=float),
                bin_labels=tuple(p.get("bin_labels", ())),
                dropped_bin=int(p.get("dropped_bin", 0)),
                has_sentinel_bin=bool(p.get("has_sentinel_bin", True)),
            )
    if isinstance(p, FittedCategoricalParams) and p.encoding == "onehot":
        return _categorical_level_arr(V, Xt, cols_set, n, p)
    if isinstance(p, FittedBinnedNumericParams):
        return _binned_level_arr(V, Xt, cols_set, n, p)

    # Pure continuous or unrecognised encoding — fall back to direct lookup
    return np.array(X[V].cast(pl.String).to_list(), dtype=object)


def _categorical_level_arr(
    variable: str,
    transformed: pl.DataFrame,
    columns: set[str],
    row_count: int,
    params: FittedCategoricalParams,
) -> np.ndarray:
    """Resolve one-hot rows to their original categorical levels."""
    base = f"{params.dropped_category} (base)" if params.dropped_category else ""
    levels = np.full(row_count, base, dtype=object)
    for category in params.categories:
        feature = f"{variable}_{category}"
        if feature in columns:
            levels[transformed[feature].to_numpy().astype(bool)] = str(category)
    return levels


def _binned_level_arr(
    variable: str,
    transformed: pl.DataFrame,
    columns: set[str],
    row_count: int,
    params: FittedBinnedNumericParams,
) -> np.ndarray:
    """Resolve numeric dummy rows to their fitted bin labels."""
    levels = np.full(
        row_count,
        f"{params.bin_labels[params.dropped_bin]} (base)",
        dtype=object,
    )
    missing_feature = f"{variable}_missing"
    if missing_feature in columns:
        levels[transformed[missing_feature].to_numpy().astype(bool)] = "Missing"
    for index, label in enumerate(params.bin_labels):
        if index == params.dropped_bin:
            continue
        feature = f"{variable}_{label}"
        if feature in columns:
            levels[transformed[feature].to_numpy().astype(bool)] = label
    return levels


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
    variables: list[str]
    factor_table: pl.DataFrame
    preprocessor: Any | None  # Optional[Preprocessor]
    preprocessor_vars: list[str]
    train_predictions: np.ndarray
    offset_col: str | None = None

    # Stubs — keep list_versions / compare_models happy
    alpha: float | None = None
    l1_ratio: float | None = None
    feature_names: list[str] = field(default_factory=list)
    coefficients: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame({"feature": [], "coefficient": []})
    )
    fit_info: dict[str, Any] = field(default_factory=dict)

    def predict(
        self,
        X: pl.DataFrame,
        missing_factor: float = 1.0,
        offset: np.ndarray | None = None,
    ) -> np.ndarray:
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
        offset : np.ndarray, optional
            Per-row multiplicative offset applied to the final prediction
            (response scale, i.e. already exponentiated).  The factor product
            is multiplied element-wise by ``offset``.
        """
        Xt = (
            self.preprocessor.transform(X)
            if (self.preprocessor and self.preprocessor_vars)
            else None
        )
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
            p = (
                self.preprocessor._params.get(V)
                if (V in self.preprocessor_vars and Xt is not None)
                else None
            )
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
    variables: list[str]
    preprocessor: Preprocessor
    glm: Any
    feature_names: list[str]
    coefficients: pl.DataFrame  # columns: ['feature', 'coefficient']
    alpha: float
    l1_ratio: float
    family: Any
    link: str
    train_predictions: np.ndarray
    fit_info: dict[str, Any] = field(default_factory=dict)
    cv_stability: pl.DataFrame | None = None
    tweedie_power: float | None = 1.50
    gradient_tol: float | None = None

    def predict(self, X: pl.DataFrame, offset: np.ndarray | None = None) -> np.ndarray:
        """Transform *X* through the preprocessor and return model predictions.

        Parameters
        ----------
        offset : np.ndarray, optional
            Per-row offset on the **linear predictor (log) scale**.  For a
            log-link GLM this means ``log(existing_model_prediction)``.
            Passed directly to glum's ``predict``.
        """
        Xt = self.preprocessor.transform(X).to_numpy().astype(float)
        return self.glm.predict(Xt, offset=offset)

    def coefficient_table(self) -> pl.DataFrame:
        """Return coefficients sorted by descending absolute value."""
        return (
            self.coefficients.with_columns(pl.col("coefficient").abs().alias("_abs"))
            .sort("_abs", descending=True)
            .drop("_abs")
        )


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
    variables: list[str],
    X: pl.DataFrame,
    configs: dict[str, VariableConfig] | None = None,
) -> Preprocessor:
    """Build (unfitted) Preprocessor, filling absent variables with defaults."""
    config_map = configs or {}
    required: dict[str, VariableConfig] = {}

    for col in variables:
        _collect_required_config(col, X, config_map, required, ())
    return Preprocessor(list(required.values()), output_cols=variables)


def _collect_required_config(
    column: str,
    data: pl.DataFrame,
    config_map: dict[str, VariableConfig],
    required: dict[str, VariableConfig],
    visiting: tuple[str, ...],
) -> None:
    """Collect a variable config and its derived dependencies exactly once."""
    if column in required:
        return
    if column in visiting:
        cycle = " -> ".join(visiting[visiting.index(column) :] + (column,))
        raise ValueError(f"Circular dependency in derived variable chain: {cycle}.")
    config = config_map.get(column)
    if config is None:
        _collect_default_config(column, data, required)
        return
    for dependency in config.input_cols or []:
        if dependency in config_map:
            _collect_required_config(
                dependency,
                data,
                config_map,
                required,
                visiting + (column,),
            )
        elif dependency not in data.columns:
            raise KeyError(
                f"Input column '{dependency}' for derived variable '{column}' "
                "is not in the DataFrame and has no registered config."
            )
    required[column] = config


def _collect_default_config(
    column: str,
    data: pl.DataFrame,
    required: dict[str, VariableConfig],
) -> None:
    """Collect a default config for a raw DataFrame column."""
    if column not in data.columns:
        raise KeyError(
            f"Variable '{column}' is not in the DataFrame and has no "
            "VariableConfig registered. Add it with tool.add_variable()."
        )
    required[column] = default_config(column, data[column])


def _extract_coefficients(
    glm: GeneralizedLinearRegressor, feature_names: list[str]
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
    gradient_tol: float | None = None,
) -> GeneralizedLinearRegressor:
    return GeneralizedLinearRegressor(
        family=family,
        link=link,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        scale_predictors=True,
        gradient_tol=gradient_tol,
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


def _fit_estimator(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    offset: np.ndarray | None,
    family: Any,
    link: Any,
    alpha: float | None,
    l1_ratio: float | list[float],
    use_cv: bool,
    cv: Any,
    fit_intercept: bool,
    max_iter: int,
    gradient_tol: float | None,
    n_jobs: int | None,
    alphas: np.ndarray | None,
) -> tuple[GeneralizedLinearRegressor, float, float, dict[str, Any]]:
    """Fit either the CV-selected or fixed-penalty GLM."""
    if not use_cv:
        best_alpha = alpha if alpha is not None else 0.0
        best_l1 = l1_ratio if not isinstance(l1_ratio, list) else l1_ratio[0]
        model = _build_glm(
            family,
            link,
            best_alpha,
            best_l1,
            fit_intercept,
            max_iter,
            gradient_tol,
        )
        model.fit(design, target, sample_weight=weights, offset=offset)
        return model, best_alpha, best_l1, {}

    cv_l1 = l1_ratio if isinstance(l1_ratio, list) else [l1_ratio]
    cv_model = GeneralizedLinearRegressorCV(
        family=family,
        link=link,
        l1_ratio=cv_l1,
        cv=cv,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        scale_predictors=True,
        gradient_tol=gradient_tol,
        n_jobs=n_jobs,
        alphas=alphas,
    )
    cv_model.fit(design, target, sample_weight=weights, offset=offset)
    best_alpha = float(cv_model.alpha_)
    best_l1 = float(cv_model.l1_ratio_)
    model = _build_glm(
        family,
        link,
        best_alpha,
        best_l1,
        fit_intercept,
        max_iter,
        gradient_tol,
    )
    model.fit(design, target, sample_weight=weights, offset=offset)
    cv_label = cv if isinstance(cv, int) else type(cv).__name__
    return (
        model,
        best_alpha,
        best_l1,
        {
            "cv_folds": cv_label,
            "cv_l1_ratio_grid": cv_l1,
        },
    )


def fit_model(
    X: pl.DataFrame,
    y: np.ndarray,
    variables: list[str],
    version_name: str,
    configs: dict[str, VariableConfig] | None = {},
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    family: Any = None,
    link: Any = LogLink(),
    tweedie_power: float = 1.5,
    preprocessor: Preprocessor | None = None,
    alpha: float | None = None,
    l1_ratio: float | list[float] = 0.5,
    use_cv: bool = True,
    cv: Any = 5,
    max_iter: int = 1000,
    fit_intercept: bool = True,
    drop_reference: str = "max_weight",
    gradient_tol: float | None = None,
    n_jobs: int | None = None,
    alphas: np.ndarray | None = None,
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
    offset : np.ndarray, optional
        Per-row offset on the **linear predictor (log) scale**.  For a
        log-link model this is ``log(existing_model_prediction)``.  Passed
        directly to glum's ``fit`` and ``predict``.
    family : str or glum distribution, optional
        Accepted strings: ``"tweedie"`` (default), ``"poisson"``, ``"gamma"``.
        Defaults to ``TweedieDistribution(power=tweedie_power)``.
    alpha : float, optional
        Fixed regularisation strength.  Ignored when ``use_cv=True``.
        Set to ``0`` for an unpenalised GLM.
    l1_ratio : float or list of float
        Elastic-net mixing.  list triggers CV search.
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

    prep = (
        _build_preprocessor(variables, X, configs)
        if preprocessor is None
        else preprocessor
    )
    emitted = prep._emitted_cols()
    if emitted != variables:
        raise ValueError(
            "The supplied preprocessor must emit exactly the requested variables "
            f"in order. Requested {variables}, but it emits {emitted}."
        )

    if not prep._fitted:
        fit_weights = weights if drop_reference == "max_weight" else None
        prep.fit(X, weights=fit_weights)
    Xt = prep.transform(X).to_numpy().astype(float)
    feature_names = prep.get_feature_names()

    link = LogLink() if link is None else link
    use_cv = use_cv and alpha is None
    glm, best_alpha, best_l1, fit_info = _fit_estimator(
        Xt,
        y,
        weights,
        offset,
        family,
        link,
        alpha,
        l1_ratio,
        use_cv,
        cv,
        fit_intercept,
        max_iter,
        gradient_tol,
        n_jobs,
        alphas,
    )
    fit_info["Fit_Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        gradient_tol=gradient_tol,
    )


# ── CV stability ──────────────────────────────────────────────────────────────


def _stability_preprocessor(
    features: list[str],
    data: pl.DataFrame,
    configs: dict[str, VariableConfig],
    weights: np.ndarray | None,
    drop_reference: str,
    preprocessor: Preprocessor | None,
) -> Preprocessor:
    """Resolve and fit the shared reference preprocessor for CV stability."""
    if preprocessor is not None:
        return preprocessor
    resolved = _build_preprocessor(features, data, configs)
    fit_weights = weights if drop_reference == "max_weight" else None
    resolved.fit(data, weights=fit_weights)
    return resolved


def _fit_stability_fold(
    fold_value: Any,
    fold_values: pl.Series,
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    offset: np.ndarray | None,
    feature_names: list[str],
    family: Any,
    link: Any,
    alpha: float,
    l1_ratio: float,
    fit_intercept: bool,
    max_iter: int,
    gradient_tol: float | None,
) -> dict[str, Any]:
    """Fit one training fold and return its coefficient record."""
    train_mask = (fold_values != fold_value).to_numpy()
    model = _build_glm(
        family,
        link,
        alpha,
        l1_ratio,
        fit_intercept,
        max_iter,
        gradient_tol,
    )
    model.fit(
        design[train_mask],
        target[train_mask],
        sample_weight=weights[train_mask] if weights is not None else None,
        offset=offset[train_mask] if offset is not None else None,
    )
    record: dict[str, Any] = {
        "fold": str(fold_value),
        "intercept": float(model.intercept_),
    }
    record.update(
        (name, float(value)) for name, value in zip(feature_names, model.coef_)
    )
    return record


def _stability_summary(stability: pl.DataFrame) -> pl.DataFrame:
    """Build geometric-mean, standard-deviation, and CV summary rows."""
    numeric = stability.drop("fold")
    rows = {
        "geomean": {"fold": "geomean"},
        "std": {"fold": "std"},
        "cv_pct": {"fold": "cv_pct"},
    }
    for column, values in zip(numeric.columns, numeric.to_numpy().T):
        geometric_mean = _geometric_mean_signed(values)
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        coefficient_variation = (
            abs(standard_deviation / geometric_mean) * 100
            if abs(geometric_mean) > _ZERO_THRESHOLD
            else float("nan")
        )
        rows["geomean"][column] = geometric_mean
        rows["std"][column] = standard_deviation
        rows["cv_pct"][column] = coefficient_variation
    return pl.DataFrame(list(rows.values())).select(stability.columns)


def fit_cv_stability(
    X: pl.DataFrame,
    y: np.ndarray,
    variables: list[str],
    configs: dict[str, VariableConfig],
    fold_col: str,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    family: Any = None,
    link: str = "log",
    tweedie_power: float = 1.5,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    gradient_tol: float | None = None,
    fit_intercept: bool = True,
    drop_reference: str = "max_weight",
    preprocessor: Any = None,
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
    offset : np.ndarray, optional
        Per-row offset on the **linear predictor (log) scale**.  Sliced
        per fold alongside ``y`` and ``weights``.
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
    y = np.asarray(y, dtype=float)
    weights = None if weights is None else np.asarray(weights, dtype=float)
    offset = None if offset is None else np.asarray(offset, dtype=float)
    resolved_family = _resolve_family(family, tweedie_power)
    feature_variables = [variable for variable in variables if variable != fold_col]
    feature_data = X.drop(fold_col) if fold_col in X.columns else X
    reference = _stability_preprocessor(
        feature_variables,
        feature_data,
        configs,
        weights,
        drop_reference,
        preprocessor,
    )
    feature_names = reference.get_feature_names()
    design = reference.transform(feature_data).to_numpy().astype(float)
    fold_values = X[fold_col]
    records = [
        _fit_stability_fold(
            fold_value,
            fold_values,
            design,
            y,
            weights,
            offset,
            feature_names,
            resolved_family,
            link,
            alpha,
            l1_ratio,
            fit_intercept,
            max_iter,
            gradient_tol,
        )
        for fold_value in sorted(fold_values.unique().to_list())
    ]
    stability = pl.DataFrame(records)
    return pl.concat([stability, _stability_summary(stability)])
