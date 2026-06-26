"""ModelingTool — main orchestration class for elastic net insurance GLMs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .io_utils import load_version, save_version
from .metrics import compare_metrics, compute_metrics, double_lift_score, double_lift_table, gini_coefficient
from .model import FactorModelVersion, ModelVersion, fit_cv_stability, fit_model, _build_preprocessor
from tabulate import tabulate
from .plots import (
    _resolve_level,
    _sort_labels,
    ae_chart,
    coefficient_plot,
    cv_stability_plot,
    double_lift_chart,
    residual_chart,
    univariate_plot,
    decile_lift_chart
)
from .variable import Preprocessor, VariableConfig, default_config, make_bin_labels


# ── Module-level helpers for relativities_table ───────────────────────────────

def _weighted_feat_map(
    Xt_df: pl.DataFrame,
    feats: List[str],
    w_arr: np.ndarray,
) -> Dict[str, float]:
    """Sum of exposure weight per dummy feature column: {feat_name: total_weight}."""
    return {
        feat: float((Xt_df[feat].to_numpy() * w_arr).sum())
        for feat in feats
        if feat in Xt_df.columns
    }


def _make_row(
    var_col: str,
    level: str,
    weight: float,
    train_coef: float,
    fold_names: List[str],
    fold_coef_map: Dict[str, Dict[str, float]],
    feat: Optional[str],
    *,
    calib_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """Build one row dict for the relativities table."""
    row: Dict[str, Any] = {
        "variable": var_col,
        "level": level,
        "weight": weight,
        "train_coef": train_coef,
    }
    if calib_weight is not None:
        row["calib_weight"] = calib_weight
    for fn in fold_names:
        row[fn] = fold_coef_map[fn].get(feat, 0.0) if feat is not None else 0.0
    return row


class ModelingTool:
    """
    End-to-end elastic net GLM tool for insurance loss ratio modelling.

    All DataFrame arguments use **polars**.

    Workflow
    --------
    1. **Variable creation** — :meth:`add_variable` + :meth:`univariate_plot`.
    2. **Model fitting** — :meth:`fit_model` (CV or fixed alpha).
       :meth:`fit_cv_stability` evaluates coefficient stability across folds
       defined by a user-supplied column.
    3. **Variable evaluation** — :meth:`ae_chart`.
    4. **Model comparison** — :meth:`compare_models`.
    5. **Persistence** — :meth:`save` / :meth:`load`.

    Parameters
    ----------
    data : pl.DataFrame
        Source dataset.
    target_col : str
        Column name of the loss ratio target.
    weight_col : str, optional
        Column name of the exposure weights (e.g. earned premium).
    family : str or glum distribution, optional
        GLM family.  Defaults to ``TweedieDistribution(power=tweedie_power)``.
    link : str
        GLM link function (default ``'log'``).
    tweedie_power : float
        Tweedie variance power when family is Tweedie (default ``1.5``).
    """

    def __init__(
        self,
        data: pl.DataFrame,
        target_col: str,
        weight_col: Optional[str] = None,
        offset_col: Optional[str] = None,
        link: Any = None,
        drop_reference: str = "max_weight",
        cv_column: Optional[str] = None,
        current_version: Optional[str] = None,
        base_version: Optional[str] = None
    ):
        """
        Parameters
        ----------
        drop_reference : {'max_weight', 'first'}
            Controls which level is dropped when one-hot encoding categorical
            variables.

            ``'max_weight'`` (default) — drop the level with the highest total
            exposure weight.  This is typically the most common class and
            makes coefficient interpretation more natural (every other
            coefficient is a relativity vs the dominant group).

            ``'first'`` — drop the first level alphabetically (legacy
            behaviour, used when no weight column is available or for
            reproducibility with older runs).
        cv_column : str, optional
            Column in *data* whose values indicate which CV fold each
            observation belongs to for hyperparameter selection
            (alpha / l1_ratio).  Any hashable value is accepted as a fold
            label; the column is converted to a :class:`sklearn.model_selection.PredefinedSplit`
            automatically.  When set, this becomes the default ``cv`` for
            every :meth:`fit_model` call.  Pass an explicit ``cv=<int>`` to
            :meth:`fit_model` to override with k-fold for that specific fit.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError(f"data must be a polars DataFrame, got {type(data).__name__}.")
        if cv_column is not None and cv_column not in data.columns:
            raise ValueError(
                f"cv_column '{cv_column}' not found in data.  "
                f"Available columns: {data.columns}"
            )
        self.data = data
        self.target_col = target_col
        self.weight_col = weight_col
        self.offset_col = offset_col
        self.drop_reference = drop_reference
        self.cv_column = cv_column
        self.link = link
        self.variable_configs: Dict[str, VariableConfig] = {}
        self.model_versions: Dict[str, ModelVersion] = {}

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def _y(self) -> pl.Series:
        return self.data[self.target_col]
    
    @property
    def _y_array(self) -> np.ndarray:
        """Target values as a float64 numpy array."""
        return self._y.to_numpy().astype(float)

    @property
    def _weights(self) -> Optional[pl.Series]:
        return self.data[self.weight_col] if self.weight_col else None

    @property
    def _weights_array(self) -> Optional[np.ndarray]:
        """Exposure weights as a float64 numpy array, or None if no weight column."""
        return self._weights.to_numpy().astype(float) if self._weights is not None else None
    
    @property
    def _offset_array(self) -> Optional[np.ndarray]:
        """Offset values as a float64 numpy array, or None if no offset column."""
        return self.data[self.offset_col].to_numpy().astype(float) if self.offset_col is not None else None

    # ── Variable management ───────────────────────────────────────────────────

    def add_variable(
        self,
        col: str,
        config: Optional[VariableConfig] = None,
        input_cols: Optional[List[str]] = None,
        custom_transform: Optional[Callable] = None,
        **kwargs,
    ) -> "ModelingTool":
        """
        Register preprocessing for a variable.

        Three calling styles:

        1. **Pass a VariableConfig directly**::

               tool.add_variable('x', config=VariableConfig(col='x', n_bins=10))

        2. **Keyword arguments** (most common)::

               tool.add_variable('vehicle_age', cap_upper=0.99, log_transform=True)
               tool.add_variable('state', encoding='onehot')
               tool.add_variable('driver_age', bin_edges=[16,25,35,50,65,100])

        3. **Multi-input derived variable** (new named variable from multiple columns)::

               tool.add_variable(
                   'age_x_veh',
                   input_cols=['driver_age', 'vehicle_age'],
                   custom_transform=lambda age, veh: age * veh,
                   cap_upper=0.99,
               )

        For categorical variables, ``custom_transform`` is applied before encoding::

               tool.add_variable(
                   'state',
                   custom_transform=lambda v: 'South' if v in ('TX', 'FL') else 'Other',
                   encoding='onehot',
               )

        If no arguments are provided, a default config is inferred from dtype.
        """
        if config is not None:
            self.variable_configs[col] = config
            return self

        # breakpoints is a user-friendly alias for bin_edges
        if "breakpoints" in kwargs:
            kwargs["bin_edges"] = kwargs.pop("breakpoints")

        if input_cols is not None:
            kwargs["input_cols"] = input_cols
        if custom_transform is not None:
            kwargs["custom_transform"] = custom_transform

        if kwargs:
            self.variable_configs[col] = VariableConfig(col=col, **kwargs)
        else:
            # Auto-detect: col must be in data for dtype detection
            if col in self.data.columns:
                self.variable_configs[col] = default_config(col, self.data[col])
            else:
                self.variable_configs[col] = VariableConfig(col=col)
        return self

    def get_variable_config(self, col: str) -> Optional[VariableConfig]:
        """Return the registered :class:`VariableConfig` for *col*."""
        return self.variable_configs.get(col)

    def list_variables(self) -> pl.DataFrame:
        """Summary table of all registered variable configs."""
        rows = []
        for col, cfg in self.variable_configs.items():
            rows.append({
                "col": col,
                "input_cols": str(cfg.input_cols) if cfg.input_cols else None,
                "is_categorical": cfg.is_categorical,
                "cap_lower": cfg.cap_lower,
                "cap_upper": cfg.cap_upper,
                "log_transform": cfg.log_transform,
                "n_bins": cfg.n_bins,
                "bin_edges": str(cfg.bin_edges) if cfg.bin_edges else None,
                "standardize": cfg.standardize,
                "encoding": cfg.encoding,
                "impute_strategy": cfg.impute_strategy,
                "custom_transform": cfg.custom_transform is not None,
            })
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    # ── Exploration ───────────────────────────────────────────────────────────

    def univariate_plot(
        self,
        col: str,
        n_bins: int = 10,
        breaks: Optional[List[float]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        version: Optional[str] = None,
    ) -> plt.Figure:
        """
        Weighted mean target vs each level (or quantile bin) of *col*.

        For continuous variables, bins into ``n_bins`` quantile groups.
        The sentinel value ``-999999999`` is labelled ``'Missing'``.

        Parameters
        ----------
        col : str
            Column to analyse.  Does not need to be in ``variable_configs``.
        version : str, optional
            When supplied, the fitted preprocessor from this version is used
            to resolve bin labels for *col* (consistent with
            ``relativities_table``).
        """
        preprocessor = None
        if version is not None:
            mv = self._get_version(version)
            preprocessor = getattr(mv, "preprocessor", None)
        elif col in self.variable_configs:
            preprocessor = Preprocessor([self.variable_configs[col]])
            preprocessor.fit(self.data, weights=self._weights_array)
        
        fig = univariate_plot(
            self.data, self._y, col=col,
            weights=self._weights, n_bins=n_bins, breaks=breaks, figsize=figsize,
            preprocessor=preprocessor,
        )

        return fig

    # ── Bin suggestion ────────────────────────────────────────────────────────

    def suggest_bins_quantile(
        self,
        col: str,
        n_bins: int = 10,
        verbose: bool = True,
        **kwargs,
    ) -> List[float]:
        """Equal-weight quantile breakpoints for *col*. Shortcut for ``suggest_bins(methods=['quantile'])[...]``."""
        from .bin_suggestor import suggest_bins_quantile as _fn
        return _fn(col, self.data, n_bins=n_bins, weights=self._weights,
                   verbose=verbose, **kwargs)

    def suggest_bins_equal_width(
        self,
        col: str,
        n_bins: int = 10,
        verbose: bool = True,
        **kwargs,
    ) -> List[float]:
        """Equal-width breakpoints for *col*. Shortcut for ``suggest_bins(methods=['equal_width'])[...]``."""
        from .bin_suggestor import suggest_bins_equal_width as _fn
        return _fn(col, self.data, n_bins=n_bins, verbose=verbose, **kwargs)

    def suggest_bins_gbm(
        self,
        col: str,
        max_splits: int = 20,
        verbose: bool = True,
        **kwargs,
    ) -> List[float]:
        """GBM-derived breakpoints for *col*. Shortcut for ``suggest_bins(methods=['gbm'])[...]``."""
        from .bin_suggestor import suggest_bins_gbm as _fn
        return _fn(col, self.data, self._y, weights=self._weights,
                   max_splits=max_splits, verbose=verbose, **kwargs)
    
    def suggest_bins_optbin(
        self,
        col: str,
        max_n_bins: int = 10,
        monotonic_trend: str = "auto",
        verbose: bool = True,
        **kwargs,
    ) -> List[float]:
        """Optimal binning breakpoints for *col*. Shortcut for ``suggest_bins(methods=['optbin'])[...]``."""
        from .bin_suggestor import suggest_bins_optbin as _fn
        return _fn(col, self.data, self._y, weights=self._weights,
                   max_n_bins=max_n_bins, monotonic_trend=monotonic_trend,
                   verbose=verbose, **kwargs)

    def suggest_bins(
        self,
        col: str,
        methods: Sequence[str] = ("quantile", "equal_width", "optbin", "gbm"),
        n_bins: int = 10,
        max_splits: int = 20,
        show_plot: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        **method_kwargs: Any,
    ) -> Dict[str, List[float]]:
        """
        Run multiple bin-suggestion strategies for a continuous variable.

        Prints each method's splits, then shows a weighted histogram with all
        split points overlaid as colour-coded vertical lines so results can be
        compared visually before committing to any breakpoints.

        Parameters
        ----------
        col : str
            Continuous variable to analyse.  Does not need to be in the model.
        methods : sequence of str
            Any subset of ``"quantile"``, ``"equal_width"``, ``"optbin"``,
            ``"gbm"``.  Defaults to running all four.
        n_bins : int
            Target bin count for ``"quantile"`` and ``"equal_width"``.
        max_splits : int
            Maximum thresholds returned by the ``"gbm"`` method (selected by
            frequency of use across all trees).
        show_plot : bool
            Display the distribution chart after all methods run.
        method_kwargs
            Forward kwargs to individual methods via ``quantile_kwargs``,
            ``equal_width_kwargs``, ``optbin_kwargs``, or ``gbm_kwargs``
            as dicts.

        Returns
        -------
        dict[str, list[float]]
            Method name → sorted list of suggested split points.

        Examples
        --------
        >>> splits = tool.suggest_bins('driver_age')
        >>> splits = tool.suggest_bins(
        ...     'vehicle_value',
        ...     methods=['optbin', 'gbm'],
        ...     optbin_kwargs={'max_n_bins': 6, 'monotonic_trend': 'auto'},
        ...     gbm_kwargs={'n_estimators': 200, 'learning_rate': 0.05},
        ... )
        >>> # Apply the optbin result directly
        >>> tool.add_variable('vehicle_value', breakpoints=splits['optbin'])
        """
        from .bin_suggestor import suggest_bins as _suggest_bins

        return _suggest_bins(
            col=col,
            X=self.data,
            y=self._y,
            weights=self._weights,
            methods=methods,
            n_bins=n_bins,
            max_splits=max_splits,
            figsize=figsize,
            **method_kwargs,
        )

    # ── Model fitting ─────────────────────────────────────────────────────────

    def fit_model(
        self,
        variables: List[str],
        version: str,
        alpha: Optional[float] = None,
        l1_ratio: Union[float, List[float]] = 0.5,
        use_cv: bool = True,
        cv: Optional[int] = None,
        family: Any = None,
        link: Optional[str] = None,
        tweedie_power: Optional[float] = 1.50,
        preprocessor: Optional[Preprocessor] = None,
        max_iter: int = 1000,
        gradient_tol: Optional[float] = None,
        fit_intercept: bool = True,
        print_summary: bool = True
    ) -> ModelVersion:
        """
        Fit an elastic net GLM and store it as a named version.

        Variables without a registered :class:`VariableConfig` are given
        sensible defaults based on their dtype.

        Parameters
        ----------
        variables : list of str
            Predictor column names.  Multi-input derived variable names can
            appear here if their config has been registered via
            :meth:`add_variable`.
        version : str
            Version label (e.g. ``'v1'``, ``'with_geo'``).
        alpha : float, optional
            Fixed regularisation strength.  Ignored when ``use_cv=True``.
            Pass ``0.0`` for an unpenalised GLM.
        l1_ratio : float or list of float
            Elastic-net mixing (0=ridge, 1=lasso).  A list triggers a CV
            grid search.
        use_cv : bool
            Select best alpha (and l1_ratio if a list) via CV.
        cv : int, optional
            Number of k-fold splits for hyperparameter selection.  When
            ``None`` (default) and a ``cv_column`` was supplied at
            construction, a :class:`~sklearn.model_selection.PredefinedSplit`
            is built from that column automatically.  When ``None`` and no
            ``cv_column`` exists, falls back to 5-fold CV.  Pass an explicit
            integer to override the ``cv_column`` for this specific fit.
        """
        if alpha is not None: 
            use_cv = False # No regularization, so no need for CV; ignore any cv argument passed
            resolved_cv = None
        elif cv is not None:
            resolved_cv: Any = cv
        elif self.cv_column is not None:
            from sklearn.model_selection import PredefinedSplit

            fold_values = self.data[self.cv_column]
            # Map arbitrary fold labels to contiguous integers (required by PredefinedSplit)
            unique_folds = sorted(set(fold_values))
            fold_map = {f: i for i, f in enumerate(unique_folds)}
            test_fold = [fold_map[f] for f in fold_values]
            resolved_cv = PredefinedSplit(test_fold)
        else:
            resolved_cv = 5  # sklearn default


        mv = fit_model(
            X=self.data,
            y=self._y_array,
            variables=variables,
            version_name=version,
            configs=self.variable_configs,
            weights=self._weights_array,
            offset=self._offset_array,
            family=family,
            link=link or self.link,
            tweedie_power=tweedie_power,
            preprocessor=preprocessor,
            alpha=alpha,
            l1_ratio=l1_ratio,
            use_cv=use_cv,
            cv=resolved_cv,
            max_iter=max_iter,
            gradient_tol=gradient_tol,
            fit_intercept=fit_intercept,
            drop_reference=self.drop_reference,
        )
        self.model_versions[version] = mv
        self.current_version = version

        if print_summary:
            self.model_summary(version)
        return None

    def fit_cv_stability(
        self,
        version: Optional[str] = None,
        family: Any = None,
        link: Optional[str] = None,
        tweedie_power: Optional[float] = None,
        plot: bool = True,
        show: bool = True,
    ) -> pl.DataFrame:
        """
        Assess coefficient stability using user-defined CV folds.

        For each unique value in ``self.cv_column``, the model is trained on all
        other rows and the coefficients are stored.  The geometric mean,
        standard deviation, and coefficient of variation (%) across folds
        are appended as summary rows.

        Parameters
        ----------
        version : str, optional
            Borrow ``alpha`` and ``l1_ratio`` from a previously fitted version.
        plot : bool
            Show coefficient stability box-plot.

        Returns
        -------
        pl.DataFrame
            Rows = one per fold + ``'geomean'``, ``'std'``, ``'cv_pct'``.
            Columns = ``'fold'`` + intercept + one per feature.
        """
        if version is not None and version in self.model_versions:
            mv = self.model_versions[version]
        elif version is None:
            print("Using current version since none was specified or specified version does not exist.")
            mv = self.model_versions[self.current_version]
        else:
            raise AttributeError('No model has yet been fit, please fit a model first and then specify a version.')

        resolved_alpha = mv.alpha
        resolved_l1 = mv.l1_ratio

        stability = fit_cv_stability(
            X=self.data,
            y=self._y_array,
            variables=mv.variables,
            configs=self.variable_configs,
            fold_col=self.cv_column,
            weights=self._weights_array,
            offset=self._offset_array,
            family=family or mv.family,
            link=link or self.link,
            tweedie_power=tweedie_power or mv.tweedie_power,
            alpha=resolved_alpha if resolved_alpha is not None else 0.01,
            l1_ratio=resolved_l1 if resolved_l1 is not None else 0.5,
            drop_reference=self.drop_reference,
            gradient_tol=mv.gradient_tol if hasattr(mv, "gradient_tol") else None
        )

        if plot:
            fig = cv_stability_plot(stability)
            if show:
                plt.show()

        self.model_versions[mv.name].cv_stability = stability

        return stability

    def set_base_version(self, version: str) -> "ModelingTool":
        if version not in self.model_versions and version not in self.data.columns:
            raise ValueError(f"Version '{version}' not found and not in data columns. Available versions: {list(self.model_versions.keys())}")
        
        self.base_version = version
        return self

    def predict(self, data: pl.DataFrame, version: Optional[str] = None, missing_factor: float = 1.0, offset: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate predictions for *data* using the specified model version."""
        mv = self._get_version(version or self.current_version)
        if offset is None:
            offset = self.__offset_array

        if isinstance(mv, FactorModelVersion):
            if mv.offset_col and mv.offset_col in data.columns and mv.offset_col != self.offset_col:
                offset = data[mv.offset_col].to_numpy().astype(float)
            return mv.predict(data, missing_factor=missing_factor, offset=offset)
        else:
            return mv.predict(data, offset=offset)
    # ── Excel factor version ──────────────────────────────────────────────────

    def add_excel_version(
        self,
        filepath: str,
        sheet_name: str,
        version: str = "excel",
        missing_factor: float = 1.0,
        base_version: Optional[str] = None,
        offset_col: Optional[str] = None
    ) -> "ModelingTool":
        """
        Load factors from an Excel sheet and register them as a new model version.

        The sheet must have columns **Variable**, **Level**, **Factor**.
        Level strings for variables covered by a fitted preprocessor must match
        :meth:`relativities_table` output (e.g. ``'TX'``, ``'[16, 25) (base)'``,
        ``'Missing'``).  For all other variables the raw column value is used as
        the level (direct string match).

        An optional row with ``Variable='intercept'`` and ``Level='intercept'``
        applies a global multiplicative factor to every prediction.

        Parameters
        ----------
        filepath : str
            Path to the ``.xlsx`` workbook.
        sheet_name : str
            Sheet name containing the Variable / Level / Factor table.
        version : str
            Version label to register (default ``'excel'``).
        missing_factor : float
            Factor applied to rows whose level is absent from the table
            (default 1.0, with a printed warning).
        base_version : str, optional
            Name of an existing fitted version whose preprocessor is used for
            numeric/binned level resolution.  When ``None``, the preprocessor
            that covers the most Excel variables is chosen automatically.
            Variables not covered by any preprocessor fall back to direct
            string lookup.

        Returns
        -------
        self  (fluent API — supports method chaining)
        """
        try:
            import openpyxl  # noqa: F401  — existence check only
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to read Excel files.\n"
                "Install it with:  pip install openpyxl"
            ) from exc

        factor_table = pl.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")

        missing_cols = {"Variable", "Level", "Factor"} - set(factor_table.columns)
        if missing_cols:
            raise ValueError(
                f"Excel sheet '{sheet_name}' is missing required columns: "
                f"{sorted(missing_cols)}.  Found: {factor_table.columns}"
            )

        factor_table = factor_table.with_columns(
            pl.col("Variable").cast(pl.String),
            pl.col("Level").cast(pl.String),
            pl.col("Factor").cast(pl.Float64),
        )

        variables = [
            v for v in factor_table["Variable"].unique()
            if v != "intercept"
        ]

        if base_version is not None:
            mv_base = self._get_version(base_version)
            prep = getattr(mv_base, "preprocessor", None)
        else:
            # Pick the fitted GLM preprocessor that covers the most excel variables
            prep = _build_preprocessor(variables, self.data, self.variable_configs)
            prep.fit(self.data, self._y_array, weights=self._weights_array)

        preprocessor_vars = (
            [v for v in variables if v in prep.configs]
            if prep is not None else []
        )

        # Variables not covered by preprocessor → direct lookup; validate present in data
        direct_vars = [v for v in variables if v not in preprocessor_vars]
        if len(self.data) > 0:
            missing_direct = [v for v in direct_vars if v not in self.data.columns]
            if missing_direct:
                raise ValueError(
                    f"Variables {missing_direct} are not covered by any preprocessor "
                    f"and are not found in the data columns. "
                    f"Ensure these columns exist or specify base_version."
                )

        fmv = FactorModelVersion(
            name=version,
            variables=variables,
            factor_table=factor_table,
            preprocessor=prep,
            preprocessor_vars=preprocessor_vars,
            train_predictions=np.array([]),
            offset_col=offset_col
        )

        if len(self.data) > 0:
            offset_array = None
            if fmv.offset_col and fmv.offset_col in self.data.columns:
                offset_array = self.data[fmv.offset_col].to_numpy().astype(float)
            # elif self.offset_col is not None:
            #     offset_array = self._offset_array
            fmv.train_predictions = fmv.predict(self.data, missing_factor=missing_factor, offset=offset_array)
        else:
            fmv.train_predictions = np.array([])

        self.model_versions[version] = fmv

        n_prep = len(preprocessor_vars)
        n_direct = len(direct_vars)
        print(
            f"  [Excel] Version '{version}' registered — "
            f"{len(variables)} variable(s): "
            f"{n_prep} preprocessor-resolved, {n_direct} direct-lookup."
        )
        return self

    # ── Model summary ─────────────────────────────────────────────────────────

    def model_summary(self, version: Optional[str] = None) -> pl.DataFrame:
        """Print and return the coefficient table for *version*."""
        mv = self._get_version(version or self.current_version)
        if isinstance(mv, FactorModelVersion):
            raise TypeError(
                f"Version '{version}' is an Excel factor model; "
                "model_summary is not applicable."
            )
        nonzero = int((mv.coefficients["coefficient"] != 0).sum()) - 1  # exclude intercept

        top_data = [
            ("Model Version:", mv.name),
            ("Dep. Variable:", self.target_col),
            ("Model:", f"GLM, Penalty Weight = {np.round(mv.alpha, 3)}, L1 Ratio = {np.round(mv.l1_ratio, 3)}"),
            ("Model Family:", mv.family.__class__.__name__),
            ("Link Function:", mv.link.__class__.__name__),
            ("Method:", "IRLS-CD"),
            ("Fit Date:", mv.fit_info['Fit_Time']),
            ("No. Iterations:", len(mv.glm.diagnostics_)),
            ("No. Observations:", len(self.data)),
            ('# Features:', len(mv.feature_names)),
            ('# Nonzero Coefs:', nonzero)
        ]
        top_part = tabulate(top_data, tablefmt="plain")

        coef_table = tabulate(mv.coefficients.to_dict())

        print('='*80)
        print(top_part)
        print('='*80)
        print(coef_table)

        return None

    def coefficient_plot(
        self,
        version: Optional[str] = None,
        top_n: int = 30,
        figsize: Optional[Tuple[int, int]] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Horizontal bar chart of coefficients for *version*."""
        mv = self._get_version(version or self.current_version)
        if isinstance(mv, FactorModelVersion):
            raise TypeError(
                f"Version '{version}' is an Excel factor model; "
                "coefficient_plot is not applicable."
            )
        fig = coefficient_plot(mv.coefficients, version_name=version,
                               top_n=top_n, figsize=figsize)
        if show:
            plt.show()
        return fig

    # ── Private helpers for relativities_table ───────────────────────────────

    def _get_fold_info(
        self,
        mv: "ModelVersion"
    ) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        """Refit on each fold; return (fold_names, fold_label -> {feat: coef})."""
        if mv.cv_stability is None:
            stability = fit_cv_stability(
                X=self.data,
                y=self._y_array,
                variables=mv.variables,
                configs=self.variable_configs,
                fold_col=self.cv_column,
                weights=self._weights_array,
                offset=self._offset_array,
                family=mv.family,
                link=mv.link,
                alpha=mv.alpha,
                l1_ratio=mv.l1_ratio,
                drop_reference=self.drop_reference,
            )
            mv.cv_stability = stability
        else:
            stability = mv.cv_stability
        fold_rows = stability.filter(
            ~pl.col("fold").is_in(["geomean", "std", "cv_pct"])
        )
        fold_names = fold_rows["fold"].to_list()
        fold_coef_map: Dict[str, Dict[str, float]] = {
            fn: {k: v for k, v in row_d.items() if k != "fold"}
            for fn, row_d in zip(fold_names, fold_rows.to_dicts())
        }
        return fold_names, fold_coef_map

    def _get_calib_arrays(
        self,
        prep: Any,
        calib_df: Optional[pl.DataFrame],
    ) -> Tuple[Optional[pl.DataFrame], Optional[np.ndarray], float]:
        """Transform calib_df; return (Xt_calib, w_calib, total_calib_w)."""
        if calib_df is None:
            return None, None, 0.0
        Xt_calib = prep.transform(calib_df)
        w_calib = (
            calib_df[self.weight_col].to_numpy().astype(float)
            if self.weight_col and self.weight_col in calib_df.columns
            else np.ones(len(calib_df))
        )
        return Xt_calib, w_calib, float(w_calib.sum())

    def _cat_var_rows(
        self,
        var_col: str,
        p: Dict[str, Any],
        Xt_df: pl.DataFrame,
        w_arr: np.ndarray,
        total_w: float,
        coef_map: Dict[str, float],
        fold_names: List[str],
        fold_coef_map: Dict[str, Dict[str, float]],
        Xt_calib: Optional[pl.DataFrame],
        w_calib: Optional[np.ndarray],
        total_calib_w: float,
    ) -> List[Dict[str, Any]]:
        """Row dicts for one one-hot categorical variable."""
        categories = p["categories"]
        dropped = p.get("dropped_category")
        feats = [f"{var_col}_{cat}" for cat in categories]

        train_fw = _weighted_feat_map(Xt_df, feats, w_arr)
        other_w = sum(train_fw.values())

        calib_fw = (
            _weighted_feat_map(Xt_calib, feats, w_calib)
            if Xt_calib is not None else {}
        )
        base_cw: Optional[float] = (
            total_calib_w - sum(calib_fw.values()) if Xt_calib is not None else None
        )

        rows: List[Dict[str, Any]] = []
        if dropped is not None:
            rows.append(_make_row(
                var_col, dropped, total_w - other_w, 0.0,
                fold_names, fold_coef_map, None, calib_weight=base_cw,
            ))
        for cat, feat in zip(categories, feats):
            rows.append(_make_row(
                var_col, str(cat),
                train_fw.get(feat, 0.0), coef_map.get(feat, 0.0),
                fold_names, fold_coef_map, feat,
                calib_weight=calib_fw.get(feat, 0.0) if Xt_calib is not None else None,
            ))
        return rows

    def _binned_var_rows(
        self,
        var_col: str,
        p: Dict[str, Any],
        cfg: Any,
        Xt_df: pl.DataFrame,
        w_arr: np.ndarray,
        total_w: float,
        coef_map: Dict[str, float],
        fold_names: List[str],
        fold_coef_map: Dict[str, Dict[str, float]],
        Xt_calib: Optional[pl.DataFrame],
        w_calib: Optional[np.ndarray],
        total_calib_w: float,
    ) -> List[Dict[str, Any]]:
        """Row dicts for one binned numeric variable."""
        edges = p["bin_edges"]
        dropped_bin = p.get("dropped_bin", 0)
        all_labels = p.get("bin_labels")

        missing_feat = f"{var_col}_missing"
        bin_feats = [
            f"{var_col}_{label}"
            for i, label in enumerate(all_labels)
            if i != dropped_bin
        ]
        all_feats = (
            ([missing_feat] if missing_feat in Xt_df.columns else []) + bin_feats
        )

        train_fw = _weighted_feat_map(Xt_df, all_feats, w_arr)
        other_w = sum(train_fw.values())

        calib_fw = (
            _weighted_feat_map(Xt_calib, all_feats, w_calib)
            if Xt_calib is not None else {}
        )
        base_cw: Optional[float] = (
            total_calib_w - sum(calib_fw.values()) if Xt_calib is not None else None
        )

        base_label = all_labels[dropped_bin]
        rows: List[Dict[str, Any]] = [
            _make_row(
                var_col, f"{base_label} (base)", total_w - other_w, 0.0,
                fold_names, fold_coef_map, None, calib_weight=base_cw,
            )
        ]
        if missing_feat in Xt_df.columns:
            rows.append(_make_row(
                var_col, "Missing",
                train_fw.get(missing_feat, 0.0), coef_map.get(missing_feat, 0.0),
                fold_names, fold_coef_map, missing_feat,
                calib_weight=calib_fw.get(missing_feat, 0.0) if Xt_calib is not None else None,
            ))
        for i, label in enumerate(all_labels):
            if i == dropped_bin:
                continue
            feat = f"{var_col}_{label}"
            rows.append(_make_row(
                var_col, label,
                train_fw.get(feat, 0.0), coef_map.get(feat, 0.0),
                fold_names, fold_coef_map, feat,
                calib_weight=calib_fw.get(feat, 0.0) if Xt_calib is not None else None,
            ))
        return rows

    def summary_table(
        self,
        version: Optional[str] = None,
        calib_df: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Summary table for all categorical and binned variables in *version*.

        Each row is one level of one discrete variable.  The dropped base
        level is included with a coefficient of zero so the full picture is
        visible at a glance.  Pure continuous variables are excluded.

        Parameters
        ----------
        version : str, optional
            Version key of the fitted model to inspect. Uses current version if none specified.
        calib_df : pl.DataFrame, optional
            An independent DataFrame (e.g. a calibration or holdout set).
            When supplied, a ``calib_weight`` column is added showing the
            total exposure weight from *calib_df* assigned to each level.
            The same ``weight_col`` used for training is read from this
            DataFrame; if absent, unit weights are assumed.

        Returns
        -------
        pl.DataFrame
            Columns: ``variable``, ``level``, ``weight``
            [, ``calib_weight``], ``train_coef`` [, ``fold_{k}`` …].
        """
        mv = self._get_version(version or self.current_version)
        if isinstance(mv, FactorModelVersion):
            raise TypeError(
                f"Version '{version}' is an Excel factor model; "
                "summary_table is not applicable (the factor table IS the relativity table)."
            )
        prep = mv.preprocessor

        w_arr = self._weights_array
        if w_arr is None:
            w_arr = np.ones(len(self.data))
        total_w = float(w_arr.sum())
        Xt_df = prep.transform(self.data)

        coef_map: Dict[str, float] = {
            r["feature"]: r["coefficient"]
            for r in mv.coefficients.to_dicts()
            if r["feature"] != "intercept"
        }

        fold_names, fold_coef_map = (
            self._get_fold_info(mv) if self.cv_column is not None else ([], {})
        )
        Xt_calib, w_calib, total_calib_w = self._get_calib_arrays(prep, calib_df)

        rows: List[Dict[str, Any]] = []
        for var_col in mv.variables:
            if var_col not in prep.configs:
                continue
            p = prep._params.get(var_col, {})
            cfg = prep.configs[var_col]

            if p.get("is_categorical") and p.get("encoding") == "onehot":
                rows.extend(self._cat_var_rows(
                    var_col, p, Xt_df, w_arr, total_w, coef_map,
                    fold_names, fold_coef_map, Xt_calib, w_calib, total_calib_w,
                ))
            elif "bin_edges" in p:
                rows.extend(self._binned_var_rows(
                    var_col, p, cfg, Xt_df, w_arr, total_w, coef_map,
                    fold_names, fold_coef_map, Xt_calib, w_calib, total_calib_w,
                ))
            # else: pure continuous variable — excluded

        return pl.DataFrame(rows) if rows else pl.DataFrame()

    # ── AvE data table ───────────────────────────────────────────────────────

    def _glm_factor_arrays(
        self, mv: ModelVersion,
    ) -> Dict[str, np.ndarray]:
        """Per-row factor array for each variable in a fitted GLM."""
        prep = mv.preprocessor
        Xt_df = prep.transform(self.data)
        coef_map: Dict[str, float] = {
            r["feature"]: r["coefficient"]
            for r in mv.coefficients.to_dicts()
            if r["feature"] != "intercept"
        }
        use_exp = getattr(mv, "link", "log") == "log"
        n = len(self.data)
        result: Dict[str, np.ndarray] = {}

        for var_col in mv.variables:
            if var_col not in prep.configs:
                continue
            p = prep._params.get(var_col, {})
            cfg = prep.configs[var_col]

            if p.get("is_categorical") and p.get("encoding") == "onehot":
                feats = [f"{var_col}_{cat}" for cat in p["categories"]]
            elif "bin_edges" in p:
                dropped_bin = p.get("dropped_bin", 0)
                all_labels = p.get("bin_labels")
                feats = []
                if f"{var_col}_missing" in Xt_df.columns:
                    feats.append(f"{var_col}_missing")
                feats += [
                    f"{var_col}_{lbl}"
                    for i, lbl in enumerate(all_labels)
                    if i != dropped_bin
                ]
            else:
                feats = [var_col] + [f"{var_col}^{d}" for d in range(2, cfg.degree + 1)]

            linear = np.zeros(n, dtype=float)
            for feat in feats:
                c = coef_map.get(feat, 0.0)
                if c != 0.0 and feat in Xt_df.columns:
                    linear += c * Xt_df[feat].to_numpy().astype(float)
            result[var_col] = np.exp(linear) if use_exp else linear

        return result

    def _factor_model_factor_arrays(
        self, mv: FactorModelVersion,
    ) -> Dict[str, np.ndarray]:
        """Per-row factor array for each variable in a factor-table model."""
        prep = mv.preprocessor
        Xt: Optional[pl.DataFrame] = None
        if prep is not None and mv.preprocessor_vars:
            Xt = prep.transform(self.data)

        n = len(self.data)
        factor_by_var = {
            grp: df.select(["Level", "Factor"])
            for (grp, ), df in mv.factor_table.group_by("Variable")
        }
        result: Dict[str, np.ndarray] = {}

        for V in mv.variables:
            ft_v = factor_by_var.get(V, pl.DataFrame({"Level": [], "Factor": []}))
            level_arr = self._resolve_factor_model_levels(V, mv, prep, Xt, n)

            tmp = pl.DataFrame({"Level": pl.Series("_l", level_arr)}).join(
                ft_v, on="Level", how="left"
            )
            result[V] = tmp["Factor"].fill_null(1.0).to_numpy().astype(float)

        return result

    def _resolve_factor_model_levels(
        self,
        V: str,
        mv: FactorModelVersion,
        prep: Optional[Any],
        Xt: Optional[pl.DataFrame],
        n: int,
    ) -> np.ndarray:
        """Resolve level strings for one variable, mirroring FactorModelVersion.predict."""
        if V in mv.preprocessor_vars and Xt is not None:
            p = prep._params[V]
            if p.get("is_categorical") and p.get("encoding") == "onehot":
                dropped = p.get("dropped_category", "")
                level_arr = np.full(n, f"{dropped} (base)", dtype=object)
                for cat in p["categories"]:
                    feat = f"{V}_{cat}"
                    if feat in Xt.columns:
                        level_arr[Xt[feat].to_numpy().astype(bool)] = str(cat)
                return level_arr
            if "bin_edges" in p:
                dropped_bin = p.get("dropped_bin", 0)
                all_labels = p.get("bin_labels")
                level_arr = np.full(n, f"{all_labels[dropped_bin]} (base)", dtype=object)
                missing_feat = f"{V}_missing"
                if missing_feat in Xt.columns:
                    level_arr[Xt[missing_feat].to_numpy().astype(bool)] = "Missing"
                for i, label in enumerate(all_labels):
                    if i == dropped_bin:
                        continue
                    feat = f"{V}_{label}"
                    if feat in Xt.columns:
                        level_arr[Xt[feat].to_numpy().astype(bool)] = label
                return level_arr

        # Direct string match on raw column
        return np.array(self.data[V].cast(pl.String).to_list(), dtype=object)

    def decile_lift_chart(
        self,
        version: Optional[str] = None,
        n_bins: int = 10
    ) -> plt.Figure:
        """
        Decile lift chart for *version*.

        The data is sorted by predicted value and split into *n_bins* equal-sized
        groups.  The mean actual and predicted values are plotted for each group
        to show how well the model discriminates between high- and low-risk
        segments.

        Parameters
        ----------
        version : str
            Model version name.
        n_bins : int
            Number of equal-sized groups to split the data into (default 10).

        Returns
        -------
        plt.Figure
            Line chart with one line each for actual and predicted.
        """
        mv = self._get_version(version or self.current_version)
        y = self._y_array
        w = self._weights_array
        p = mv.train_predictions

        fig = decile_lift_chart(y, p, w, n_bins, mv.name)

        return fig

    def ave_table(
        self,
        variables: List[str],
        version: Optional[str] = None,
        n_bins: int = 10,
    ) -> pl.DataFrame:
        """
        Actual-vs-Expected breakdown table for a list of analysis variables.

        For each analysis variable and each of its levels, returns the total
        weighted loss (actual), weighted prediction, exposure weight, and one
        column per model variable showing ``sum(factor_i * weight_i)``.

        The factor for a model variable at row *i* is:

        - **GLM (ModelVersion)**: ``exp(linear_contribution)`` for log-link,
          where the linear contribution is the sum of ``coef * feature_value``
          across all design-matrix features belonging to that variable
          (including polynomial terms for higher-degree continuous variables).
        - **Factor model (FactorModelVersion)**: the factor looked up directly
          from the factor table for the row's level.

        Parameters
        ----------
        variables : list of str
            Analysis variables to break down by.  Need not be model variables.
        version : str, optional
            Model version name. Uses current version if none specified.
        n_bins : int
            Quantile bins for continuous non-binned analysis variables.

        Returns
        -------
        pl.DataFrame
            Columns: ``variable``, ``level``, ``weight``, ``loss``,
            ``prediction``, then ``{model_var}_factor`` for each model
            variable.
        """
        mv = self._get_version(version or self.current_version)
        prep = getattr(mv, "preprocessor", None)
        w_arr = self._weights_array if self._weights_array is not None else np.ones(len(self.data))
        y_arr = self._y_array
        pred_arr = mv.train_predictions

        if isinstance(mv, FactorModelVersion):
            factor_arrays = self._factor_model_factor_arrays(mv)
        else:
            factor_arrays = self._glm_factor_arrays(mv)

        model_vars = [v for v in mv.variables if v in factor_arrays]
        factor_cols = [f"{mv_var}_factor" for mv_var in model_vars]
        col_order = ["variable", "level", "weight", "loss", "prediction"] + factor_cols

        # Precompute weighted arrays once, reused across all analysis variables
        yw = y_arr * w_arr
        pw = pred_arr * w_arr
        weighted_factors = {mv_var: factor_arrays[mv_var] * w_arr for mv_var in model_vars}

        agg_exprs = [
            pl.col("_w").sum().alias("weight"),
            pl.col("_yw").sum().alias("loss"),
            pl.col("_pw").sum().alias("prediction"),
        ] + [
            pl.col(f"_f_{mv_var}").sum().alias(f"{mv_var}_factor")
            for mv_var in model_vars
        ]

        all_parts: List[pl.DataFrame] = []
        for var in variables:
            level_series = _resolve_level(var, self.data, prep, n_bins)

            tmp_data: Dict[str, Any] = {
                "_level": level_series, "_w": w_arr, "_yw": yw, "_pw": pw,
            }
            for mv_var in model_vars:
                tmp_data[f"_f_{mv_var}"] = weighted_factors[mv_var]

            summary = pl.DataFrame(tmp_data).group_by("_level").agg(agg_exprs)

            labels = _sort_labels(summary["_level"].to_list())
            order_df = pl.DataFrame(
                {"_level": labels, "_order": list(range(len(labels)))}
            )
            summary = (
                summary.join(order_df, on="_level").sort("_order").drop("_order")
                .with_columns(pl.lit(var).alias("variable"))
                .rename({"_level": "level"})
                .select(col_order)
            )
            all_parts.append(summary)

        if not all_parts:
            return pl.DataFrame()
        return pl.concat(all_parts)

    # ── Actual vs Expected ────────────────────────────────────────────────────

    def ae_chart(
        self,
        col: str,
        version: Optional[str] = None,
        n_bins: int = 10,
        breaks: Optional[List[float]] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Actual vs Expected chart for *col* using model *version*.

        *col* does not need to be a model predictor.  Continuous variables
        are binned into ``n_bins`` quantile groups.  The sentinel value
        ``-999999999`` is labelled ``'Missing'``.
        """
        mv = self._get_version(version or self.current_version)
        fig = ae_chart(
            X=self.data,
            y=self._y,
            col=col,
            predictions=mv.train_predictions,
            weights=self._weights,
            n_bins=n_bins,
            breaks=breaks,
            figsize=figsize,
            version_name=version,
            preprocessor=getattr(mv, "preprocessor", None),
        )

        return fig

    def residual_chart(
        self,
        col: str,
        version: Optional[str] = None,
        n_bins: int = 10,
        breaks: Optional[List[float]] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Residual signal chart: ``mean_actual / mean_predicted`` per level of *col*.

        Where an A/E chart plots actual and predicted side-by-side, this chart
        shows their ratio directly.  A ratio of 1.1 / 1.05 ≈ 1.048 appears as
        a point at 1.048, making residual signal immediately readable as
        deviation from the 1.0 reference line.

        - Values **above 1.0** → model is *under-predicting* for that group.
        - Values **below 1.0** → model is *over-predicting* for that group.

        Exposure (weight) is shown as bars on the primary axis so that the
        credibility of each ratio is visible at a glance.

        Parameters
        ----------
        col : str
            Variable to slice by.  Does not need to be a model predictor.
        version : str, optional
            Version key of the model whose predictions are used. Uses current version if none specified.
        n_bins : int
            Number of quantile bins for continuous variables.
        """
        mv = self._get_version(version or self.current_version)
        fig = residual_chart(
            X=self.data,
            y=self._y,
            col=col,
            predictions=mv.train_predictions,
            weights=self._weights,
            n_bins=n_bins,
            breaks=breaks,
            figsize=figsize,
            version_name=version,
            preprocessor=getattr(mv, "preprocessor", None),
        )

        return fig

    def plot_all_variables(
        self,
        version: Optional[str] = None,
        chart: str = "residual",
        n_bins: int = 10,
        figsize: Optional[Tuple[int, int]] = None
    ) -> List[plt.Figure]:
        """
        Plot a residual or A/E chart for every variable in *version*.

        Parameters
        ----------
        version : str, optional
            Version key whose variable list drives the loop. Uses current version if none specified.
        chart : {'residual', 'ae'}
            ``'residual'`` (default) — ``mean_actual / mean_predicted`` per
            level, with a horizontal reference line at 1.0.
            ``'ae'`` — side-by-side actual vs expected bars.
        n_bins : int
            Number of quantile bins used for continuous variables.
        show : bool
            Call ``plt.show()`` after each chart.

        Returns
        -------
        list of matplotlib.figure.Figure
            One figure per variable, in the same order as ``mv.variables``.
        """
        if chart not in ("residual", "ae"):
            raise ValueError(f"chart must be 'residual' or 'ae', got {chart!r}")
        version = version or self.current_version
        mv = self._get_version(version)
        figs: List[plt.Figure] = []
        for col in mv.variables:
            if chart == "ae":
                fig = self.ae_chart(col, version=version, n_bins=n_bins,
                                    figsize=figsize)
            else:
                fig = self.residual_chart(col, version=version, n_bins=n_bins,
                                          figsize=figsize)
            figs.append(fig)
        return figs

    # ── Model comparison ──────────────────────────────────────────────────────

    def _get_dl_score(self, y, p1, p2, weights, n_buckets, deviation='absolute'):
        dl_data = double_lift_table(y, p1, p2, weights=weights, n_buckets=n_buckets)
        return dl_data, double_lift_score(dl_data, deviation=deviation)

    def compare_models(
        self,
        version1: Optional[str] = None,
        version2: Optional[str] = None,
        n_buckets: int = 10,
        figsize: Optional[Tuple[int, int]] = None,
        dl_deviation: Optional[str] = 'absolute',
        show: bool = True
    ) -> Dict[str, Any]:
        """
        Compare two model versions: metrics table and double-lift chart.

        ``version2`` can be either:

        * A fitted model version name registered with :meth:`fit_model` or
          :meth:`add_excel_version` (the original behaviour), **or**
        * A **column name** in the tool's dataset whose values are pre-computed
          predictions (e.g. an incumbent / external model stored in the data).
        * A preset version using the set_base_version method, which also registers the version name.

        Registered model versions take priority: if a name matches both a
        version and a column, the version is used.

        Parameters
        ----------
        version1 : str, optional
            First model version (must be a registered version name). Uses base version if none specified.
        version2 : str, optional
            Second model — either a registered version name or a dataframe
            column containing predictions.
        n_buckets : int
            Number of buckets for the double-lift table.

        Returns
        -------
        dict
            ``{'metrics': pl.DataFrame, 'double_lift': pl.DataFrame}``
        """
        mv1 = self._get_version(version1 or self.current_version)

        y = self._y_array
        w = self._weights_array
        p1 = mv1.train_predictions

        # Resolve version2: registered model version takes priority over column.
        if version2 is None and self.base_version is None:
            raise ValueError("version2 must be specified if no base version is set.")
        version2 = version2 or self.base_version

        if version2 in self.model_versions:
            p2 = self._get_version(version2).train_predictions
        elif version2 in self.data.columns:
            col_s = self.data[version2]
            if not col_s.dtype.is_numeric():
                raise ValueError(
                    f"Column '{version2}' has dtype {col_s.dtype}; "
                    "predictions must be numeric."
                )
            p2 = col_s.cast(pl.Float64).to_numpy()
        else:
            # Delegate to _get_version to raise the standard helpful KeyError.
            self._get_version(version2)
            p2 = np.array([])  # unreachable; satisfies type checkers

        dl_data, dl_sc = self._get_dl_score(y, p1, p2, weights=w, n_buckets=n_buckets, deviation=dl_deviation)

        metrics = compare_metrics(
            y, p1, p2,
            weights=w,
            name1=version1,
            name2=version2,
            dl_score=dl_sc,
        )

        print("\n" + "=" * 60)
        print(f"  Comparison: {version1}  vs  {version2}")
        print("=" * 60)
        print(metrics)
        # print("=" * 60)
        # if dl_sc < 0:
        #     dl_interp = f"negative -> {version1} wins"
        # elif dl_sc > 0:
        #     dl_interp = f"positive -> {version2} wins"
        # else:
        #     dl_interp = "tie"
        # print(f"  double_lift_score interpretation: {dl_interp}")
        # print("=" * 60 + "\n")

        double_lift_chart(y, p1, p2, weights=w, n_buckets=n_buckets,
                          name1=version1, name2=version2, figsize=figsize)

        if show:
            plt.show()

        return {"metrics": metrics, "double_lift": dl_data}

    def list_versions(self) -> pl.DataFrame:
        """Summary table of all stored model versions."""
        rows = []
        y = self._y_array
        w = self._weights_array
        for name, mv in self.model_versions.items():
            m = compute_metrics(y, mv.train_predictions, w, name)
            metric_vals = {r: v for r, v in zip(m["metric"].to_list(), m[name].to_list())}
            rows.append({
                "version": name,
                "n_variables": len(mv.variables),
                "alpha": mv.alpha,
                "l1_ratio": mv.l1_ratio,
                "n_nonzero": max(0, int((mv.coefficients["coefficient"] != 0).sum()) - 1),
                "rmse": metric_vals.get("rmse", float("nan")),
                "mae": metric_vals.get("mae", float("nan")),
                "gini_norm": metric_vals.get("gini_norm", float("nan")),
            })
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, version: Optional[str] = None, filepath: Optional[str] = 'models/model.pkl') -> None:
        """
        Save a model version to *filepath* (pickle).

        Parameters
        ----------
        version : str
            Version key to save.
        filepath : str
            Destination path (e.g. ``'models/v1.pkl'``).
        """
        mv = self._get_version(version or self.current_version)
        save_version(mv, self, filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        data: pl.DataFrame,
        target_col: Optional[str] = None,
        weight_col: Optional[str] = None,
    ) -> "ModelingTool":
        """
        Load a saved version, refit it on *data*, and return a new tool.

        The saved version's variable configs and hyperparameters are restored,
        then the model is refit from scratch.  The result is registered as
        version ``'v1'``.

        Parameters
        ----------
        data : pl.DataFrame
            Training data to refit on (must contain the same columns).
        target_col, weight_col : str, optional
            Override saved column names.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError("data must be a polars DataFrame.")

        bundle = load_version(filepath, data=data, refit=True)
        snap = bundle["snapshot"]
        vs = snap["version"]
        ts = snap["tool_settings"]

        tool = cls(
            data=data,
            target_col=target_col or ts["target_col"],
            weight_col=weight_col or ts["weight_col"],
            offset_col=ts.get("offset_col", None),
            link=ts["link"],
            drop_reference=ts.get("drop_reference", "max_weight"),
        )
        for col, cfg in ts["variable_configs"].items():
            tool.variable_configs[col] = cfg

        tool.fit_model(
            variables=vs["variables"],
            version="v1",
            alpha=vs["alpha"],
            l1_ratio=vs["l1_ratio"],
            use_cv=False,
            family=vs["family"],
            tweedie_power=vs["tweedie_power"],
            link=vs["link"],
            print_summary=True,
        )
        print(f"Loaded '{vs['name']}' from {filepath!r}, refitted as version 'v1'.")
        return tool

    @classmethod
    def load_frozen(cls, filepath: str, data: pl.DataFrame) -> "ModelingTool":
        """
        Restore a saved version without refitting (prediction-only mode).

        The returned tool has no data but its ``model_versions['v1']``
        can call ``.predict(X)`` directly.
        """
        bundle = load_version(filepath, data=None, refit=False)
        snap = bundle["snapshot"]
        vs = snap["version"]
        ts = snap["tool_settings"]

        tool = cls.__new__(cls)
        tool.data = data.clone()
        tool.target_col = ts["target_col"]
        tool.weight_col = ts["weight_col"]
        tool.offset_col = ts.get("offset_col", None)
        tool.family = vs["family"]
        tool.link = ts["link"]
        tool.variable_configs = ts["variable_configs"]
        tool.model_versions = {}

        from .model import ModelVersion as MV
        mv = MV(
            name="v1",
            variables=vs["variables"],
            preprocessor=vs["preprocessor"],
            glm=vs["glm"],
            feature_names=vs["feature_names"],
            coefficients=vs["coefficients"],
            alpha=vs["alpha"],
            l1_ratio=vs["l1_ratio"],
            family=vs["family"],
            link=vs["link"],
            train_predictions=None,
            fit_info=vs["fit_info"],
            tweedie_power=vs["tweedie_power"],
        )

        offset_arr = None
        if tool.offset_col is not None and tool.offset_col in data.columns:
            offset_arr = data[tool.offset_col].cast(pl.Float64).to_numpy()

        mv.train_predictions = mv.predict(data, offset=offset_arr)
        tool.model_versions["v1"] = mv
        print(f"Loaded frozen '{vs['name']}' from {filepath!r} as 'v1'.")
        return tool

    @classmethod
    def load_from_excel(
        cls,
        excel_path: str,
        sheet_name: str,
        data: pl.DataFrame,
        target_col: str,
        weight_col: Optional[str] = None,
        pkl_path: Optional[str] = None,
        version: str = "excel",
        missing_factor: float = 1.0,
        offset_col: Optional[str] = None
    ) -> "ModelingTool":
        """
        Build a :class:`ModelingTool` from an Excel factor table.

        If *pkl_path* is supplied the saved model is loaded frozen (providing
        preprocessing context for numeric/binned variables).  Without a pkl the
        tool works in standalone mode — only categorical or pre-banded string
        columns are resolved directly.

        Parameters
        ----------
        excel_path : str
            Path to the ``.xlsx`` workbook.
        sheet_name : str
            Sheet containing ``Variable``, ``Level``, ``Factor`` columns.
        data : pl.DataFrame
            Dataset to score.
        target_col : str
            Name of the target column (required to construct the tool).
        weight_col : str, optional
            Name of the exposure-weight column.
        pkl_path : str, optional
            Path to a saved ``ModelingTool`` pickle (from :meth:`save`).
            When provided, the frozen model's preprocessors are used for
            level resolution of binned/categorical variables.
        version : str
            Version label for the Excel model (default ``'excel'``).
        missing_factor : float
            Factor applied to unseen levels (default 1.0, with a warning).

        Returns
        -------
        ModelingTool
            Contains *data* and a single registered version *version*.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError("data must be a polars DataFrame.")

        if pkl_path is not None:
            # Load the saved model frozen to get preprocessor + variable configs
            frozen = cls.load_frozen(pkl_path, data)
            tool = cls(
                data=data,
                target_col=target_col or frozen.target_col,
                weight_col=weight_col or frozen.weight_col
            )
            tool.variable_configs = frozen.variable_configs
            tool.model_versions = frozen.model_versions  # 'v1' has the preprocessor
        else:
            tool = cls(data=data, target_col=target_col, weight_col=weight_col)

        tool.add_excel_version(
            excel_path,
            sheet_name,
            version=version,
            missing_factor=missing_factor,
            offset_col=offset_col
        )
        return tool

    # ── Discovery ─────────────────────────────────────────────────────────────

    def fit_shadow_gbm(
        self,
        feature_cols: Optional[List[str]] = None,
        tweedie_power: Optional[float] = 1.50,
        **kwargs: Any,
    ) -> Any:
        """
        Fit a LightGBM on raw features for diagnostic purposes.

        Stores the fitted model on ``self._shadow_model``.  See
        :func:`~elastic_net_tool.discovery.fit_shadow_gbm` for parameters.
        """
        from .discovery import fit_shadow_gbm

        model = fit_shadow_gbm(
            self.data,
            self.target_col,
            weight_col=self.weight_col,
            offset_col=self.offset_col,
            feature_cols=feature_cols,
            tweedie_power=tweedie_power,
            variable_configs=self.variable_configs,
            **kwargs,
        )
        self._shadow_model = model
        print(f"  Shadow GBM fitted on {len(model._shadow_feature_cols)} features.")
        return model

    def interaction_ranking(self, top_n: int = 20, **kwargs: Any) -> pl.DataFrame:
        """Rank variable pairs by H-statistic.  Requires :meth:`fit_shadow_gbm` first."""
        from .discovery import interaction_ranking

        if not hasattr(self, "_shadow_model"):
            raise RuntimeError("Call fit_shadow_gbm() first.")
        return interaction_ranking(
            self._shadow_model, self.data,
            top_n=top_n, **kwargs,
        )

    def partial_dependence_2d(self, var1: str, var2: str, **kwargs: Any) -> pl.DataFrame:
        """2D partial dependence for a variable pair.  Requires :meth:`fit_shadow_gbm` first."""
        from .discovery import partial_dependence_2d

        if not hasattr(self, "_shadow_model"):
            raise RuntimeError("Call fit_shadow_gbm() first.")
        return partial_dependence_2d(self._shadow_model, self.data, var1, var2, **kwargs)

    def permutation_importance(
        self,
        version: Optional[str] = None,
        metric_fn: Optional[Any] = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Permutation importance.

        If *version* is given, uses the fitted GLM; otherwise uses
        the shadow GBM (must call :meth:`fit_shadow_gbm` first).
        """
        from .discovery import permutation_importance as _perm_imp

        if version is not None:
            raise NotImplementedError(
                "Permutation importance on fitted GLMs requires the full "
                "transform pipeline.  Use with shadow GBM instead, or pass "
                "a version=None to use the shadow model."
            )
        else:
            if not hasattr(self, "_shadow_model"):
                raise RuntimeError("Call fit_shadow_gbm() first.")
            return _perm_imp(
                self._shadow_model, self.data, self.target_col,
                weight_col=self.weight_col, metric_fn=metric_fn, **kwargs,
            )

    def shap_importance(
        self,
        feature_cols: Optional[List[str]] = None,
        sample_size: int = 500,
        random_state: int = 42,
    ) -> pl.DataFrame:
        """
        SHAP-based feature importance using TreeExplainer.

        Requires :meth:`fit_shadow_gbm` first and ``pip install shap``.

        Returns
        -------
        pl.DataFrame
            Columns: ``variable``, ``importance_mean``, ``importance_std``.
        """
        from .discovery import shap_importance as _shap_importance

        if not hasattr(self, "_shadow_model"):
            raise RuntimeError("Call fit_shadow_gbm() first.")
        return _shap_importance(
            self._shadow_model, self.data,
            feature_cols=feature_cols,
            sample_size=sample_size,
            random_state=random_state,
        )

    def shap_dependence(
        self,
        var: str,
        color_var: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        sample_size: int = 500,
        random_state: int = 42,
    ) -> pl.DataFrame:
        """
        SHAP dependence data for *var* — reveals transform shape and breakpoints.

        Requires :meth:`fit_shadow_gbm` first and ``pip install shap``.

        Returns
        -------
        pl.DataFrame
            Columns: ``{var}``, ``shap_value`` [, ``{color_var}``].
        """
        from .discovery import shap_dependence as _shap_dependence

        if not hasattr(self, "_shadow_model"):
            raise RuntimeError("Call fit_shadow_gbm() first.")
        return _shap_dependence(
            self._shadow_model, self.data, var,
            color_var=color_var,
            feature_cols=feature_cols,
            sample_size=sample_size,
            random_state=random_state,
        )

    def shap_interaction_ranking(
        self,
        feature_cols: Optional[List[str]] = None,
        sample_size: int = 200,
        random_state: int = 42,
        top_n: int = 20,
    ) -> pl.DataFrame:
        """
        Rank variable pairs by SHAP interaction strength.

        Faster and more accurate than Friedman H-statistic for tree models.
        Requires :meth:`fit_shadow_gbm` first and ``pip install shap``.

        Returns
        -------
        pl.DataFrame
            Columns: ``var1``, ``var2``, ``interaction_strength``.
        """
        from .discovery import shap_interaction_ranking as _shap_ir

        if not hasattr(self, "_shadow_model"):
            raise RuntimeError("Call fit_shadow_gbm() first.")
        return _shap_ir(
            self._shadow_model, self.data,
            feature_cols=feature_cols,
            sample_size=sample_size,
            random_state=random_state,
            top_n=top_n,
        )

    def tree_interaction_cooccurrence(self, top_n: int = 20) -> pl.DataFrame:
        """
        Fast interaction ranking by tree co-occurrence weighted by split gain.

        Use as a cheap pre-screen before running SHAP interaction ranking.
        Requires :meth:`fit_shadow_gbm` first.

        Returns
        -------
        pl.DataFrame
            Columns: ``var1``, ``var2``, ``cooccurrence_score``.
        """
        from .discovery import tree_interaction_cooccurrence as _tic

        if not hasattr(self, "_shadow_model"):
            raise RuntimeError("Call fit_shadow_gbm() first.")
        return _tic(self._shadow_model, top_n=top_n)

    def suggest_category_groups(
        self,
        col: str,
        max_groups: int = 10,
        min_exposure_pct: float = 0.01,
        verbose: bool = True,
    ):
        """
        Suggest groupings for a high-cardinality categorical variable.

        Levels are sorted by exposure-weighted mean target and merged greedily
        until at most ``max_groups`` groups remain.

        Returns
        -------
        tuple[dict[str, str], pl.DataFrame]
            ``(level_to_group, summary)`` — mapping dict and a summary table
            with columns ``group``, ``levels``, ``exposure``, ``mean_target``.
        """
        from .discovery import suggest_category_groups as _scg

        return _scg(
            col, self.data, self._y,
            weights=self._weights,
            max_groups=max_groups,
            min_exposure_pct=min_exposure_pct,
            verbose=verbose,
        )

    def monotonicity_test(
        self,
        var: str,
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 100,
        random_state: int = 42,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Measure the RMSE cost of enforcing a monotone constraint on *var*.

        Fits constrained GBMs (increasing and decreasing) and reports how much
        accuracy is lost versus an unconstrained baseline.  A small cost
        (< ~1 %) means the monotone constraint is safe to apply.

        Returns
        -------
        dict
            Keys: ``unconstrained_rmse``, ``constrained_rmse_pos``,
            ``constrained_rmse_neg``, ``cost_pos``, ``cost_neg``,
            ``recommended``.
        """
        from .discovery import monotonicity_test as _mt

        return _mt(
            self.data, self.target_col, var,
            weight_col=self.weight_col,
            feature_cols=feature_cols,
            n_estimators=n_estimators,
            random_state=random_state,
            variable_configs=self.variable_configs,
            verbose=verbose,
            **kwargs,
        )

    def boruta_select(
        self,
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 100,
        n_iterations: int = 20,
        threshold: float = 0.05,
        random_state: int = 42,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Boruta-style feature selection using shadow (shuffled) features.

        Each real feature must beat the maximum shadow-feature importance in
        at least ``1 - threshold`` of iterations to be selected.

        Returns
        -------
        pl.DataFrame
            Columns: ``variable``, ``pass_rate``, ``selected``.
            Sorted by ``pass_rate`` descending.
        """
        from .discovery import boruta_select as _boruta

        return _boruta(
            self.data, self.target_col,
            weight_col=self.weight_col,
            feature_cols=feature_cols,
            n_estimators=n_estimators,
            n_iterations=n_iterations,
            threshold=threshold,
            random_state=random_state,
            variable_configs=self.variable_configs,
            **kwargs,
        )

    def residual_gbm(
        self,
        version: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        top_n: int = 10,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Fit a GBM on GLM residuals to find missing signal.

        Parameters
        ----------
        version : str
            Model version whose residuals to analyse.
        feature_cols : list of str, optional
            Raw feature columns.  Defaults to all numeric columns.
        """
        from .discovery import residual_gbm as _residual_gbm

        mv = self._get_version(version or self.current_version)
        actual = self._y_array
        predicted = mv.train_predictions
        safe_pred = np.where(np.abs(predicted) < 1e-12, 1e-12, predicted)
        residuals = actual / safe_pred

        if feature_cols is None:
            exclude = {self.target_col}
            if self.weight_col:
                exclude.add(self.weight_col)
            feature_cols = [
                c for c in self.data.columns
                if c not in exclude
                and self.data[c].dtype.is_numeric()
            ]

        return _residual_gbm(
            self.data, residuals, feature_cols,
            weight_col=self.weight_col, offset_col=self.offset_col, top_n=top_n, variable_configs=self.variable_configs, **kwargs,
        )

    # ── Enhanced residual analysis ───────────────────────────────────────────

    def residual_heatmap(
        self,
        col1: str,
        col2: str,
        version: Optional[str] = None,
        n_bins: int = 8,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, pl.DataFrame]:
        """
        2D residual heatmap: A/E ratio across two variable dimensions.

        See :func:`~elastic_net_tool.plots.residual_heatmap`.
        """
        from .plots import residual_heatmap as _residual_heatmap

        mv = self._get_version(version or self.current_version)
        fig, data = _residual_heatmap(
            self.data, self._y, col1, col2,
            predictions=mv.train_predictions,
            weights=self._weights,
            preprocessor=mv.preprocessor,
            n_bins=n_bins,
            **kwargs,
        )
        
        return fig, data

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def regularization_path(
        self,
        variables: Optional[List[str]] = None,
        version: Optional[str] = None,
        tweedie_power: Optional[float] = 1.50,
        l1_ratio: float = 0.5,
        n_alphas: int = 50,
        alpha_min: float = 1e-5,
        alpha_max: float = 10.0,
        show: bool = True,
    ) -> pl.DataFrame:
        """
        Fit the GLM at a sequence of alpha values and track coefficient evolution.

        Parameters
        ----------
        variables : list of str, optional
            Variables to use. Defaults to the variables in *version*.
        version : str, optional
            Existing version to derive variable list from.
        """
        from .plots import regularization_path_plot

        mv = None
        if variables is None:
            if version is not None:
                mv = self._get_version(version)
                variables = mv.variables
            else:
                raise ValueError("Provide variables or version.")

        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)[::-1]
        tweedie_power = mv.tweedie_power if mv is not None else tweedie_power
        family = mv.family if mv is not None else 'tweedie'
        prep = mv.preprocessor if mv is not None else None

        rows = []
        for alpha_val in alphas:
            mv = fit_model(
                X=self.data,
                y=self._y,
                variables=variables,
                version_name="_regpath",
                configs=self.variable_configs,
                preprocessor=prep,
                weights=self._weights_array,
                offset=self._offset_array,
                family=family,
                link=self.link,
                tweedie_power=tweedie_power,
                alpha=float(alpha_val),
                l1_ratio=l1_ratio,
                use_cv=False,
                drop_reference=self.drop_reference,
            )
            coefs = mv.coefficients
            for feat_row in coefs.iter_rows(named=True):
                if feat_row["feature"] != "intercept":
                    rows.append({
                        "alpha": float(alpha_val),
                        "variable": feat_row["feature"],
                        "coefficient": feat_row["coefficient"],
                    })

        path_df = pl.DataFrame(rows)

        if show:
            fig = regularization_path_plot(path_df)
            plt.show()

        return path_df

    def _apply_cv_fold(self, mask: pl.Series):
        '''
        Shortcut for filtering the data, target, and weights to the train/test fold in cross-validation methods.
        '''
        x, y = self.data.filter(mask), self._y_array[mask]
        w = self._weights_array[mask] if self._weights_array is not None else None
        o = self._offset_array[mask] if self._offset_array is not None else None
        return x, y, w, o

    def _get_cv_metrics_gini(self, train_mask, test_mask, model_version):
        '''
        Fit model on the given train fold and return the gini coefficient of the test fold.  Used for cross-validation metrics in gini mode.
        '''
        X_train, y_train, w_train, o_train = self._apply_cv_fold(train_mask)
        X_test, y_test, w_test, o_test = self._apply_cv_fold(test_mask)

        fold_mv = fit_model(X=X_train, y=y_train, variables=model_version.variables,
            version_name=f"_cv_fold", weights=w_train, offset=o_train, family=model_version.family, link=self.link, preprocessor=model_version.preprocessor,
            alpha=model_version.alpha, l1_ratio=model_version.l1_ratio, use_cv=False, drop_reference=self.drop_reference
        )

        pred_test = fold_mv.predict(X_test, offset=o_test)

        return gini_coefficient(y_test, pred_test, w_test), w_test.sum()
    
    def _get_cv_metrics_dls(self, train_mask, test_mask, dl_base_preds, dl_refit_base, dl_base_version, model_version, dl_deviation='absolute'):
        '''
        Fit model on the given train fold and return the double lift score of the test fold.  Used for cross-validation metrics in double lift mode.
        '''
        X_train, y_train, w_train, o_train = self._apply_cv_fold(train_mask)
        X_test, y_test, w_test, o_test = self._apply_cv_fold(test_mask)

        fold_mv = fit_model(X=X_train, y=y_train, variables=model_version.variables,
            version_name=f"_cv_fold", weights=w_train, offset=o_train, family=model_version.family, link=self.link, preprocessor=model_version.preprocessor,
            alpha=model_version.alpha,l1_ratio=model_version.l1_ratio, use_cv=False, drop_reference=self.drop_reference
        )
        pred_test = fold_mv.predict(X_test, offset=o_test)

        if dl_refit_base:
            bv = self._get_version(dl_base_version)
            base_mv = fit_model(X=X_train, y=y_train, variables=bv.variables,
                version_name=f"_cv_base_fold", weights=w_train, offset=o_train, family=bv.family, link=self.link, preprocessor=bv.preprocessor,
                alpha=bv.alpha, l1_ratio=bv.l1_ratio, use_cv=False, drop_reference=self.drop_reference
            )
            base_preds_test = base_mv.predict(X_test, offset=o_test)
        else:
            base_preds_test = dl_base_preds[test_mask]

        _, fold_dl_score = self._get_dl_score(y_test, base_preds_test, pred_test, weights=w_test, n_buckets=10, deviation=dl_deviation)

        return fold_dl_score, w_test.sum()

    def overfitting_monitor(
        self,
        version_names: List[str],
        metric_fn: Optional[Any] = 'double_lift_score',
        dl_base_version: Optional[str] = None,
        dl_deviation: str = 'absolute',
        show: bool = True,
    ) -> pl.DataFrame:
        """
        Track train vs CV metric across existing model versions.

        Uses the stored train predictions and CV scores from each version.

        metric_fn should be one of `gini` or `double_lift_score`. If `double_lift_score` is chosen, a model version or column name containing predictions must be provided
        """
        from .plots import overfitting_plot

        assert metric_fn in ('gini', 'double_lift_score', None), "metric_fn must be 'gini' or 'double_lift_score'"

        dl_refit_base = False
        if metric_fn == 'double_lift_score':
            assert dl_base_version is not None or self.base_version is not None, "dl_base_version must be provided when metric_fn is 'double_lift_score' and no base version is set."
            dl_base_version = dl_base_version or self.base_version

            if dl_base_version in self.model_versions:
                dl_base_preds = None
                dl_refit_base = True
                version_names = [x for x in version_names if x != dl_base_version]
            elif dl_base_version in self.data.columns:
                col_s = self.data[dl_base_version]
                if not col_s.dtype.is_numeric():
                    raise ValueError(
                        f"Column '{dl_base_version}' has dtype {col_s.dtype}; "
                        "predictions must be numeric."
                    )
                dl_base_preds = col_s.cast(pl.Float64).to_numpy()
            else:
                # Delegate to _get_version to raise the standard helpful KeyError.
                self._get_version(dl_base_version)
                dl_base_preds = np.array([])  # unreachable; satisfies type checkers

        rows = []
        cumulative_vars: List[str] = []
        y_true = self._y_array
        w = self._weights_array

        for i, vname in enumerate(version_names):
            mv = self._get_version(vname)

            if metric_fn == 'gini':
                train_metric = gini_coefficient(y_true, mv.train_predictions, w)
            else:
                _, train_metric = self._get_dl_score(y_true, dl_base_preds, mv.train_predictions, weights=w, n_buckets=10, deviation=dl_deviation)

            if self.cv_column is not None:
                fold_arr = self.data[self.cv_column].to_numpy()
                unique_folds = np.unique(fold_arr)
                fold_metrics, fold_weights = [], []

                for fold in unique_folds:
                    train_mask = pl.Series(fold_arr != fold)
                    test_mask = fold_arr == fold
                    if metric_fn == 'gini':
                        fold_metric, fold_weight = self._get_cv_metrics_gini(train_mask, test_mask, mv)
                    else:
                        fold_metric, fold_weight = self._get_cv_metrics_dls(train_mask, test_mask, dl_base_preds, dl_refit_base, dl_base_version, mv, dl_deviation)
                    fold_metrics.append(fold_metric)
                    fold_weights.append(float(fold_weight))
                cv_metric = float(np.average(fold_metrics, weights=fold_weights))
            else:
                cv_metric = train_metric

            new_vars = [v for v in mv.variables if v not in cumulative_vars]
            cumulative_vars.extend(new_vars)

            rows.append({
                "step": i + 1,
                "n_variables": len(mv.variables),
                "variables_added": vname,
                "train_metric": train_metric,
                "cv_metric": cv_metric,
                "gap": train_metric - cv_metric,
            })

        monitor_df = pl.DataFrame(rows)
        if show:
            fig = overfitting_plot(monitor_df)
            plt.show()
        return monitor_df

    # ── Statistical ──────────────────────────────────────────────────────────

    def vif_table(self, version: Optional[str] = None) -> pl.DataFrame:
        """
        Compute VIF for each feature in a fitted model's design matrix.

        Parameters
        ----------
        version : str
            Model version to analyse.
        """
        from .metrics import vif_table as _vif_table

        mv = self._get_version(version or self.current_version)
        # Reconstruct design matrix
        preprocessor = mv.preprocessor
        Xt = preprocessor.transform(self.data)
        feature_cols = [c for c in Xt.columns if c in mv.feature_names]
        design = Xt.select(feature_cols)
        return _vif_table(design)

    def bootstrap_metrics(
        self,
        version: Optional[str] = None,
        metric_fns: Optional[Any] = None,
        n_bootstrap: int = 500,
        ci: float = 0.95,
        show: bool = True,
    ) -> pl.DataFrame:
        """
        Bootstrap confidence intervals on model performance metrics.

        See :func:`~elastic_net_tool.metrics.bootstrap_metrics`.
        """
        from .metrics import bootstrap_metrics as _bootstrap_metrics
        from .plots import bootstrap_ci_plot

        mv = self._get_version(version or self.current_version)
        result = _bootstrap_metrics(
            self._y_array,
            mv.train_predictions,
            weights=self._weights_array,
            metric_fns=metric_fns,
            n_bootstrap=n_bootstrap,
            ci=ci,
        )
        if show:
            fig = bootstrap_ci_plot(result, title=f"Bootstrap CIs — {version or self.current_version}")
            plt.show()
        return result

    def bootstrap_relativities(
        self,
        version: Optional[str] = None,
        n_bootstrap: int = 200,
        ci: float = 0.95,
        random_state: int = 42,
        show: bool = False,
    ) -> pl.DataFrame:
        """
        Bootstrap CIs on each factor relativity by resampling and refitting.

        Parameters
        ----------
        version : str
            Model version to bootstrap.
        n_bootstrap : int
            Number of bootstrap resamples.
        ci : float
            Confidence level.

        Returns
        -------
        pl.DataFrame
            Columns: ``variable``, ``level``, ``relativity``,
            ``ci_lower``, ``ci_upper``, ``std_error``.
        """
        mv = self._get_version(version or self.current_version)
        variables = mv.variables
        n = len(self.data)
        rng = np.random.RandomState(random_state)
        alpha = (1 - ci) / 2

        # Get baseline relativities
        base_rel = self.summary_table(version or self.current_version)
        # Collect (variable, level) pairs and their bootstrap coefficient samples
        base_coefs = {
            (r["variable"], r["level"]): r["train_coef"]
            for r in base_rel.iter_rows(named=True)
        }
        keys = list(base_coefs.keys())

        boot_alpha = mv.alpha
        boot_l1_ratio = mv.l1_ratio
        boot_coefs: Dict[Tuple[str, str], List[float]] = {k: [] for k in keys}

        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            boot_data = self.data[idx]

            try:
                boot_mv = fit_model(
                    X=boot_data,
                    y=boot_data[self.target_col],
                    variables=variables,
                    version_name="_bootstrap",
                    configs=self.variable_configs,
                    weights=boot_data[self.weight_col].to_numpy().astype(float) if self.weight_col else None,
                    offset=boot_data[self.offset_col].to_numpy().astype(float) if self.offset_col else None,
                    family=mv.family,
                    link=self.link,
                    tweedie_power=mv.tweedie_power,
                    alpha=boot_alpha,
                    l1_ratio=boot_l1_ratio,
                    use_cv=False,
                    drop_reference=self.drop_reference,
                )
                boot_coef_table = boot_mv.coefficients
                coef_dict = {
                    r["feature"]: r["coefficient"]
                    for r in boot_coef_table.iter_rows(named=True)
                }
                # Map back to (variable, level) keys
                for key in keys:
                    var, level = key
                    feat = f"{var}_{level}" if level != "(base)" else None
                    if feat and feat in coef_dict:
                        boot_coefs[key].append(coef_dict[feat])
                    else:
                        boot_coefs[key].append(base_coefs[key])
            except Exception:
                # If a bootstrap sample fails to fit, skip it
                for key in keys:
                    boot_coefs[key].append(base_coefs[key])

        rows = []
        for key in keys:
            var, level = key
            coef = base_coefs[key]
            samples = np.array(boot_coefs[key])
            rel = np.exp(coef)
            boot_rels = np.exp(samples)
            rows.append({
                "variable": var,
                "level": level,
                "relativity": float(rel),
                "ci_lower": float(np.quantile(boot_rels, alpha)),
                "ci_upper": float(np.quantile(boot_rels, 1 - alpha)),
                "std_error": float(np.std(boot_rels)),
            })

        result = pl.DataFrame(rows)

        if show:
            from .plots import relativities_ci_plot
            for var in result["variable"].unique().to_list():
                fig = relativities_ci_plot(result, var)
                plt.show()

        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_version(self, version: str) -> ModelVersion:
        if version not in self.model_versions:
            available = list(self.model_versions.keys())
            raise KeyError(
                f"Version '{version}' not found.  Available: {available}"
            )
        return self.model_versions[version]
