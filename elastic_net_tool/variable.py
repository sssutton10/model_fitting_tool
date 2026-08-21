"""Variable configuration and robust, fitted preprocessing (polars backend)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union

import numpy as np
import polars as pl

# ── Constants ─────────────────────────────────────────────────────────────────

MISSING_SENTINEL: float = -999_999_999.0
"""Sentinel value used to indicate 'missing' in continuous variables during binning."""

_CAT_MISSING: str = "__MISSING__"
"""Placeholder used to represent missing values in categorical columns before imputation."""

# Polars numeric dtypes
_NUMERIC_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
)

_IMPUTE_STRATEGIES = frozenset({"median", "mean", "most_frequent", "constant"})


def _is_str_or_cat(dtype: pl.PolarsDataType) -> bool:
    """Return True for string / categorical polars dtypes."""
    return dtype not in _NUMERIC_DTYPES and dtype not in (
        pl.Boolean,
        pl.Date,
        pl.Datetime,
        pl.Duration,
        pl.Time,
        pl.Null,
    )


def _is_sentinel(arr: np.ndarray) -> np.ndarray:
    """Return a boolean mask where *arr* equals MISSING_SENTINEL (within floating-point tolerance)."""
    return np.isclose(arr, MISSING_SENTINEL, rtol=0, atol=1.0)


# ── Configuration dataclass ───────────────────────────────────────────────────


@dataclass
class VariableConfig:
    """
    Preprocessing configuration for one model variable (or derived variable).

    Single-column usage
    -------------------
    Set ``col`` to the source column name.  All transforms operate on that
    column.

    Multi-column derived variable
    -----------------------------
    Set ``col`` to the *output* name and ``input_cols`` to the list of source
    columns. ``custom_transform`` is then called as::

        custom_transform(df: pl.DataFrame, **kwargs) -> array-like

    ``df`` contains the columns in ``input_cols``. Inputs may themselves be
    derived variables; they are resolved recursively, and their raw transform
    results are supplied before their own preprocessing options are applied.

    Parameters
    ----------
    col : str
        Output variable name (also the source column when ``input_cols`` is
        ``None``).
    input_cols : list of str, optional
        Source columns for multi-input transforms.  When ``None``, ``col``
        itself is the only input.
    cap_lower : float, optional
        Lower cap
    cap_upper : float, optional
        Upper cap
    log_transform : bool
        Apply log1p after capping.
    impute_strategy : str, optional
        ``'median'``, ``'mean'``, ``'most_frequent'``, ``'constant'``, or
        ``None`` (leave nulls in place).
    impute_value : scalar, optional
        Fill value for ``impute_strategy='constant'``.
    n_bins : int, optional
        Number of *quantile-based* bins for a continuous variable.  Ignored
        when ``bin_edges`` is supplied.
    bin_edges : list of float, optional
        Explicit breakpoints for binning (e.g. ``[0, 2, 5, 10, 20]``).
        Takes precedence over ``n_bins``.  The column is one-hot encoded
        after binning.  Any value equal to :data:`MISSING_SENTINEL` gets its
        own ``{col}_missing`` dummy column.
    standardize : bool
        Standardise the variable after caps / log.  Ignored when binning.
    degree : int
        Polynomial degree for continuous (unbinned) variables.  ``1`` (default)
        means no expansion.  ``2`` adds a squared term (``col^2``), ``3`` adds
        cubic (``col^3``), etc.  Ignored when ``bin_edges`` / ``n_bins`` is set
        or for categorical variables.  Standardisation is applied to the base
        value first; higher-degree terms are powers of the standardised value.
    encoding : {'auto', 'onehot', None}
        Encoding for categorical variables.  ``'auto'`` detects from dtype.
    is_categorical : bool, optional
        Force categorical treatment.  ``None`` auto-detects from dtype.
    right_closed: bool
        When binning, whether bins include the right edge (``(lo, hi]``) or left
    custom_transform : callable, optional
        May be a **named function** or a lambda.  Any callable is accepted.

        Called once as ``f(df: pl.DataFrame, **kw) -> array-like``. For a
        single-column transform, ``df`` contains ``col``. When ``input_cols``
        is set, it contains those columns in configured order. The transform
        runs before capping, logging, binning, or categorical encoding.
    transform_kwargs : dict, optional
        Keyword arguments forwarded to ``custom_transform`` on every call.
        Useful for passing parameters to a named function without a closure::

            def scale(df, factor=1.0):
                return df["mileage"] / factor

            VariableConfig('mileage', custom_transform=scale,
                           transform_kwargs={'factor': 1000})
    """

    col: str
    input_cols: list[str] | None = None
    cap_lower: float | None = None
    cap_upper: float | None = None
    log_transform: bool = False
    impute_strategy: str | None = None
    impute_value: Any | None = None
    n_bins: int | None = None
    bin_edges: list[float] | None = None
    standardize: bool = False
    degree: int = 1
    encoding: str | None = "auto"
    is_categorical: bool | None = None
    right_closed: bool = False
    custom_transform: Callable[..., Any] | None = None
    transform_kwargs: dict[str, Any] = None

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_transform_options()
        self._validate_numeric_options()
        self._normalize_bin_edges()

    def _validate_identity(self) -> None:
        """Validate variable and dependency names."""
        if not isinstance(self.col, str) or not self.col.strip():
            raise ValueError("VariableConfig.col must be a non-empty string.")
        if self.input_cols is not None:
            invalid = not self.input_cols or any(
                not isinstance(column, str) or not column for column in self.input_cols
            )
            if invalid:
                raise ValueError(
                    f"Variable '{self.col}' input_cols must be non-empty strings."
                )
            if len(set(self.input_cols)) != len(self.input_cols):
                raise ValueError(
                    f"Variable '{self.col}' input_cols must not contain duplicates."
                )

    def _validate_transform_options(self) -> None:
        """Validate custom transform, imputation, and encoding options."""
        if self.custom_transform is not None and not callable(self.custom_transform):
            raise TypeError(f"Variable '{self.col}' custom_transform must be callable.")
        if self.transform_kwargs is not None and not isinstance(
            self.transform_kwargs, dict
        ):
            raise TypeError(
                f"Variable '{self.col}' transform_kwargs must be a dict or None."
            )
        if self.impute_strategy not in _IMPUTE_STRATEGIES | {None}:
            raise ValueError(
                f"Variable '{self.col}' has unsupported impute_strategy "
                f"{self.impute_strategy!r}."
            )
        if self.impute_strategy == "constant" and self.impute_value is None:
            raise ValueError(
                f"Variable '{self.col}' requires impute_value for constant imputation."
            )
        if self.encoding not in {"auto", "onehot", None}:
            raise ValueError(
                f"Variable '{self.col}' encoding must be 'auto', 'onehot', or None."
            )

    def _validate_numeric_options(self) -> None:
        """Validate degree, binning, and cap settings."""
        invalid_degree = (
            not isinstance(self.degree, int)
            or isinstance(self.degree, bool)
            or self.degree < 1
        )
        if invalid_degree:
            raise ValueError(
                f"Variable '{self.col}' degree must be a positive integer."
            )
        invalid_bins = self.n_bins is not None and (
            not isinstance(self.n_bins, int)
            or isinstance(self.n_bins, bool)
            or self.n_bins < 2
        )
        if invalid_bins:
            raise ValueError(f"Variable '{self.col}' n_bins must be an integer >= 2.")
        if self.n_bins is not None and self.bin_edges is not None:
            raise ValueError(
                f"Variable '{self.col}' cannot set both n_bins and bin_edges."
            )
        for name, value in (
            ("cap_lower", self.cap_lower),
            ("cap_upper", self.cap_upper),
        ):
            invalid = value is not None and (
                not np.isscalar(value) or not np.isfinite(float(value))
            )
            if invalid:
                raise ValueError(
                    f"Variable '{self.col}' {name} must be finite when supplied."
                )
        invalid_caps = (
            self.cap_lower is not None
            and self.cap_upper is not None
            and self.cap_lower > self.cap_upper
        )
        if invalid_caps:
            raise ValueError(
                f"Variable '{self.col}' cap_lower cannot exceed cap_upper."
            )

    def _normalize_bin_edges(self) -> None:
        """Validate and normalize explicit bin edges."""
        if self.bin_edges is None:
            return
        edges = np.asarray(self.bin_edges, dtype=float)
        invalid = (
            edges.ndim != 1
            or len(edges) == 0
            or not np.all(np.isfinite(edges))
            or not np.all(np.diff(edges) > 0)
        )
        if invalid:
            raise ValueError(
                f"Variable '{self.col}' bin_edges must be a non-empty, finite, "
                "strictly increasing sequence."
            )
        self.bin_edges = edges.tolist()


@dataclass(frozen=True)
class FittedVariableParams:
    is_categorical: bool
    impute_val: float | str | None


@dataclass(frozen=True)
class FittedCategoricalParams(FittedVariableParams):
    encoding: Literal["onehot"] | None
    categories: tuple[str, ...] = ()
    dropped_category: str | None = None


@dataclass(frozen=True)
class FittedNumericParams(FittedVariableParams):
    cap_lower_val: float | None = None
    cap_upper_val: float | None = None
    right_closed: bool = False
    std_mean: float | None = None
    std_std: float | None = None


@dataclass(frozen=True)
class FittedBinnedNumericParams(FittedNumericParams):
    bin_edges: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bin_labels: tuple[str, ...] = ()
    dropped_bin: int = 0
    has_sentinel_bin: bool = False


FittedParams = Union[
    FittedCategoricalParams, FittedNumericParams, FittedBinnedNumericParams
]


def default_config(col: str, s: pl.Series) -> VariableConfig:
    """Return a sensible default :class:`VariableConfig` based on dtype."""
    if _is_str_or_cat(s.dtype):
        return VariableConfig(
            col=col,
            log_transform=False,
            impute_strategy="most_frequent",
            encoding="auto",
            is_categorical=True,
        )
    return VariableConfig(
        col=col,
        is_categorical=False,
    )


# ── Binning helpers ───────────────────────────────────────────────────────────


def compute_quantile_bin_edges(
    arr: np.ndarray, n_bins: int, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute quantile-based bin edges from *arr*, excluding sentinel/NaN values.
    Returns a monotone array of at least 2 unique values.
    """
    valid_mask = (arr != MISSING_SENTINEL) & np.isfinite(arr)
    valid = arr[valid_mask]
    if len(valid) == 0:
        return np.array([-np.inf, np.inf])

    percentiles = np.linspace(0, 100, n_bins + 1)
    if weights is not None:
        w_valid = np.asarray(weights, dtype=float)[valid_mask]
        edges = np.unique(
            np.percentile(valid, percentiles, weights=w_valid, method="inverted_cdf")
        )
    else:
        edges = np.unique(np.percentile(valid, percentiles))

    if len(edges) < 2:
        mn, mx = float(valid.min()), float(valid.max())
        edges = np.array([mn - 1e-9, mx + 1e-9])
    return edges


# def _fmt_edge(v: float) -> str:
#     """Format a bin edge: no decimal point for whole numbers, else ≤3 decimal places."""
#     if v == int(v):
#         return str(int(v))
#     return f"{v:.3f}".rstrip("0").rstrip(".")


# def _bin_letter(i: int) -> str:
#     """0→'A', 1→'B', …, 25→'Z', 26→'AA', 27→'AB', …"""
#     letters = string.ascii_uppercase
#     if i < 26:
#         return letters[i]
#     return letters[i // 26 - 1] + letters[i % 26]


# def make_bin_labels(breaks: np.ndarray, right: bool = False, min_val: float | None = None, max_val: float | None = None) -> list[str]:
#     """
#     Return human-readable label strings for the n+1 bins defined by *n* break points.

#     Labels:
#       bin 0 (all below first break): ``'{letter}_<hi'``
#       bins 1 … n-1 (interior):       ``'{letter}_[lo, hi)'`` or ``'{letter}_(lo, hi]'`` if right-closed
#       bin n (all above last break):   ``'{letter}_lo+'``
#     """
#     n = len(breaks)
#     if n == 0:
#         return ["A_all"]
#     labels: list[str] = []
#     for i in range(n + 1):
#         letter = _bin_letter(i)
#         if i == 0:
#             hi = _fmt_edge(float(breaks[0]))
#             if min_val is not None and abs(min_val - breaks[0]) <= 1e-5:
#                 labels.append(f"{letter}_{hi}")
#             else:
#                 symb = '<' if not right else '≤'
#                 labels.append(f"{letter}_{symb}{hi}")
#         elif i == n:
#             lo = _fmt_edge(float(breaks[-1]))
#             if max_val is not None and abs(max_val - breaks[i - 1]) <= 1e-5:
#                 labels.append(f"{letter}_{lo}")
#             else:
#                 labels.append(f"{letter}_{lo}+")
#         else:
#             lo = _fmt_edge(float(breaks[i - 1]))
#             l_sym = "[" if not right else '('
#             hi = _fmt_edge(float(breaks[i]))
#             h_sym = ")" if not right else ']'
#             labels.append(f"{letter}_{l_sym}{lo}, {hi}{h_sym}")
#     return labels


def _fmt(x, dec=3):
    if x == int(x):
        return f"{int(x):,}"
    s = f"{round(x, dec):,.{dec}f}"
    integer_part, _, decimal_part = s.partition(".")
    decimal_stripped = decimal_part.rstrip("0")
    return f"{integer_part}.{decimal_stripped}" if decimal_stripped else integer_part


def make_bin_labels(breaks, right=False, dec=3):
    """
    Generate labeled interval strings from internal break points,
    producing open-ended first and last buckets.

    Parameters
    ----------
    breaks : array-like of float
        Internal break points only (sorted).
    right : bool, default False
        Whether intervals are right-closed.
    dec : int
        Decimal places for formatting.

    Returns
    -------
    list[str]
    """

    breaks = np.asarray(breaks, dtype=float)

    out = []
    ascii_val = 65  # 'A'
    n = len(breaks)

    # ---- FIRST bucket ----
    label = chr(ascii_val)
    sym = "<=" if right else "<"
    out.append(f"{label}_{sym}{_fmt(breaks[0], dec)}")
    ascii_val += 1

    # ---- MIDDLE buckets (only if n > 1) ----
    for i in range(1, n):
        label = chr(ascii_val)

        left_bracket = "(" if right else "["
        right_bracket = "]" if right else ")"

        # duplicate break adjustment
        if i > 1 and breaks[i - 2] == breaks[i - 1]:
            left_bracket = "("
        if (i + 1) < n and breaks[i + 1] == breaks[i]:
            right_bracket = ")"

        if breaks[i - 1] == breaks[i]:
            out.append(f"{label}_{_fmt(breaks[i], dec)}")
        else:
            out.append(
                f"{label}_"
                f"{left_bracket}{_fmt(breaks[i - 1], dec)}, {_fmt(breaks[i], dec)}"
                f"{right_bracket}"
            )
        ascii_val += 1

    # ---- LAST bucket ----
    label = chr(ascii_val)
    out.append(f"{label}_{_fmt(breaks[-1], dec)}+")

    return out


# ── Preprocessor ─────────────────────────────────────────────────────────────


class Preprocessor:
    """
    Fits variable transformations on training data and applies them to any
    polars DataFrame with matching columns.

    Parameters
    ----------
    configs : list of VariableConfig
        Configurations for emitted variables and any derived-variable
        dependencies needed to construct them.
    output_cols : sequence of str, optional
        Variables whose fitted features should be emitted. Dependency-only
        configs remain available for raw-column construction but are not
        fitted or included in the design matrix. Defaults to every config.
    """

    def __init__(
        self,
        configs: list[VariableConfig],
        output_cols: Sequence[str] | None = None,
    ):
        if not configs:
            raise ValueError("Preprocessor requires at least one VariableConfig.")
        if any(not isinstance(c, VariableConfig) for c in configs):
            raise TypeError("configs must contain VariableConfig instances.")
        if len({c.col for c in configs}) != len(configs):
            raise ValueError("Preprocessor configs must have unique output names.")
        self.configs = {c.col: c for c in configs}
        self.output_cols = (
            list(self.configs) if output_cols is None else list(output_cols)
        )
        if (
            not self.output_cols
            or any(not isinstance(c, str) or not c for c in self.output_cols)
            or len(set(self.output_cols)) != len(self.output_cols)
        ):
            raise ValueError(
                "output_cols must contain unique, non-empty variable names."
            )
        unknown_outputs = [c for c in self.output_cols if c not in self.configs]
        if unknown_outputs:
            raise ValueError(
                f"output_cols contains variables without configs: {unknown_outputs}."
            )
        self._params: dict[str, FittedParams] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    # ── Public API ───────────────────────────────────────────────────────

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> Preprocessor:
        """
        Learn transformation parameters from the training DataFrame.

        Parameters
        ----------
        weights : np.ndarray, optional
            Exposure weights used to select the reference (dropped) level when
            one-hot encoding categorical variables.  The level with the highest
            total weight is dropped.  When ``None``, the first level
            alphabetically is dropped (legacy behaviour).
        """
        self._params = {}
        X_aug = self._materialize_raw_columns(X)
        for col in self._emitted_cols():
            cfg = self.configs[col]
            raw = X_aug[col]
            self._params[col] = self._fit_col(raw, cfg, weights=weights)
        self._compute_feature_names()
        self._fitted = True
        return self

    def transform(self, X: pl.DataFrame, strict: bool = True) -> pl.DataFrame:
        """Apply fitted transformations. Returns design matrix as pl.DataFrame.

        Parameters
        ----------
        strict : bool
            When True (default), raise on unseen categorical levels and missing
            un-imputed numerics.  Set to False only for weight-counting on
            out-of-sample data (e.g. summary_table calibration), where unknown
            rows are simply ignored rather than causing an error.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        out: dict[str, np.ndarray] = {}
        X_aug = self._materialize_raw_columns(X)
        for col in self._emitted_cols():
            cfg = self.configs[col]
            raw = X_aug[col]
            self._transform_col(raw, cfg, self._params[col], out, strict=strict)
        return pl.DataFrame(out)

    def fit_transform(
        self,
        X: pl.DataFrame,
        y: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> pl.DataFrame:
        return self.fit(X, y=y, weights=weights).transform(X)

    def get_feature_names(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() before requesting feature names.")
        return list(self.feature_names_)

    def _emitted_cols(self) -> list[str]:
        """Return outputs, including compatibility with older pickles."""
        return getattr(self, "output_cols", list(self.configs))

    # ── Raw series resolution ────────────────────────────────────────────

    def _materialize_raw_columns(
        self,
        X: pl.DataFrame,
        requested_cols: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Resolve requested raw variables and their dependencies once.

        Downstream transforms receive upstream custom-transform results before
        any configured cap, log, imputation, binning, standardisation, or
        encoding is applied.
        """
        resolved: set[str] = set()
        visiting: list[str] = []
        outputs = self._emitted_cols() if requested_cols is None else requested_cols
        materialized = X
        for output_col in outputs:
            materialized = self._materialize_raw_column(
                materialized,
                output_col,
                resolved,
                visiting,
            )
        return materialized

    def _materialize_raw_column(
        self,
        data: pl.DataFrame,
        column: str,
        resolved: set[str],
        visiting: list[str],
        dependent: str | None = None,
    ) -> pl.DataFrame:
        """Recursively materialize one raw variable and its dependencies."""
        if column in resolved:
            return data
        config = self.configs.get(column)
        if config is None:
            self._validate_unconfigured_input(data, column, dependent)
            resolved.add(column)
            return data
        if column in visiting:
            cycle = " -> ".join(visiting[visiting.index(column) :] + [column])
            raise ValueError(f"Circular dependency in derived variable chain: {cycle}.")
        if config.custom_transform is None:
            self._validate_plain_config(data, config)
            resolved.add(column)
            return data

        visiting.append(column)
        if config.input_cols is None:
            self._validate_single_column_transform(data, column)
        else:
            for input_column in config.input_cols:
                data = self._materialize_raw_column(
                    data,
                    input_column,
                    resolved,
                    visiting,
                    dependent=column,
                )
        raw = self._resolve_raw_series(data, config)
        if len(raw) != len(data):
            raise ValueError(
                f"custom_transform for variable '{column}' returned {len(raw)} rows; "
                f"expected {len(data)}."
            )
        visiting.pop()
        resolved.add(column)
        return data.with_columns(raw.alias(column))

    @staticmethod
    def _validate_unconfigured_input(
        data: pl.DataFrame,
        column: str,
        dependent: str | None,
    ) -> None:
        if column in data.columns:
            return
        context = f" for derived variable '{dependent}'" if dependent else ""
        raise ValueError(
            f"Input column '{column}'{context} is not in the DataFrame "
            "and has no registered config."
        )

    @staticmethod
    def _validate_plain_config(data: pl.DataFrame, config: VariableConfig) -> None:
        if config.input_cols is not None:
            raise ValueError(
                f"Variable '{config.col}' has input_cols but no custom_transform."
            )
        if config.col not in data.columns:
            raise ValueError(
                f"Variable '{config.col}' is not in the DataFrame and has no "
                "custom_transform capable of deriving it."
            )

    @staticmethod
    def _validate_single_column_transform(
        data: pl.DataFrame,
        column: str,
    ) -> None:
        if column not in data.columns:
            raise ValueError(
                f"Variable '{column}' uses a single-column custom_transform, "
                "but its source column is not in the DataFrame. Set input_cols "
                "when deriving a new variable name."
            )

    def _resolve_raw_series(self, X: pl.DataFrame, cfg: VariableConfig) -> pl.Series:
        """
        Return the raw (pre-pipeline) series for *cfg*.

        If ``cfg.custom_transform`` is set, it is called as
        ``custom_transform(df, **transform_kwargs)`` where *df* is a
        :class:`polars.DataFrame` containing only the relevant input columns
        (``input_cols`` when set, otherwise ``[col]``).  The result is wrapped
        as a :class:`pl.Series` named ``cfg.col``.

        For multi-input configs without a custom_transform a ``ValueError`` is
        raised, since there is no meaningful default combination.
        """
        if cfg.custom_transform is not None:
            cols = cfg.input_cols if cfg.input_cols is not None else [cfg.col]
            result = cfg.custom_transform(
                X.select(cols), **(cfg.transform_kwargs or {})
            )
            return pl.Series(cfg.col, result)
        if cfg.input_cols is not None:
            raise ValueError(
                f"Variable '{cfg.col}' has input_cols but no custom_transform."
            )
        return X[cfg.col]

    # ── Fitting ──────────────────────────────────────────────────────────

    def _fit_col(
        self,
        s: pl.Series,
        cfg: VariableConfig,
        weights: np.ndarray | None = None,
    ) -> FittedParams:
        is_cat = (
            _is_str_or_cat(s.dtype)
            if cfg.is_categorical is None
            else cfg.is_categorical
        )
        if is_cat:
            if (
                cfg.n_bins is not None
                or cfg.bin_edges is not None
                or cfg.standardize
                or cfg.degree != 1
                or cfg.log_transform
                or cfg.cap_lower is not None
                or cfg.cap_upper is not None
            ):
                raise ValueError(
                    f"Categorical variable '{cfg.col}' cannot use numeric transforms or binning."
                )
            return self._fit_cat(s, cfg, weights)
        if cfg.encoding not in {"auto", None}:
            raise ValueError(
                f"Numeric variable '{cfg.col}' cannot use encoding={cfg.encoding!r}."
            )
        return self._fit_num(s, cfg, weights)

    def _fit_cat(
        self, s: pl.Series, cfg: VariableConfig, weights: np.ndarray | None
    ) -> FittedCategoricalParams:
        impute_val = self._fit_cat_impute(s, cfg)
        encoding: Literal["onehot"] | None = (
            "onehot" if cfg.encoding in {"auto", "onehot"} else None
        )
        vals = self._normalize_cat_vals(s, impute_val)
        if any(v == _CAT_MISSING for v in vals):
            raise ValueError(
                f"Categorical variable '{cfg.col}' has missing values but no usable imputation strategy."
            )
        cats = tuple(sorted(set(vals)))
        if not cats:
            raise ValueError(
                f"Categorical variable '{cfg.col}' has no observed categories after imputation."
            )
        dropped = self._max_weight_category(vals, cats, weights)
        return FittedCategoricalParams(
            True, impute_val, encoding, tuple(c for c in cats if c != dropped), dropped
        )

    def _fit_num(
        self, s: pl.Series, cfg: VariableConfig, weights: np.ndarray | None
    ) -> FittedParams:
        arr = self._to_float_array(s, cfg.col)
        impute_val = self._fit_num_impute(arr, cfg)
        missing = ~np.isfinite(arr) | _is_sentinel(arr)
        if (
            missing.any()
            and impute_val is None
            and cfg.n_bins is None
            and cfg.bin_edges is None
        ):
            raise ValueError(
                f"Numeric variable '{cfg.col}' has missing values; configure imputation or binning."
            )
        imputed = arr.copy()
        if impute_val is not None:
            imputed[missing] = impute_val
        transformed = self._apply_num_transforms(
            imputed, cfg, cfg.cap_lower, cfg.cap_upper
        )
        common = {
            "is_categorical": False,
            "impute_val": impute_val,
            "cap_lower_val": cfg.cap_lower,
            "cap_upper_val": cfg.cap_upper,
            "right_closed": cfg.right_closed,
        }
        breaks = self._numeric_breaks(transformed, cfg, weights)
        if breaks is not None:
            return self._fit_binned_num(transformed, cfg, weights, breaks, common)
        return self._fit_continuous_num(transformed, cfg, common)

    @staticmethod
    def _numeric_breaks(
        transformed: np.ndarray,
        config: VariableConfig,
        weights: np.ndarray | None,
    ) -> np.ndarray | None:
        """Resolve explicit or weighted-quantile numeric breakpoints."""
        if config.bin_edges is not None:
            return np.asarray(config.bin_edges, dtype=float)
        if config.n_bins is not None:
            return compute_quantile_bin_edges(
                transformed,
                config.n_bins,
                weights,
            )[1:-1]
        return None

    @staticmethod
    def _fit_binned_num(
        transformed: np.ndarray,
        config: VariableConfig,
        weights: np.ndarray | None,
        breaks: np.ndarray,
        common: dict[str, Any],
    ) -> FittedBinnedNumericParams:
        """Fit labels, reference bin, and missing-bin state."""
        if not len(breaks):
            raise ValueError(
                f"Variable '{config.col}' cannot form bins from a constant or "
                "all-missing feature."
            )
        valid = transformed[np.isfinite(transformed) & ~_is_sentinel(transformed)]
        if not len(valid):
            raise ValueError(
                f"Variable '{config.col}' has no finite values for binning."
            )
        labels = tuple(make_bin_labels(breaks, config.right_closed))
        sentinel = _is_sentinel(transformed)
        cut = (
            pl.Series(config.col, transformed)
            .set(pl.Series(sentinel), None)
            .cut(
                list(breaks),
                labels=list(labels),
                left_closed=not config.right_closed,
            )
            .cast(pl.String)
        )
        effective_weights = (
            weights if weights is not None else np.ones(len(transformed))
        )
        bin_weights = np.array(
            [
                effective_weights[(cut == label).fill_null(False).to_numpy()].sum()
                for label in labels
            ]
        )
        return FittedBinnedNumericParams(
            **common,
            bin_edges=breaks,
            bin_labels=labels,
            dropped_bin=int(bin_weights.argmax()),
            has_sentinel_bin=bool(sentinel.any()),
        )

    @staticmethod
    def _fit_continuous_num(
        transformed: np.ndarray,
        config: VariableConfig,
        common: dict[str, Any],
    ) -> FittedNumericParams:
        """Fit optional standardization parameters for a continuous variable."""
        valid = transformed[np.isfinite(transformed) & ~_is_sentinel(transformed)]
        mean = (
            float(valid.mean())
            if config.standardize and len(valid)
            else (0.0 if config.standardize else None)
        )
        std = (
            max(float(valid.std()), 1e-10)
            if config.standardize and len(valid)
            else (1.0 if config.standardize else None)
        )
        return FittedNumericParams(**common, std_mean=mean, std_std=std)

    @staticmethod
    def _to_float_array(s: pl.Series, col: str) -> np.ndarray:
        try:
            # .copy() ensures a writable array — polars zero-copy views are read-only
            return (
                s.cast(pl.Float64, strict=True)
                .fill_null(MISSING_SENTINEL)
                .to_numpy(allow_copy=True)
                .copy()
            )
        except Exception as exc:
            raise ValueError(
                f"Numeric variable '{col}' cannot be cast to Float64."
            ) from exc

    @staticmethod
    def _fit_num_impute(arr: np.ndarray, cfg: VariableConfig) -> float | None:
        if cfg.impute_strategy is None:
            return None
        valid = arr[np.isfinite(arr) & ~_is_sentinel(arr)]
        if cfg.impute_strategy == "median":
            return float(np.median(valid)) if len(valid) else 0.0
        if cfg.impute_strategy == "mean":
            return float(np.mean(valid)) if len(valid) else 0.0
        if cfg.impute_strategy == "most_frequent":
            vals, counts = np.unique(valid, return_counts=True)
            return float(vals[counts.argmax()]) if len(vals) else 0.0
        return float(cfg.impute_value)

    @staticmethod
    def _fit_cat_impute(s: pl.Series, cfg: VariableConfig) -> str | None:
        strategy = cfg.impute_strategy or "most_frequent"
        if strategy == "most_frequent":
            modes = s.drop_nulls().cast(pl.String, strict=False).mode()
            return str(modes[0]) if len(modes) else None
        if strategy == "constant":
            return str(cfg.impute_value)
        if strategy in {"mean", "median"}:
            raise ValueError(
                f"Categorical variable '{cfg.col}' cannot use {strategy} imputation."
            )
        return None

    @staticmethod
    def _max_weight_category(
        vals: list[str], cats: Sequence[str], weights: np.ndarray | None
    ) -> str:
        if weights is None:
            return cats[0]
        totals = {cat: 0.0 for cat in cats}
        for value, weight in zip(vals, weights):
            totals[value] += float(weight)
        return min(cats, key=lambda cat: (-totals[cat], cat))

    @staticmethod
    def _apply_num_transforms(
        arr: np.ndarray,
        cfg: VariableConfig,
        cap_lower: float | None,
        cap_upper: float | None,
    ) -> np.ndarray:
        out = arr.copy()
        sentinel = _is_sentinel(out)
        if cap_lower is not None:
            out = np.where(sentinel, out, np.maximum(out, cap_lower))
        if cap_upper is not None:
            out = np.where(sentinel, out, np.minimum(out, cap_upper))
        if cfg.log_transform:
            valid = out[~sentinel]
            if len(valid) and (not np.all(np.isfinite(valid)) or np.any(valid <= 0)):
                raise ValueError(
                    f"Variable '{cfg.col}' cannot apply log_transform to zero, negative, or non-finite values."
                )
            out = np.where(sentinel, out, np.log(out))
        return out

    # ── Transformation ───────────────────────────────────────────────────
    @staticmethod
    def _normalize_cat_vals(s: pl.Series, impute_val: str | None) -> list[str]:
        vals = s.cast(pl.String, strict=False).fill_null(_CAT_MISSING).to_list()
        return [
            str(impute_val)
            if str(v) in {_CAT_MISSING, "None"} and impute_val is not None
            else str(v)
            for v in vals
        ]

    def _transform_col(
        self,
        s: pl.Series,
        cfg: VariableConfig,
        p: FittedParams,
        out: dict[str, np.ndarray],
        strict: bool = True,
    ) -> None:
        if isinstance(p, FittedCategoricalParams):
            self._transform_cat(s, cfg, p, out, strict=strict)
        else:
            self._transform_num(s, cfg, p, out, strict=strict)

    def _transform_cat(
        self,
        s: pl.Series,
        cfg: VariableConfig,
        p: FittedCategoricalParams,
        out: dict[str, np.ndarray],
        strict: bool = True,
    ) -> None:
        vals = self._normalize_cat_vals(s, p.impute_val)
        known = set(p.categories) | (
            {p.dropped_category} if p.dropped_category is not None else set()
        )
        unknown = sorted(set(vals) - known)
        if unknown and strict:
            raise ValueError(
                f"Variable '{cfg.col}' contains unseen categorical levels: {unknown[:10]}."
            )
        if p.encoding == "onehot":
            # Rows with unseen levels produce 0 in every dummy column (no contribution to any bucket).
            vals_arr = np.asarray(vals, dtype=object)
            for cat in p.categories:
                out[f"{cfg.col}_{cat}"] = (vals_arr == cat).astype(float)
        else:
            out[cfg.col] = np.asarray(vals, dtype=object)

    def _transform_num(
        self,
        s: pl.Series,
        cfg: VariableConfig,
        p: FittedNumericParams | FittedBinnedNumericParams,
        out: dict[str, np.ndarray],
        strict: bool = True,
    ) -> None:
        arr = self._to_float_array(s, cfg.col)
        sentinel = _is_sentinel(arr)
        missing = ~np.isfinite(arr) | sentinel
        if missing.any() and p.impute_val is not None:
            arr[missing] = float(p.impute_val)
            sentinel = _is_sentinel(arr)
        if (
            missing.any()
            and p.impute_val is None
            and not isinstance(p, FittedBinnedNumericParams)
        ):
            if strict:
                raise ValueError(
                    f"Numeric variable '{cfg.col}' contains missing values but was fitted without imputation."
                )
            arr[missing] = (
                0.0  # neutral fill; col is excluded from summary_table output anyway
            )
        transformed = self._apply_num_transforms(
            arr, cfg, p.cap_lower_val, p.cap_upper_val
        )
        if isinstance(p, FittedBinnedNumericParams):
            sentinel = _is_sentinel(transformed)
            labelled = (
                pl.Series(cfg.col, transformed)
                .set(pl.Series(sentinel), None)
                .cut(
                    list(p.bin_edges),
                    labels=list(p.bin_labels),
                    left_closed=not p.right_closed,
                )
                .cast(pl.String)
            )
            if p.has_sentinel_bin:
                out[f"{cfg.col}_missing"] = sentinel.astype(float)
            for i, label in enumerate(p.bin_labels):
                if i != p.dropped_bin:
                    out[f"{cfg.col}_{label}"] = (
                        (labelled == label).fill_null(False).to_numpy().astype(float)
                    )
            return
        if p.std_mean is not None and p.std_std is not None:
            transformed = (transformed - p.std_mean) / p.std_std
        out[cfg.col] = transformed
        for degree in range(2, cfg.degree + 1):
            out[f"{cfg.col}^{degree}"] = transformed**degree

    def _compute_feature_names(self) -> None:
        names: list[str] = []
        for col in self._emitted_cols():
            cfg = self.configs[col]
            p = self._params[col]
            if isinstance(p, FittedCategoricalParams):
                if p.encoding == "onehot":
                    names.extend(f"{col}_{cat}" for cat in p.categories)
                else:
                    names.append(col)
            elif isinstance(p, FittedBinnedNumericParams):
                if p.has_sentinel_bin:
                    names.append(f"{col}_missing")
                names.extend(
                    f"{col}_{label}"
                    for i, label in enumerate(p.bin_labels)
                    if i != p.dropped_bin
                )
            else:
                names.append(col)
                names.extend(f"{col}^{d}" for d in range(2, cfg.degree + 1))
        self.feature_names_ = names

    def get_bin_labels(self, col: str, s: pl.Series) -> pl.Series:
        if (
            not self._fitted
            or col not in self._params
            or not isinstance(self._params[col], FittedBinnedNumericParams)
        ):
            raise ValueError(f"'{col}' has no fitted bin edges.")
        cfg, p = self.configs[col], self._params[col]
        assert isinstance(p, FittedBinnedNumericParams)
        arr = self._to_float_array(s, col)
        sentinel = _is_sentinel(arr)
        if p.impute_val is not None:
            arr[~np.isfinite(arr) | sentinel] = float(p.impute_val)
            sentinel = _is_sentinel(arr)
        transformed = self._apply_num_transforms(
            arr, cfg, p.cap_lower_val, p.cap_upper_val
        )
        return (
            pl.Series(col, transformed)
            .set(pl.Series(sentinel), None)
            .cut(
                list(p.bin_edges),
                labels=list(p.bin_labels),
                left_closed=not p.right_closed,
            )
            .cast(pl.String)
            .fill_null("Missing")
            .rename(col + "_label")
        )

    def get_level_labels(self, col: str, X: pl.DataFrame) -> pl.Series:
        if not self._fitted or col not in self._params:
            raise ValueError(f"'{col}' is not in a fitted preprocessor.")
        p = self._params[col]
        raw = self._materialize_raw_columns(X, [col])[col]
        if isinstance(p, FittedBinnedNumericParams):
            return self.get_bin_labels(col, raw)
        if isinstance(p, FittedCategoricalParams):
            vals = self._normalize_cat_vals(raw, p.impute_val)
            known = set(p.categories) | (
                {p.dropped_category} if p.dropped_category else set()
            )
            unknown = sorted(set(vals) - known)
            if unknown:
                raise ValueError(
                    f"Variable '{col}' contains unseen categorical levels: "
                    f"{unknown[:10]}."
                )
            return pl.Series(
                col + "_label", ["Missing" if v == _CAT_MISSING else v for v in vals]
            )
        return raw.cast(pl.String).fill_null("Missing").rename(col + "_label")
