"""Variable configuration and robust, fitted preprocessing (polars backend)."""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

MISSING_SENTINEL: float = -999_999_999.0
_CAT_MISSING = "__MISSING__"
_NUMERIC_DTYPES = frozenset({pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8,
    pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64})
_IMPUTE_STRATEGIES = frozenset({"median", "mean", "most_frequent", "constant"})


def _is_str_or_cat(dtype: pl.PolarsDataType) -> bool:
    return dtype not in _NUMERIC_DTYPES and dtype not in (pl.Boolean, pl.Date,
        pl.Datetime, pl.Duration, pl.Time, pl.Null)


def _is_sentinel(arr: np.ndarray) -> np.ndarray:
    return np.isclose(arr, MISSING_SENTINEL, rtol=0, atol=1.0)


@dataclass
class VariableConfig:
    """Declarative preprocessing configuration for one output variable.

    ``custom_transform`` receives a Polars DataFrame containing ``input_cols``
    (or ``[col]``) and must return one scalar value per input row.
    """
    col: str
    input_cols: Optional[List[str]] = None
    cap_lower: Optional[float] = None
    cap_upper: Optional[float] = None
    log_transform: bool = False
    impute_strategy: Optional[str] = None
    impute_value: Optional[Any] = None
    n_bins: Optional[int] = None
    bin_edges: Optional[List[float]] = None
    standardize: bool = False
    degree: int = 1
    encoding: Optional[str] = "auto"
    is_categorical: Optional[bool] = None
    right_closed: bool = False
    custom_transform: Optional[Callable[..., Any]] = None
    transform_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.col, str) or not self.col.strip():
            raise ValueError("VariableConfig.col must be a non-empty string.")
        if self.input_cols is not None:
            if not self.input_cols or any(not isinstance(c, str) or not c for c in self.input_cols):
                raise ValueError(f"Variable '{self.col}' input_cols must be non-empty strings.")
            if len(set(self.input_cols)) != len(self.input_cols):
                raise ValueError(f"Variable '{self.col}' input_cols must not contain duplicates.")
            if self.custom_transform is None:
                raise ValueError(f"Variable '{self.col}' has input_cols but no custom_transform.")
        if self.custom_transform is not None and not callable(self.custom_transform):
            raise TypeError(f"Variable '{self.col}' custom_transform must be callable.")
        if self.transform_kwargs is not None and not isinstance(self.transform_kwargs, dict):
            raise TypeError(f"Variable '{self.col}' transform_kwargs must be a dict or None.")
        if self.impute_strategy not in _IMPUTE_STRATEGIES | {None}:
            raise ValueError(f"Variable '{self.col}' has unsupported impute_strategy {self.impute_strategy!r}.")
        if self.impute_strategy == "constant" and self.impute_value is None:
            raise ValueError(f"Variable '{self.col}' requires impute_value for constant imputation.")
        if self.encoding not in {"auto", "onehot", None}:
            raise ValueError(f"Variable '{self.col}' encoding must be 'auto', 'onehot', or None.")
        if not isinstance(self.degree, int) or isinstance(self.degree, bool) or self.degree < 1:
            raise ValueError(f"Variable '{self.col}' degree must be a positive integer.")
        if self.n_bins is not None and (not isinstance(self.n_bins, int) or isinstance(self.n_bins, bool) or self.n_bins < 2):
            raise ValueError(f"Variable '{self.col}' n_bins must be an integer >= 2.")
        if self.n_bins is not None and self.bin_edges is not None:
            raise ValueError(f"Variable '{self.col}' cannot set both n_bins and bin_edges.")
        for name, value in (("cap_lower", self.cap_lower), ("cap_upper", self.cap_upper)):
            if value is not None and (not np.isscalar(value) or not np.isfinite(float(value))):
                raise ValueError(f"Variable '{self.col}' {name} must be finite when supplied.")
        if self.cap_lower is not None and self.cap_upper is not None and self.cap_lower > self.cap_upper:
            raise ValueError(f"Variable '{self.col}' cap_lower cannot exceed cap_upper.")
        if self.bin_edges is not None:
            edges = np.asarray(self.bin_edges, dtype=float)
            if edges.ndim != 1 or len(edges) == 0 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
                raise ValueError(f"Variable '{self.col}' bin_edges must be a non-empty, finite, strictly increasing sequence.")
            self.bin_edges = edges.tolist()


@dataclass(frozen=True)
class FittedVariableParams:
    is_categorical: bool
    impute_val: Optional[Union[float, str]]


@dataclass(frozen=True)
class FittedCategoricalParams(FittedVariableParams):
    encoding: Optional[Literal["onehot"]]
    categories: tuple[str, ...] = ()
    dropped_category: Optional[str] = None


@dataclass(frozen=True)
class FittedNumericParams(FittedVariableParams):
    cap_lower_val: Optional[float] = None
    cap_upper_val: Optional[float] = None
    right_closed: bool = False
    std_mean: Optional[float] = None
    std_std: Optional[float] = None


@dataclass(frozen=True)
class FittedBinnedNumericParams(FittedNumericParams):
    bin_edges: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bin_labels: tuple[str, ...] = ()
    dropped_bin: int = 0
    has_sentinel_bin: bool = False


FittedParams = Union[FittedCategoricalParams, FittedNumericParams, FittedBinnedNumericParams]


def default_config(col: str, s: pl.Series) -> VariableConfig:
    if _is_str_or_cat(s.dtype):
        return VariableConfig(col=col, impute_strategy="most_frequent", encoding="auto", is_categorical=True)
    return VariableConfig(col=col, is_categorical=False)


def compute_quantile_bin_edges(arr: np.ndarray, n_bins: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2.")
    values = np.asarray(arr, dtype=float)
    if values.ndim != 1:
        raise ValueError("arr must be one-dimensional.")
    valid_mask = np.isfinite(values) & ~_is_sentinel(values)
    valid = values[valid_mask]
    if weights is not None:
        w = _validate_weights(weights, len(values))
        w_valid = w[valid_mask]
    else:
        w_valid = None
    if not len(valid):
        return np.array([-np.inf, np.inf])
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(valid, percentiles, weights=w_valid, method="inverted_cdf")) if w_valid is not None else np.unique(np.percentile(valid, percentiles))
    if len(edges) < 2:
        value = float(valid[0])
        return np.array([value - 1e-9, value + 1e-9])
    return edges


def _fmt(x: float, dec: int = 3) -> str:
    return str(int(x)) if x == int(x) else format(round(x, dec), f",.{dec}f")


def _bin_letter(index: int) -> str:
    letters = string.ascii_uppercase
    return letters[index] if index < 26 else letters[index // 26 - 1] + letters[index % 26]


def make_bin_labels(breaks: Sequence[float], min_val: float, max_val: float, right: bool = False, dec: int = 3) -> List[str]:
    edges = np.asarray(breaks, dtype=float)
    if edges.ndim != 1 or not len(edges):
        raise ValueError("breaks must be a non-empty one-dimensional sequence.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("breaks must be strictly increasing.")
    labels = [f"{_bin_letter(0)}_[{_fmt(float(min_val), dec)}, {_fmt(edges[0], dec)}{']' if right else ')'}"]
    for i in range(1, len(edges)):
        labels.append(f"{_bin_letter(i)}_{'(' if right else '['}{_fmt(edges[i - 1], dec)}, {_fmt(edges[i], dec)}{']' if right else ')'}")
    labels.append(f"{_bin_letter(len(edges))}_({_fmt(edges[-1], dec)}, {_fmt(float(max_val), dec)}]")
    return labels


def _validate_weights(weights: np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or len(arr) != n_rows:
        raise ValueError(f"weights must be one-dimensional with length {n_rows}.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or arr.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have a positive total.")
    return arr


class Preprocessor:
    """Fit and apply validated variable transformations to Polars DataFrames."""
    def __init__(self, configs: List[VariableConfig]):
        if not configs:
            raise ValueError("Preprocessor requires at least one VariableConfig.")
        if any(not isinstance(c, VariableConfig) for c in configs):
            raise TypeError("configs must contain VariableConfig instances.")
        if len({c.col for c in configs}) != len(configs):
            raise ValueError("Preprocessor configs must have unique output names.")
        self.configs = {c.col: c for c in configs}
        self._params: Dict[str, FittedParams] = {}
        self.feature_names_: List[str] = []
        self._fitted = False

    def fit(self, X: pl.DataFrame, y: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> "Preprocessor":
        self._validate_X(X, "fit")
        if len(X) == 0:
            raise ValueError("Cannot fit a Preprocessor on an empty DataFrame.")
        if y is not None and (np.asarray(y).ndim != 1 or len(y) != len(X)):
            raise ValueError("y must be one-dimensional and aligned with X.")
        w = _validate_weights(weights, len(X)) if weights is not None else None
        self._params = {col: self._fit_col(self._resolve_raw_series(X, cfg), cfg, w) for col, cfg in self.configs.items()}
        self._compute_feature_names()
        self._fitted = True
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        self._validate_X(X, "transform")
        out: Dict[str, np.ndarray] = {}
        for col, cfg in self.configs.items():
            self._transform_col(self._resolve_raw_series(X, cfg), cfg, self._params[col], out)
        return pl.DataFrame(out)

    def fit_transform(self, X: pl.DataFrame, y: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> pl.DataFrame:
        return self.fit(X, y=y, weights=weights).transform(X)

    def get_feature_names(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() before requesting feature names.")
        return list(self.feature_names_)

    def _validate_X(self, X: pl.DataFrame, operation: str) -> None:
        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"X for {operation} must be a polars DataFrame.")
        required = {source for cfg in self.configs.values() for source in (cfg.input_cols or [cfg.col])}
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"X for {operation} is missing required columns: {missing}.")

    def _resolve_raw_series(self, X: pl.DataFrame, cfg: VariableConfig) -> pl.Series:
        if cfg.custom_transform is None:
            return X[cfg.col]
        cols = cfg.input_cols or [cfg.col]
        try:
            result = cfg.custom_transform(X.select(cols), **(cfg.transform_kwargs or {}))
        except Exception as exc:
            raise ValueError(f"custom_transform for variable '{cfg.col}' failed: {exc}") from exc
        arr = np.asarray(result)
        if arr.ndim != 1 or len(arr) != len(X):
            raise ValueError(f"custom_transform for variable '{cfg.col}' must return one value per row ({len(X)}); got shape {arr.shape}.")
        return pl.Series(cfg.col, result)

    def _fit_col(self, s: pl.Series, cfg: VariableConfig, weights: Optional[np.ndarray]) -> FittedParams:
        is_cat = _is_str_or_cat(s.dtype) if cfg.is_categorical is None else cfg.is_categorical
        if is_cat:
            if cfg.n_bins is not None or cfg.bin_edges is not None or cfg.standardize or cfg.degree != 1 or cfg.log_transform or cfg.cap_lower is not None or cfg.cap_upper is not None:
                raise ValueError(f"Categorical variable '{cfg.col}' cannot use numeric transforms or binning.")
            return self._fit_cat(s, cfg, weights)
        if cfg.encoding not in {"auto", None}:
            raise ValueError(f"Numeric variable '{cfg.col}' cannot use encoding={cfg.encoding!r}.")
        return self._fit_num(s, cfg, weights)

    def _fit_cat(self, s: pl.Series, cfg: VariableConfig, weights: Optional[np.ndarray]) -> FittedCategoricalParams:
        impute_val = self._fit_cat_impute(s, cfg)
        encoding: Optional[Literal["onehot"]] = "onehot" if cfg.encoding in {"auto", "onehot"} else None
        vals = self._normalize_cat_vals(s, impute_val)
        if any(v == _CAT_MISSING for v in vals):
            raise ValueError(f"Categorical variable '{cfg.col}' has missing values but no usable imputation strategy.")
        cats = tuple(sorted(set(vals)))
        if not cats:
            raise ValueError(f"Categorical variable '{cfg.col}' has no observed categories after imputation.")
        dropped = self._max_weight_category(vals, cats, weights)
        return FittedCategoricalParams(True, impute_val, encoding, tuple(c for c in cats if c != dropped), dropped)

    def _fit_num(self, s: pl.Series, cfg: VariableConfig, weights: Optional[np.ndarray]) -> FittedParams:
        arr = self._to_float_array(s, cfg.col)
        impute_val = self._fit_num_impute(arr, cfg)
        missing = ~np.isfinite(arr) | _is_sentinel(arr)
        if missing.any() and impute_val is None and cfg.n_bins is None and cfg.bin_edges is None:
            raise ValueError(f"Numeric variable '{cfg.col}' has missing values; configure imputation or binning.")
        imputed = arr.copy()
        if impute_val is not None:
            imputed[missing] = impute_val
        transformed = self._apply_num_transforms(imputed, cfg, cfg.cap_lower, cfg.cap_upper)
        breaks = np.asarray(cfg.bin_edges, dtype=float) if cfg.bin_edges is not None else None
        if cfg.n_bins is not None:
            breaks = compute_quantile_bin_edges(transformed, cfg.n_bins, weights)
            breaks = breaks[1:-1]
        common = dict(is_categorical=False, impute_val=impute_val, cap_lower_val=cfg.cap_lower,
                      cap_upper_val=cfg.cap_upper, right_closed=cfg.right_closed)
        if breaks is not None:
            if not len(breaks):
                raise ValueError(f"Variable '{cfg.col}' cannot form bins from a constant or all-missing feature.")
            valid = transformed[np.isfinite(transformed) & ~_is_sentinel(transformed)]
            if not len(valid):
                raise ValueError(f"Variable '{cfg.col}' has no finite values for binning.")
            labels = tuple(make_bin_labels(breaks, float(valid.min()), float(valid.max()), cfg.right_closed))
            sentinel = _is_sentinel(transformed)
            cut = pl.Series(cfg.col, transformed).set(pl.Series(sentinel), None).cut(list(breaks), labels=list(labels), left_closed=not cfg.right_closed).cast(pl.String)
            eff_weights = weights if weights is not None else np.ones(len(transformed))
            bin_weights = np.array([eff_weights[(cut == label).fill_null(False).to_numpy()].sum() for label in labels])
            return FittedBinnedNumericParams(**common, bin_edges=breaks, bin_labels=labels,
                dropped_bin=int(bin_weights.argmax()), has_sentinel_bin=bool(sentinel.any()))
        valid = transformed[np.isfinite(transformed) & ~_is_sentinel(transformed)]
        mean = float(valid.mean()) if cfg.standardize and len(valid) else (0.0 if cfg.standardize else None)
        std = max(float(valid.std()), 1e-10) if cfg.standardize and len(valid) else (1.0 if cfg.standardize else None)
        return FittedNumericParams(**common, std_mean=mean, std_std=std)

    @staticmethod
    def _to_float_array(s: pl.Series, col: str) -> np.ndarray:
        try:
            return s.cast(pl.Float64, strict=True).fill_null(MISSING_SENTINEL).to_numpy(allow_copy=True)
        except Exception as exc:
            raise ValueError(f"Numeric variable '{col}' cannot be cast to Float64.") from exc

    @staticmethod
    def _fit_num_impute(arr: np.ndarray, cfg: VariableConfig) -> Optional[float]:
        if cfg.impute_strategy is None:
            return None
        valid = arr[np.isfinite(arr) & ~_is_sentinel(arr)]
        if cfg.impute_strategy == "median": return float(np.median(valid)) if len(valid) else 0.0
        if cfg.impute_strategy == "mean": return float(np.mean(valid)) if len(valid) else 0.0
        if cfg.impute_strategy == "most_frequent":
            vals, counts = np.unique(valid, return_counts=True)
            return float(vals[counts.argmax()]) if len(vals) else 0.0
        return float(cfg.impute_value)

    @staticmethod
    def _fit_cat_impute(s: pl.Series, cfg: VariableConfig) -> Optional[str]:
        strategy = cfg.impute_strategy or "most_frequent"
        if strategy == "most_frequent":
            modes = s.drop_nulls().cast(pl.String, strict=False).mode()
            return str(modes[0]) if len(modes) else None
        if strategy == "constant": return str(cfg.impute_value)
        if strategy in {"mean", "median"}: raise ValueError(f"Categorical variable '{cfg.col}' cannot use {strategy} imputation.")
        return None

    @staticmethod
    def _max_weight_category(vals: List[str], cats: Sequence[str], weights: Optional[np.ndarray]) -> str:
        if weights is None: return cats[0]
        totals = {cat: 0.0 for cat in cats}
        for value, weight in zip(vals, weights): totals[value] += float(weight)
        return min(cats, key=lambda cat: (-totals[cat], cat))

    @staticmethod
    def _apply_num_transforms(arr: np.ndarray, cfg: VariableConfig, cap_lower: Optional[float], cap_upper: Optional[float]) -> np.ndarray:
        out = arr.copy(); sentinel = _is_sentinel(out)
        if cap_lower is not None: out = np.where(sentinel, out, np.maximum(out, cap_lower))
        if cap_upper is not None: out = np.where(sentinel, out, np.minimum(out, cap_upper))
        if cfg.log_transform:
            valid = out[~sentinel]
            if len(valid) and (not np.all(np.isfinite(valid)) or np.any(valid <= 0)):
                raise ValueError(f"Variable '{cfg.col}' cannot apply log_transform to zero, negative, or non-finite values.")
            out = np.where(sentinel, out, np.log(out))
        return out

    @staticmethod
    def _normalize_cat_vals(s: pl.Series, impute_val: Optional[str]) -> List[str]:
        vals = s.cast(pl.String, strict=False).fill_null(_CAT_MISSING).to_list()
        return [str(impute_val) if str(v) in {_CAT_MISSING, "None"} and impute_val is not None else str(v) for v in vals]

    def _transform_col(self, s: pl.Series, cfg: VariableConfig, p: FittedParams, out: Dict[str, np.ndarray]) -> None:
        if isinstance(p, FittedCategoricalParams): self._transform_cat(s, cfg, p, out)
        else: self._transform_num(s, cfg, p, out)

    def _transform_cat(self, s: pl.Series, cfg: VariableConfig, p: FittedCategoricalParams, out: Dict[str, np.ndarray]) -> None:
        vals = self._normalize_cat_vals(s, p.impute_val)
        known = set(p.categories) | ({p.dropped_category} if p.dropped_category is not None else set())
        unknown = sorted(set(vals) - known)
        if unknown:
            raise ValueError(f"Variable '{cfg.col}' contains unseen categorical levels: {unknown[:10]}.")
        if p.encoding == "onehot":
            for cat in p.categories: out[f"{cfg.col}_{cat}"] = (np.asarray(vals, dtype=object) == cat).astype(float)
        else: out[cfg.col] = np.asarray(vals, dtype=object)

    def _transform_num(self, s: pl.Series, cfg: VariableConfig, p: Union[FittedNumericParams, FittedBinnedNumericParams], out: Dict[str, np.ndarray]) -> None:
        arr = self._to_float_array(s, cfg.col); sentinel = _is_sentinel(arr); missing = ~np.isfinite(arr) | sentinel
        if missing.any() and p.impute_val is not None: arr[missing] = float(p.impute_val); sentinel = _is_sentinel(arr)
        if missing.any() and p.impute_val is None and not isinstance(p, FittedBinnedNumericParams):
            raise ValueError(f"Numeric variable '{cfg.col}' contains missing values but was fitted without imputation.")
        transformed = self._apply_num_transforms(arr, cfg, p.cap_lower_val, p.cap_upper_val)
        if isinstance(p, FittedBinnedNumericParams):
            sentinel = _is_sentinel(transformed)
            labelled = pl.Series(cfg.col, transformed).set(pl.Series(sentinel), None).cut(list(p.bin_edges), labels=list(p.bin_labels), left_closed=not p.right_closed).cast(pl.String)
            if p.has_sentinel_bin: out[f"{cfg.col}_missing"] = sentinel.astype(float)
            for i, label in enumerate(p.bin_labels):
                if i != p.dropped_bin: out[f"{cfg.col}_{label}"] = (labelled == label).fill_null(False).to_numpy().astype(float)
            return
        if p.std_mean is not None and p.std_std is not None: transformed = (transformed - p.std_mean) / p.std_std
        out[cfg.col] = transformed
        for degree in range(2, cfg.degree + 1): out[f"{cfg.col}^{degree}"] = transformed ** degree

    def _compute_feature_names(self) -> None:
        names: List[str] = []
        for col, cfg in self.configs.items():
            p = self._params[col]
            if isinstance(p, FittedCategoricalParams):
                names.extend(f"{col}_{cat}" for cat in p.categories) if p.encoding == "onehot" else names.append(col)
            elif isinstance(p, FittedBinnedNumericParams):
                if p.has_sentinel_bin: names.append(f"{col}_missing")
                names.extend(f"{col}_{label}" for i, label in enumerate(p.bin_labels) if i != p.dropped_bin)
            else:
                names.append(col); names.extend(f"{col}^{d}" for d in range(2, cfg.degree + 1))
        self.feature_names_ = names

    def get_bin_labels(self, col: str, s: pl.Series) -> pl.Series:
        if not self._fitted or col not in self._params or not isinstance(self._params[col], FittedBinnedNumericParams):
            raise ValueError(f"'{col}' has no fitted bin edges.")
        cfg, p = self.configs[col], self._params[col]
        assert isinstance(p, FittedBinnedNumericParams)
        arr = self._to_float_array(s, col); sentinel = _is_sentinel(arr)
        if p.impute_val is not None: arr[~np.isfinite(arr) | sentinel] = float(p.impute_val); sentinel = _is_sentinel(arr)
        transformed = self._apply_num_transforms(arr, cfg, p.cap_lower_val, p.cap_upper_val)
        return pl.Series(col, transformed).set(pl.Series(sentinel), None).cut(list(p.bin_edges), labels=list(p.bin_labels), left_closed=not p.right_closed).cast(pl.String).fill_null("Missing").rename(col + "_label")

    def get_level_labels(self, col: str, X: pl.DataFrame) -> pl.Series:
        if not self._fitted or col not in self.configs: raise ValueError(f"'{col}' is not in a fitted preprocessor.")
        self._validate_X(X, "get_level_labels")
        p, cfg = self._params[col], self.configs[col]; raw = self._resolve_raw_series(X, cfg)
        if isinstance(p, FittedBinnedNumericParams): return self.get_bin_labels(col, raw)
        if isinstance(p, FittedCategoricalParams):
            vals = self._normalize_cat_vals(raw, p.impute_val); known = set(p.categories) | ({p.dropped_category} if p.dropped_category else set())
            unknown = sorted(set(vals) - known)
            if unknown: raise ValueError(f"Variable '{col}' contains unseen categorical levels: {unknown[:10]}.")
            return pl.Series(col + "_label", ["Missing" if v == _CAT_MISSING else v for v in vals])
        return raw.cast(pl.String).fill_null("Missing").rename(col + "_label")
