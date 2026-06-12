"""Variable configuration and preprocessing pipeline (polars backend)."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import polars as pl


# ── Constants ─────────────────────────────────────────────────────────────────

MISSING_SENTINEL: float = -999_999_999.0
"""Sentinel value used to indicate 'missing' in continuous variables during binning."""

_CAT_MISSING: str = "__MISSING__"
"""Placeholder used to represent missing values in categorical columns before imputation."""

# Polars numeric dtypes
_NUMERIC_DTYPES = frozenset({
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
})


def _is_str_or_cat(dtype: pl.PolarsDataType) -> bool:
    """Return True for string / categorical polars dtypes."""
    return dtype not in _NUMERIC_DTYPES and dtype not in (pl.Boolean, pl.Date,
                                                           pl.Datetime, pl.Duration,
                                                           pl.Time, pl.Null)


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
    columns.  ``custom_transform`` is then called as::

        custom_transform(arr_col1, arr_col2, ...) -> np.ndarray

    where each positional argument is the numpy array for the corresponding
    entry in ``input_cols``.  The result is treated as a new numeric (or
    categorical, if ``is_categorical=True``) column named ``col``.

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
    custom_transform : callable, optional
        May be a **named function** or a lambda.  Any callable is accepted.

        **Numeric** single-col: ``f(arr: np.ndarray, **kw) -> np.ndarray``,
        applied before capping / log / binning.

        **Categorical** single-col: ``f(val: Any, **kw) -> Any``, applied
        element-wise before encoding (can remap/group categories).

        **Multi-col** (``input_cols`` set): ``f(*arrays, **kw) -> np.ndarray``,
        called once with each input column's numpy array as positional args.
    transform_kwargs : dict, optional
        Keyword arguments forwarded to ``custom_transform`` on every call.
        Useful for passing parameters to a named function without a closure::

            def scale(arr, factor=1.0):
                return arr / factor

            VariableConfig('mileage', custom_transform=scale,
                           transform_kwargs={'factor': 1000})
    """

    col: str
    input_cols: Optional[List[str]] = None
    cap_lower: Optional[float] = None
    cap_upper: Optional[float] = None
    log_transform: bool = False
    impute_strategy: Optional[str] = "median"
    impute_value: Optional[Any] = None
    n_bins: Optional[int] = None
    bin_edges: Optional[List[float]] = None
    standardize: bool = False
    degree: int = 1
    encoding: Optional[str] = "auto"
    is_categorical: Optional[bool] = None
    custom_transform: Optional[Callable[..., Any]] = None
    transform_kwargs: Optional[Dict[str, Any]] = None


# ── Default config ────────────────────────────────────────────────────────────

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
        cap_upper=0.99,
        impute_strategy="median",
        is_categorical=False,
    )


# ── Binning helpers ───────────────────────────────────────────────────────────

def compute_quantile_bin_edges(
    arr: np.ndarray, n_bins: int, weights: Optional[np.ndarray] = None
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
        edges = np.unique(np.percentile(valid, percentiles, weights=w_valid, method="inverted_cdf"))
    else:
        edges = np.unique(np.percentile(valid, percentiles))

    if len(edges) < 2:
        mn, mx = float(valid.min()), float(valid.max())
        edges = np.array([mn - 1e-9, mx + 1e-9])
    return edges


def _fmt_edge(v: float) -> str:
    """Format a bin edge: no decimal point for whole numbers, else ≤3 decimal places."""
    if v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _bin_letter(i: int) -> str:
    """0→'A', 1→'B', …, 25→'Z', 26→'AA', 27→'AB', …"""
    letters = string.ascii_uppercase
    if i < 26:
        return letters[i]
    return letters[i // 26 - 1] + letters[i % 26]


def make_bin_labels(breaks: np.ndarray) -> List[str]:
    """
    Return human-readable label strings for the n+1 bins defined by *n* break points.

    Labels:
      bin 0 (all below first break): ``'{letter}_<hi'``
      bins 1 … n-1 (interior):       ``'{letter}_[lo, hi)'``
      bin n (all above last break):   ``'{letter}_lo+'``
    """
    n = len(breaks)
    if n == 0:
        return ["A_all"]
    labels: List[str] = []
    for i in range(n + 1):
        letter = _bin_letter(i)
        if i == 0:
            hi = _fmt_edge(float(breaks[0]))
            labels.append(f"{letter}_<{hi}")
        elif i == n:
            lo = _fmt_edge(float(breaks[-1]))
            labels.append(f"{letter}_{lo}+")
        else:
            lo = _fmt_edge(float(breaks[i - 1]))
            hi = _fmt_edge(float(breaks[i]))
            labels.append(f"{letter}_[{lo}, {hi})")
    return labels


# ── Preprocessor ─────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Fits variable transformations on training data and applies them to any
    polars DataFrame with matching columns.

    Parameters
    ----------
    configs : list of VariableConfig
    """

    def __init__(self, configs: List[VariableConfig]):
        self.configs: Dict[str, VariableConfig] = {c.col: c for c in configs}
        self._params: Dict[str, Dict[str, Any]] = {}
        self.feature_names_: List[str] = []
        self._fitted = False

    # ── Public API ───────────────────────────────────────────────────────

    def fit(
        self,
        X: pl.DataFrame,
        y=None,
        weights: Optional[pl.Series] = None,
    ) -> "Preprocessor":
        """
        Learn transformation parameters from the training DataFrame.

        Parameters
        ----------
        weights : pl.Series, optional
            Exposure weights used to select the reference (dropped) level when
            one-hot encoding categorical variables.  The level with the highest
            total weight is dropped.  When ``None``, the first level
            alphabetically is dropped (legacy behaviour).
        """
        self._params = {}
        for col, cfg in self.configs.items():
            raw = self._resolve_raw_series(X, cfg)
            self._params[col] = self._fit_col(raw, cfg, weights=weights)
        self._compute_feature_names()
        self._fitted = True
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply fitted transformations. Returns design matrix as pl.DataFrame."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        out: Dict[str, np.ndarray] = {}
        for col, cfg in self.configs.items():
            raw = self._resolve_raw_series(X, cfg)
            self._transform_col(raw, cfg, self._params[col], out)
        return pl.DataFrame(out)

    def fit_transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names_)

    # ── Raw series resolution ────────────────────────────────────────────

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
            result = cfg.custom_transform(X.select(cols), **(cfg.transform_kwargs or {}))
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
        weights: Optional[pl.Series] = None,
    ) -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        is_cat = self._detect_categorical(s, cfg)
        p["is_categorical"] = is_cat

        if is_cat:
            p["impute_val"] = self._fit_cat_impute(s, cfg)

            enc = "onehot" if cfg.encoding in ("auto", "onehot") else None
            p["encoding"] = enc

            if enc == "onehot":
                # s already has custom_transform applied (via _resolve_raw_series)
                raw_arr = s.cast(pl.Utf8, strict=False).fill_null(_CAT_MISSING)
                cats = raw_arr.filter(~raw_arr.is_in([_CAT_MISSING, "None", None])).unique().sort()
                drop_cat = self._max_weight_category(raw_arr, cats, weights)
                p["categories"] = [c for c in cats if c != drop_cat]
                p["dropped_category"] = drop_cat
            else:
                p["categories"] = []
        else:
            arr = self._to_float_array(s)
            p["impute_val"] = self._fit_num_impute(arr, cfg)

            imputed = arr.copy()
            nan_mask = ~np.isfinite(imputed) | _is_sentinel(imputed)
            if p["impute_val"] is not None:
                imputed[nan_mask] = p["impute_val"]

            # Cap bounds — values in (0, 1) are treated as quantile fractions;
            # values outside that range are used as absolute thresholds.
            valid_for_cap = imputed[np.isfinite(imputed) & ~_is_sentinel(imputed)]
            if cfg.cap_lower is not None:
                if 0.0 < cfg.cap_lower < 1.0:
                    p["cap_lower_val"] = float(np.percentile(valid_for_cap, cfg.cap_lower * 100)) if len(valid_for_cap) else 0.0
                else:
                    p["cap_lower_val"] = float(cfg.cap_lower)
            if cfg.cap_upper is not None:
                if 0.0 < cfg.cap_upper < 1.0:
                    p["cap_upper_val"] = float(np.percentile(valid_for_cap, cfg.cap_upper * 100)) if len(valid_for_cap) else 0.0
                else:
                    p["cap_upper_val"] = float(cfg.cap_upper)

            transformed = self._apply_num_transforms(imputed, cfg, p)

            w_arr = weights.to_numpy() if weights is not None else None
            breaks: Optional[np.ndarray] = None
            if cfg.bin_edges is not None:
                breaks = np.asarray(cfg.bin_edges, dtype=float)
            elif cfg.n_bins is not None and cfg.n_bins > 0:
                full_edges = compute_quantile_bin_edges(transformed, cfg.n_bins, weights=w_arr)
                breaks = full_edges[1:-1]

            if breaks is not None:
                all_labels = make_bin_labels(breaks)
                is_sent_tr = _is_sentinel(transformed)
                ts = pl.Series("_v", transformed).set(pl.Series(is_sent_tr), None)
                binned_labels = ts.cut(
                    list(breaks), labels=all_labels, left_closed=True
                ).cast(pl.Utf8)
                eff_w = w_arr if w_arr is not None else np.ones(len(transformed))
                bin_weights = np.array([
                    eff_w[(binned_labels == label).fill_null(False).to_numpy()].sum()
                    for label in all_labels
                ])
                p["bin_edges"] = breaks
                p["dropped_bin"] = int(bin_weights.argmax())
                p["bin_labels"] = all_labels
                p["has_sentinel_bin"] = bool(np.any(is_sent_tr))

            if cfg.standardize and breaks is None:
                valid_t = transformed[np.isfinite(transformed)]
                p["std_mean"] = float(valid_t.mean()) if len(valid_t) else 0.0
                p["std_std"] = max(float(valid_t.std()), 1e-10)

        return p

    @staticmethod
    def _detect_categorical(s: pl.Series, cfg: VariableConfig) -> bool:
        if cfg.is_categorical is not None:
            return cfg.is_categorical
        return _is_str_or_cat(s.dtype)

    @staticmethod
    def _to_float_array(s: pl.Series) -> np.ndarray:
        """Cast to float64 numpy array, converting nulls to sentinel value."""
        return s.cast(pl.Float64, strict=False).fill_null(MISSING_SENTINEL).to_numpy(allow_copy=True)

    @staticmethod
    def _fit_num_impute(arr: np.ndarray, cfg: VariableConfig) -> Optional[float]:
        # Exclude sentinel when computing impute value
        valid = arr[np.isfinite(arr) & ~_is_sentinel(arr)]
        strat = cfg.impute_strategy
        if strat is None:
            return None
        if strat == "median":
            return float(np.median(valid)) if len(valid) else 0.0
        if strat == "mean":
            return float(np.mean(valid)) if len(valid) else 0.0
        if strat == "most_frequent":
            vals, counts = np.unique(valid, return_counts=True)
            return float(vals[counts.argmax()]) if len(vals) else 0.0
        if strat == "constant":
            return float(cfg.impute_value) if cfg.impute_value is not None else 0.0
        return None

    @staticmethod
    def _fit_cat_impute(s: pl.Series, cfg: VariableConfig) -> Optional[str]:
        strat = cfg.impute_strategy or "most_frequent"
        if strat == "most_frequent":
            modes = s.drop_nulls().mode()
            return modes.to_list()[0] if len(modes) else None
        if strat == "constant" and cfg.impute_value is not None:
            return str(cfg.impute_value)
        return None

    @staticmethod
    def _max_weight_category(
        s: pl.Series,
        cats: List[str],
        weights: Optional[pl.Series],
    ) -> str:
        """
        Return the category with the highest total weight.
        Falls back to the first alphabetical category when *weights* is None.
        """
        if weights is None or len(cats) == 0:
            return cats[0] if len(cats) > 0 else ""
        s_str = s.cast(pl.Utf8).fill_null(_CAT_MISSING).to_list()
        cat_set = set(cats.to_list())

        temp = pl.DataFrame({'vals': s_str, 'weights': weights})
        temp_agg = (
            temp.group_by(pl.col('vals')).sum()
            .filter(pl.col('vals').is_in(list(cat_set)))
            .sort('weights', descending=True)
        )
        return temp_agg['vals'][0] if len(temp_agg) > 0 else (cats[0] if len(cats) > 0 else "")

    @staticmethod
    def _apply_num_transforms(
        arr: np.ndarray, cfg: VariableConfig, p: Dict[str, Any]
    ) -> np.ndarray:
        """Apply cap → log (no standardisation, ignores sentinel)."""
        out = arr.copy()
        is_sent = _is_sentinel(out)
        if "cap_lower_val" in p:
            out = np.where(is_sent, out, np.maximum(out, p["cap_lower_val"]))
        if "cap_upper_val" in p:
            out = np.where(is_sent, out, np.minimum(out, p["cap_upper_val"]))
        if cfg.log_transform:
            out = np.where(is_sent, out, np.log1p(out))
        return out

    # ── Transformation ───────────────────────────────────────────────────

    def _transform_col(
        self,
        s: pl.Series,
        cfg: VariableConfig,
        p: Dict[str, Any],
        out: Dict[str, np.ndarray],
    ) -> None:
        if p["is_categorical"]:
            self._transform_cat(s, cfg, p, out)
        else:
            self._transform_num(s, cfg, p, out)

    @staticmethod
    def _normalize_cat_vals(s: pl.Series, impute_val: Optional[str]) -> List[str]:
        """Cast *s* to strings, fill nulls, and apply categorical imputation."""
        vals = s.cast(pl.Utf8, strict=False).fill_null(_CAT_MISSING).to_list()
        if impute_val is not None:
            return [str(impute_val) if str(v) in (_CAT_MISSING, "None") else str(v) for v in vals]
        return [str(v) for v in vals]

    def _transform_cat(
        self,
        s: pl.Series,
        cfg: VariableConfig,
        p: Dict[str, Any],
        out: Dict[str, np.ndarray],
    ) -> None:
        vals = self._normalize_cat_vals(s, p.get("impute_val"))

        if p.get("encoding") == "onehot":
            series = pl.Series(cfg.col, vals)
            dummies = series.to_dummies()
            for cat in p["categories"]:
                col_name = f"{cfg.col}_{cat}"
                if col_name in dummies.columns:
                    out[col_name] = dummies[col_name].cast(pl.Float64).to_numpy()
                else:
                    out[col_name] = np.zeros(len(vals), dtype=float)
        else:
            out[cfg.col] = np.array(vals)

    def _transform_num(
        self,
        s: pl.Series,
        cfg: VariableConfig,
        p: Dict[str, Any],
        out: Dict[str, np.ndarray],
    ) -> None:
        arr = self._to_float_array(s)

        # Identify sentinel values BEFORE imputation (sentinel == original null)
        is_sent = _is_sentinel(arr)

        iv = p.get("impute_val")
        nan_mask = ~np.isfinite(arr) & ~is_sent
        if iv is not None:
            arr = arr.copy()
            arr[nan_mask] = iv
            arr[is_sent] = iv        # original nulls imputed like NaNs
            is_sent = np.zeros(len(arr), dtype=bool)  # no sentinels remain

        # Apply cap/log transforms (skips sentinel positions)
        arr_t = self._apply_num_transforms(arr, cfg, p)

        if "bin_edges" in p:
            breaks = np.asarray(p["bin_edges"])
            dropped_bin = p.get("dropped_bin", 0)
            all_labels = p.get("bin_labels") or make_bin_labels(breaks)
            is_sent_b = _is_sentinel(arr_t)
            s_cut = pl.Series(cfg.col, arr_t).set(pl.Series(is_sent_b), None)
            labeled = s_cut.cut(
                list(breaks), labels=all_labels, left_closed=True
            ).cast(pl.Utf8)

            # Missing dummy (always present when binning to ensure consistent schema)
            out[f"{cfg.col}_missing"] = is_sent_b.astype(float)

            dummies = labeled.to_dummies()
            for i, label in enumerate(all_labels):
                if i == dropped_bin:
                    continue
                col_name = f"{cfg.col}_{label}"
                if col_name in dummies.columns:
                    out[col_name] = dummies[col_name].cast(pl.Float64).to_numpy()
                else:
                    out[col_name] = np.zeros(len(arr_t), dtype=float)
        else:
            if cfg.standardize and "std_mean" in p:
                arr_t = (arr_t - p["std_mean"]) / p["std_std"]
            out[cfg.col] = arr_t
            for d in range(2, cfg.degree + 1):
                out[f"{cfg.col}^{d}"] = arr_t ** d

    # ── Feature names ────────────────────────────────────────────────────

    def _compute_feature_names(self) -> None:
        names: List[str] = []
        for col, cfg in self.configs.items():
            p = self._params.get(col, {})
            if p.get("is_categorical"):
                if p.get("encoding") == "onehot":
                    names.extend(f"{col}_{c}" for c in p.get("categories", []))
                else:
                    names.append(col)
            elif "bin_edges" in p:
                dropped_bin = p.get("dropped_bin", 0)
                all_labels = p.get("bin_labels") or make_bin_labels(np.array(p["bin_edges"]))
                names.append(f"{col}_missing")
                for i, label in enumerate(all_labels):
                    if i != dropped_bin:
                        names.append(f"{col}_{label}")
            else:
                names.append(col)
                for d in range(2, cfg.degree + 1):
                    names.append(f"{col}^{d}")
        self.feature_names_ = names

    def get_bin_labels(self, col: str, s: pl.Series) -> pl.Series:
        """Return human-readable bin-interval strings for a binned variable."""
        p = self._params.get(col, {})
        if "bin_edges" not in p:
            raise ValueError(f"'{col}' has no bin edges.")
        cfg = self.configs[col]
        arr = self._to_float_array(s)
        is_sent = _is_sentinel(arr)
        arr_t = self._apply_num_transforms(arr, cfg, p)
        breaks = np.asarray(p["bin_edges"])
        all_labels = p.get("bin_labels") or make_bin_labels(breaks)
        s_cut = pl.Series(col, arr_t).set(pl.Series(is_sent), None)
        labeled = s_cut.cut(
            list(breaks), labels=all_labels, left_closed=True
        ).cast(pl.Utf8).fill_null("Missing")
        return labeled.rename(col + "_label")

    def get_level_labels(self, col: str, X: pl.DataFrame) -> pl.Series:
        """
        Return display label strings for *col* using fitted preprocessing params.

        Handles binned numeric, categorical (with optional custom remap), and
        multi-input derived variables.  Call this from plotting code instead of
        ``get_bin_labels`` when the variable type is not known in advance.

        Returns a pl.Series of strings aligned with *X*.
        """
        cfg = self.configs.get(col)
        if cfg is None:
            raise ValueError(f"'{col}' is not in preprocessor configs.")
        p = self._params.get(col, {})
        raw = self._resolve_raw_series(X, cfg)   # handles multi-input

        if "bin_edges" in p:
            return self.get_bin_labels(col, raw)

        if p.get("is_categorical"):
            vals = self._normalize_cat_vals(raw, p.get("impute_val"))
            vals = ["Missing" if v == _CAT_MISSING else v for v in vals]
            return pl.Series(col + "_label", vals)

        # Continuous non-binned — return raw values as strings
        return raw.cast(pl.Utf8).fill_null("Missing").rename(col + "_label")
