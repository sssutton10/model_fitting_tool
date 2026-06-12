# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Activate the environment first in any new shell:
```bash
source /c/Users/sssut/anaconda3/etc/profile.d/conda.sh && conda activate base
```

Run tests:
```bash
# Full suite
python -m pytest tests/ -q

# Fast subset (no glum required — covers variable, metrics, bin_suggestor, discovery, bootstrap)
python -m pytest tests/test_variable.py tests/test_metrics.py tests/test_bin_suggestor.py tests/test_discovery.py tests/test_bootstrap.py -q

# Single test file
python -m pytest tests/test_variable.py -q

# Single test by name
python -m pytest tests/test_variable.py::TestGetBinLabels::test_labels_have_letter_prefix -v
```

**Known pre-existing failures (6, do not fix without asking):** `test_numeric_has_cap_upper`, `test_cap_lower_clips_low_values`, `test_log_transform_applies_log1p`, `test_impute_median_uses_correct_value`, `test_impute_mean_fills_nulls`, `test_impute_constant_fills_nulls` — all in `TestNumericTransforms` / `TestDefaultConfig`.

**Other known pre-existing issues (do not fix without asking):**
- `test_bin_suggestor.py::TestOptBin` — 5 tests crash due to `optbinning` / `ortools` DLL load failure (Windows `0xc0000139`). Exclude with `-k "not TestOptBin"`.
- `test_tool.py::TestRelativitiesTable` — 15 tests fail due to numpy truth-value ambiguity on `_weights_array or np.ones()` at `tool.py:974` (fixed to use `if ... is not None else`).
- `test_tool.py::TestExcelVersion` — 12 tests fail due to polars `SchemaError` (join key type `str` vs `null`). Exclude with `-k "not TestExcelVersion"`.

**Git repository** — remote: `https://github.com/sssutton10/model_fitting_tool.git`, branch `main`.

## Architecture

`ModelingTool` in `tool.py` is the user-facing orchestration class. It delegates everything to the other modules:

- **`variable.py`** — `VariableConfig` (dataclass config per variable) + `Preprocessor` (fit/transform pipeline). The only stateful object users need to understand.
- **`model.py`** — Wraps `glum` for elastic net GLMs. Produces `ModelVersion` (fitted weights model) or `FactorModelVersion` (Excel-loaded factor table).
- **`plots.py`** — All matplotlib charts. Each function accepts a raw `pl.DataFrame` plus an optional `preprocessor=` argument; when the preprocessor is provided, `_resolve_level` uses fitted bin edges / category labels instead of re-binning on the fly. Advanced analytics plots: `interaction_heatmap`, `pd_plot_2d`, `importance_plot`, `residual_heatmap`, `regularization_path_plot`, `overfitting_plot`, `bootstrap_ci_plot`, `relativities_ci_plot`.
  - Reusable private helpers: `_resolve_level(col, X, preprocessor, n_bins)` returns level labels for any variable; `_sort_labels(labels)` sorts bin/category labels naturally.
- **`metrics.py`** — Stateless: Gini, lift tables, double lift, compare metrics, `vif_table` (multicollinearity), `bootstrap_metrics` (CI on metrics). No dependencies on other package modules except numpy/polars.
- **`bin_suggestor.py`** — Stateless breakpoint suggestion (quantile, equal-width, optbinning, GBM). Never modifies `VariableConfig`; only returns suggested break lists.
- **`discovery.py`** — Shadow GBM diagnostics: `fit_shadow_gbm`, `permutation_importance`, `interaction_ranking` (Friedman H-statistic), `partial_dependence_2d`, `residual_gbm`. Categoricals are one-hot encoded automatically; importance is reported per original variable, not per dummy. All functions are also exposed as `ModelingTool` methods.
- **`io_utils.py`** — Pickle-based `save_version` / `load_version`.

### `Preprocessor` internals

`Preprocessor.fit(X, weights)` calls `_fit_col` per variable and stores results in `_params: Dict[str, Dict]`. `Preprocessor.transform(X)` calls `_transform_num` or `_transform_cat` per column and concatenates results.

Key `_params[col]` keys:
- **Numeric binned:** `bin_edges` (break points, no outer bounds), `bin_labels` (list of strings), `dropped_bin` (int index of heaviest-weight bin), `has_sentinel_bin`
- **Categorical:** `categories` (list without the dropped level), `dropped_category`, `encoding`

### Binning conventions

- `MISSING_SENTINEL = -999_999_999.0` — sentinel for numerically encoded missings. All numeric pipes check `np.isclose(arr, MISSING_SENTINEL)` rather than `np.isnan`.
- `p["bin_edges"]` stores **break points only** (e.g. `[25, 45, 65]`), no outer bounds. Pass directly to `pl.Series.cut(breaks, ...)` — no `[1:-1]` slicing.
- `make_bin_labels(breaks)` produces `n+1` labels for `n` breaks: first bin `A_<hi`, interior bins `B_[lo, hi)`, last bin `C_lo+`.
- The **heaviest-weight level is dropped** as the reference (base) for both binned numeric and categorical variables — same logic, stored in `dropped_bin` / `dropped_category`.

### `custom_transform` API

All custom transforms now use a **unified DataFrame-based signature**:

```python
def my_transform(df: pl.DataFrame, **kwargs) -> array-like:
    ...
```

- `df` contains only the relevant columns: `[cfg.col]` for single-column, `cfg.input_cols` for multi-input.
- Applied once in `_resolve_raw_series`, before any cap / log / binning.
- For categorical remapping, the output list/array of strings becomes the category labels.

> ⚠️ The `VariableConfig` docstring and the `__init__.py` quick-start example still reference the old per-value / per-array signatures — ignore those, the code uses the DataFrame API above.

### glum dependency

`model.py` hard-imports glum at the top level. `conftest.py` inserts a `MagicMock` into `sys.modules["glum"]` before importing `elastic_net_tool` so that the five glum-free test modules can run without glum installed. Tests requiring a real glum are marked `@pytest.mark.requires_glum` and auto-skipped when the mock is active.

### Gotchas

- **`ModelVersion.train_predictions`**, not `.predictions` — all model versions (including `FactorModelVersion`) store predictions in `train_predictions`. There is no `.predictions` attribute.
- **Derived variables with `input_cols`** — when creating a variable from a different source column (e.g. "region" from "state"), use `input_cols=["state"]` not `col="state"`. The first positional arg to `add_variable()` is always the output variable name AND the `col` param. Categorical detection is automatic from the transform's output dtype (string → categorical, numeric → continuous); `is_categorical=True` is only needed to force categorical treatment when the output dtype is numeric.
- **`log_transform=True` and sentinels** — `_apply_num_transforms` checks `np.min(out) > 0` which includes sentinel values (`-999_999_999`). This means `log_transform=True` will always fail on columns with sentinel-encoded missings. Also fails if the column can contain 0.
- **`np.percentile` with weights** — requires `method="inverted_cdf"` on numpy ≥ 2.0. The fix is already applied in `compute_quantile_bin_edges`.
