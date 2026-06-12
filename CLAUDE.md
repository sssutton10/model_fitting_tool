# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project uses a local venv at `.venv` managed by `uv sync`. Use `uv run python` for all invocations:

```bash
# Install / sync dependencies (creates/updates .venv and uv.lock)
uv sync
uv sync --extra full   # include optional extras: lightgbm, shap, openpyxl, optbinning

# Full suite (all green as of 2026-06: 285 passed, 39 skipped without --extra full)
uv run python -m pytest tests/ -q

# Fast subset (no glum required — covers variable, metrics, bin_suggestor, discovery, bootstrap)
uv run python -m pytest tests/test_variable.py tests/test_metrics.py tests/test_bin_suggestor.py tests/test_discovery.py tests/test_bootstrap.py -q

# Single test by name
uv run python -m pytest tests/test_variable.py::TestGetBinLabels::test_labels_have_letter_prefix -v

# Launch the GUI (serves on port 5006, auto-opens browser)
uv run python run_gui.py
```

The test suite has **no known failures**. The previous Windows-era known-failure list (TestNumericTransforms, TestOptBin DLL crash, TestRelativitiesTable, TestExcelVersion) is resolved: those were environment artifacts or have since been fixed — all pass on the current Linux setup, including optbinning.

**Git repository** — remote: `https://github.com/sssutton10/model_fitting_tool.git`, branch `main`.

## Architecture

`ModelingTool` in `tool.py` is the user-facing orchestration class. It delegates everything to the other modules:

- **`variable.py`** — `VariableConfig` (dataclass config per variable) + `Preprocessor` (fit/transform pipeline). The only stateful object users need to understand.
- **`model.py`** — Wraps `glum` for elastic net GLMs. Produces `ModelVersion` (fitted weights model) or `FactorModelVersion` (Excel-loaded factor table).
- **`plots.py`** — All matplotlib charts. Each function accepts a raw `pl.DataFrame` plus an optional `preprocessor=` argument; when the preprocessor is provided, `_resolve_level` uses fitted bin edges / category labels instead of re-binning on the fly. Advanced analytics plots: `interaction_heatmap`, `pd_plot_2d`, `importance_plot`, `residual_heatmap`, `regularization_path_plot`, `overfitting_plot`, `bootstrap_ci_plot`, `relativities_ci_plot`, `cv_stability_plot`.
  - Reusable private helpers: `_resolve_level(col, X, preprocessor, n_bins)` returns level labels for any variable; `_sort_labels(labels)` sorts bin/category labels naturally.
- **`metrics.py`** — Stateless: Gini, lift tables, double lift, compare metrics, `vif_table` (multicollinearity), `bootstrap_metrics` (CI on metrics). No dependencies on other package modules except numpy/polars.
- **`bin_suggestor.py`** — Stateless breakpoint suggestion (quantile, equal-width, optbinning, GBM). Never modifies `VariableConfig`; only returns suggested break lists. The optbin method auto-selects `ContinuousOptimalBinning` for continuous targets (≥ 3 unique y values) vs `OptimalBinning` for binary.
- **`discovery.py`** — Shadow GBM diagnostics: `fit_shadow_gbm`, `permutation_importance`, `interaction_ranking` (Friedman H-statistic), `partial_dependence_2d`, `residual_gbm`, SHAP methods, Boruta, monotonicity, category grouping. Categoricals are one-hot encoded automatically; importance is reported per original variable, not per dummy. All functions are also exposed as `ModelingTool` methods. `lightgbm` / `shap` are lazily imported with pip-hint ImportErrors.
- **`io_utils.py`** — Pickle-based `save_version` / `load_version`.
- **`gui.py`** — Panel-based GUI (entry point: `run_gui.py`). Six tabs, one `param.Parameterized` class per tab sharing a `ModelingApp` state object: Data (incl. Preview/Explore sub-tabs — value counts + summary stats per column), Variables, Model, Evaluation (incl. A/v/E, CV stability, Excel workflows, save/load), Diagnostics (VIF, residual heatmap, regularization path, overfitting monitor, bootstrap CIs), Discovery (shadow GBM + dependent diagnostics, gated buttons). Tabs mutate `app.tool` then call `app.bump()`; other tabs watch `tool_version`. Tests: `tests/test_gui.py` drives `_on_*` callbacks headlessly (no browser/server needed).

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

All custom transforms use a **unified DataFrame-based signature** (the `VariableConfig` docstring and `__init__.py` quick-start now document this correctly):

```python
def my_transform(df: pl.DataFrame, **kwargs) -> array-like:
    ...
```

- `df` contains only the relevant columns: `[cfg.col]` for single-column, `cfg.input_cols` for multi-input.
- Applied once in `_resolve_raw_series`, before any cap / log / binning.
- For categorical remapping, the output list/array of strings becomes the category labels.

### glum dependency

`model.py` hard-imports glum at the top level. `conftest.py` inserts a `MagicMock` into `sys.modules["glum"]` before importing `elastic_net_tool` so that the five glum-free test modules can run without glum installed. Tests requiring a real glum are marked `@pytest.mark.requires_glum` and auto-skipped when the mock is active.

### Gotchas

- **`ModelVersion.train_predictions`**, not `.predictions` — all model versions (including `FactorModelVersion`) store predictions in `train_predictions`. There is no `.predictions` attribute.
- **Derived variables with `input_cols`** — when creating a variable from a different source column (e.g. "region" from "state"), use `input_cols=["state"]` not `col="state"`. The first positional arg to `add_variable()` is always the output variable name AND the `col` param. Categorical detection is automatic from the transform's output dtype (string → categorical, numeric → continuous); `is_categorical=True` is only needed to force categorical treatment when the output dtype is numeric.
- **`show=False` does not mean "returns a figure"** — several `ModelingTool` methods diverge from the plots-return-figures convention:
  - `compare_models(show=False)` still *creates* the double-lift figure internally and discards it (returns only dataframes). The GUI captures it by diffing `plt.get_fignums()` before/after; any other caller must do the same or leak figures.
  - `regularization_path`, `overfitting_monitor`, `bootstrap_metrics`, `bootstrap_relativities` with `show=False` return DataFrames only — build figures via the matching `plots.py` helpers (`regularization_path_plot`, `overfitting_plot`, `bootstrap_ci_plot`, `relativities_ci_plot`).
  - `residual_heatmap` returns a `(fig, df)` tuple.
  - `fit_cv_stability(plot=True)` creates and discards its figure — call with `plot=False` and use `plots.cv_stability_plot(df)` instead.
  - `suggest_bins(show_plot=True)` likewise discards the overlay figure — use `bin_suggestor._plot_suggestions(col, X, splits_dict, weights=...)` directly.
- **`permutation_importance(version=...)`** raises `NotImplementedError` for any non-None version — always call with `version=None` (shadow-GBM path).
- **Interaction-ranking score columns differ per method** — `interaction_ranking` → `h_statistic`, `shap_interaction_ranking` → `interaction_strength`, `tree_interaction_cooccurrence` → `cooccurrence_score`. `plots.interaction_heatmap` hardcodes `h_statistic`; rename the column before reuse.
- **`np.percentile` with weights** — requires `method="inverted_cdf"` on numpy ≥ 2.0. The fix is already applied in `compute_quantile_bin_edges`.
- **numpy / matplotlib compat shims** — `metrics.py` uses a `_trapezoid` alias (`np.trapezoid` on numpy ≥ 2.0, `np.trapz` fallback); `plots.cv_stability_plot` deliberately avoids `boxplot(labels=)` (removed in matplotlib 3.11). Don't reintroduce the removed APIs.
