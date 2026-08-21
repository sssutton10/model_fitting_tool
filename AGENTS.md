# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

Use the UV-managed environment. `uv run` selects the project interpreter, so
do not activate Conda or call a system Python directly.

Set up or refresh the environment:

```bash
uv sync --group dev --extra full
```

Run tests:

```bash
# Full suite
uv run python -m pytest tests/ -q

# Focused module
uv run python -m pytest tests/test_variable.py -q

# Focused set
uv run python -m pytest tests/test_metrics.py tests/test_validation.py tests/test_bootstrap.py -q

# Single test by name
uv run python -m pytest tests/test_variable.py::TestGetBinLabels::test_labels_have_letter_prefix -v
```

Run code-quality checks:

```bash
uv run ruff check elastic_net_tool
uv run ruff format --check elastic_net_tool
```

Ruff targets Python 3.10, enforces a maximum cyclomatic complexity and branch
count of 10, and a maximum of 50 statements per callable. `gui.py` is excluded
from Ruff because it is outside the current refactoring scope.

### Current Test Baseline

As of 2026-08-21, the locked UV environment collects 348 tests and reports
205 passed, 118 failed, and 25 errors. This is the pre-existing baseline for
the current refactor; do not broaden a scoped change to repair these failures
without asking. The main failure groups are:

- seven numeric transformation/default-configuration expectation failures in
  `test_variable.py`
- one `optbinning` compatibility failure in `test_bin_suggestor.py`
- 17 model tests and 12 metrics/validation/bootstrap tests whose expectations
  have drifted from the current APIs or dependency behavior
- 16 discovery tests affected by optional configuration handling
- 65 tool and chained-IO failures, including stale method/parameter expectations
  and current Polars schema behavior
- 25 GUI setup errors because GUI construction passes a constructor argument
  that `ModelingTool` does not accept

For refactors, compare the relevant focused-test result with this baseline and
require both Ruff commands to pass. Do not silently treat unrelated baseline
failures as regressions or fix them as part of cleanup.

Do not run `git` commands in this workspace.

## Architecture

`ModelingTool` in `tool.py` is the user-facing orchestration class. Its public
methods retain workflow and version-management responsibilities, while short
same-file private helpers handle validation, factor-table loading, summary-row
construction, comparison inputs, monitoring, and bootstrap aggregation. Keep
those helpers in `tool.py` unless the user first approves a file split.

Lower-level work is delegated to the other modules:

- **`variable.py`** — `VariableConfig` (dataclass config per variable) + `Preprocessor` (fit/transform pipeline). The only stateful object users need to understand.
- **`model.py`** — Wraps `glum` for elastic net GLMs. Produces `ModelVersion` (fitted weights model) or `FactorModelVersion` (Excel-loaded factor table).
- **`plots.py`** — All matplotlib charts. Each function accepts a raw `pl.DataFrame` plus an optional `preprocessor=` argument; when the preprocessor is provided, `_resolve_level` uses fitted bin edges / category labels instead of re-binning on the fly. Advanced analytics plots: `interaction_heatmap`, `pd_plot_2d`, `importance_plot`, `residual_heatmap`, `regularization_path_plot`, `overfitting_plot`, `bootstrap_ci_plot`, `relativities_ci_plot`.
  - Reusable private helpers: `_resolve_level(col, X, preprocessor, n_bins)` returns level labels for any variable; `_sort_labels(labels)` sorts bin/category labels naturally.
- **`metrics.py`** — Stateless: Gini, lift tables, double lift, compare metrics, `vif_table` (multicollinearity), `bootstrap_metrics` (CI on metrics). No dependencies on other package modules except numpy/polars.
- **`bin_suggestor.py`** — Stateless breakpoint suggestion (quantile, equal-width, optbinning, GBM). Never modifies `VariableConfig`; only returns suggested break lists.
- **`discovery.py`** — Shadow GBM diagnostics: `fit_shadow_gbm`, `permutation_importance`, `interaction_ranking` (Friedman H-statistic), `partial_dependence_2d`, `residual_gbm`. Categoricals are one-hot encoded automatically; importance is reported per original variable, not per dummy. All functions are also exposed as `ModelingTool` methods.
- **`io_utils.py`** — Pickle-based `save_version` / `load_version`.

All non-GUI modules follow the same internal pattern: small private helpers
separate validation, data preparation, computation, and result assembly. Avoid
recombining these stages into long methods or adding abstraction that is used
only once and does not clarify the flow.

### `Preprocessor` internals

`Preprocessor` first materializes raw derived columns recursively. `output_cols`
controls which configs are fitted into `_params` and emitted into the design
matrix; other configs are dependency-only. `transform(X)` rebuilds the raw
dependency graph, then transforms only `output_cols`.

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
- Applied once per fit/transform materialization pass, before any cap / log / binning.
- For categorical remapping, the output list/array of strings becomes the category labels.

The `VariableConfig` docstring and package quick-start use this DataFrame API.

### glum dependency

`model.py` hard-imports glum at the top level. When glum is unavailable,
`conftest.py` inserts a `MagicMock` into `sys.modules["glum"]` before importing
`elastic_net_tool`, allowing independent module tests to load. Tests requiring
a real glum are marked `@pytest.mark.requires_glum` and auto-skipped when the
mock is active. Normal development should use the fully synced UV environment.

### Gotchas

- **`ModelVersion.train_predictions`**, not `.predictions` — all model versions (including `FactorModelVersion`) store predictions in `train_predictions`. There is no `.predictions` attribute.
- **`ModelingTool.fit_model` is side-effect oriented** — it stores the fitted
  version and updates `current_version`. Despite its current return annotation,
  the wrapper returns `None`; the module-level `model.fit_model` returns a
  `ModelVersion`. Preserve that behavior unless an API change is explicitly
  requested.
- **Derived variables with `input_cols`** — when creating a variable from a different source column (e.g. "region" from "state"), use `input_cols=["state"]` not `col="state"`. The first positional arg to `add_variable()` is always the output variable name AND the `col` param. Categorical detection is automatic from the transform's output dtype (string → categorical, numeric → continuous); `is_categorical=True` is only needed to force categorical treatment when the output dtype is numeric.
- **Chained derived variables** — `input_cols` may name other registered derived variables. Downstream transforms receive the upstream raw custom-transform result, before the upstream cap/log/bin/encoding pipeline. Only variables explicitly requested by the model are emitted as predictors.
- **`log_transform=True` and sentinels** — `_apply_num_transforms` checks `np.min(out) > 0` which includes sentinel values (`-999_999_999`). This means `log_transform=True` will always fail on columns with sentinel-encoded missings. Also fails if the column can contain 0.
- **`np.percentile` with weights** — requires `method="inverted_cdf"` on numpy ≥ 2.0. The fix is already applied in `compute_quantile_bin_edges`.
