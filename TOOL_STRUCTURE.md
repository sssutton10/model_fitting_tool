# Elastic Net Tool Structure

This document explains how the project is organized and how the main pieces fit
together. It is written for someone who has not seen the code before and wants
to understand the tool well enough to read it, use it, and extend it safely.

The project provides an insurance-focused modeling workflow around elastic net
GLMs. It uses `polars` DataFrames for tabular data, `glum` for GLM fitting, and
matplotlib-based plots for diagnostics. The main user-facing object is
`ModelingTool`; most other modules implement one part of the workflow and are
called by that facade.

## Big Picture

The package is organized around this modeling flow:

1. Load a `polars.DataFrame` containing the target, optional exposure weights,
   optional offsets, optional CV fold labels, and raw predictor columns.
2. Create a `ModelingTool` with the dataset and column names.
3. Register variable preprocessing rules with `add_variable`.
4. Explore individual variables and candidate breakpoints.
5. Fit one or more named model versions.
6. Inspect coefficients, factor relativities, lift, A/E charts, residuals, and
   model comparisons.
7. Optionally run discovery diagnostics such as GBM importance, SHAP importance,
   interaction ranking, category grouping, or residual models.
8. Save fitted versions or load a GLM or Excel factor-table version later.

The important architectural idea is that `ModelingTool` coordinates the
workflow, but it does not do most of the low-level work itself. It delegates:

- variable preprocessing to `variable.py`
- GLM fitting and model containers to `model.py`
- charts to `plots.py`
- metrics to `metrics.py`
- bin suggestions to `bin_suggestor.py`
- GBM/SHAP/discovery diagnostics to `discovery.py`
- persistence to `io_utils.py`

## Repository Layout

```text
elastic_net_tool/
  __init__.py        Public package exports and quick-start example.
  tool.py            ModelingTool facade and user workflow orchestration.
  variable.py        VariableConfig and Preprocessor.
  model.py           glum model fitting plus ModelVersion and FactorModelVersion.
  metrics.py         Stateless model metrics and lift/double-lift tables.
  plots.py           Matplotlib plots used by ModelingTool.
  bin_suggestor.py   Breakpoint suggestion methods for numeric variables.
  discovery.py       LightGBM/SHAP diagnostics and feature discovery helpers.
  io_utils.py        Save/load helpers for fitted versions.
  gui.py             Panel GUI support.

example_usage.py     End-to-end synthetic insurance modeling example.
run_gui.py           GUI entry point.
tests/               Unit tests for preprocessing, metrics, discovery, IO, etc.
pyproject.toml       Package metadata and dependencies.
```

`example_usage.py` is the best runnable walkthrough. This document is meant to
explain the structure behind that example.

## The Main Facade: `ModelingTool`

`ModelingTool` lives in `elastic_net_tool/tool.py`. It is the class most users
interact with directly.

Typical construction:

```python
import polars as pl
from elastic_net_tool import ModelingTool

tool = ModelingTool(
    data=df,
    target_col="loss_ratio",
    weight_col="earned_premium",
    offset_col=None,
    cv_column="cv_fold",
)
```

The constructor stores the dataset and the key column names. It also initializes
two important registries:

- `variable_configs`: a dictionary from variable name to `VariableConfig`
- `model_versions`: a dictionary from version name to fitted model version

The tool keeps all model versions in memory by name. That allows common
workflows such as fitting `"baseline"` and `"with_geo"`, comparing them, and
then choosing one as the current or base version.

### Important Constructor Arguments

`data` must be a `polars.DataFrame`. The rest of the tool assumes polars input.

`target_col` names the modeled response, usually a loss ratio.

`weight_col` is optional but important for insurance modeling. When present, it
is used as exposure/sample weight in model fitting, plots, metrics, and reference
level selection.

`offset_col` is optional. When present, offset values are passed through to
`glum` during fitting and prediction.

`cv_column` is optional. When present, `fit_model` can automatically build a
`sklearn.model_selection.PredefinedSplit` from the column. This is useful when
fold assignments must be stable or come from an external split.

`drop_reference` controls how one-hot reference levels are chosen:

- `"max_weight"` drops the level with the highest total exposure weight.
- `"first"` drops the first sorted level.

The default is `"max_weight"` because the most common exposure group is often
the clearest reference level for interpreting rating relativities.

## Modeling Workflow Through `ModelingTool`

### 1. Register Variables

Variables are registered with `tool.add_variable(...)`. This does not fit a
model. It only records how each variable should be transformed when a model is
fit.

Examples:

```python
tool.add_variable("vehicle_age", cap_upper=0.99)
tool.add_variable("driver_age", breakpoints=[25, 45, 65])
tool.add_variable("state", encoding="onehot")
```

`breakpoints` is a user-friendly alias for `bin_edges`.

If a variable is used in `fit_model` but has not been registered, the tool tries
to infer a default config from the raw column dtype. String-like columns become
categorical. Numeric columns are treated as continuous.

Derived variables are supported by giving the output variable name as the first
argument and source columns through `input_cols`.

```python
def veh_age_x_driver_age(df: pl.DataFrame):
    return df["vehicle_age"].to_numpy() * df["driver_age"].to_numpy()

tool.add_variable(
    "veh_x_age",
    input_cols=["vehicle_age", "driver_age"],
    custom_transform=veh_age_x_driver_age,
    cap_upper=0.99,
)
```

The first positional argument is always the output variable name. For a derived
variable like `"region"` created from `"state"`, use `input_cols=["state"]`
rather than passing `"state"` as the first argument.

### 2. Explore Variables and Breakpoints

`ModelingTool` exposes univariate plotting and several breakpoint suggestion
methods:

- `univariate_plot`
- `suggest_bins_quantile`
- `suggest_bins_equal_width`
- `suggest_bins_gbm`
- `suggest_bins_optbin`
- `suggest_bins`

The individual `suggest_bins_*` methods return breakpoint lists. The combined
`suggest_bins` method can run several strategies and optionally plot the
candidate breaks together.

```python
splits = tool.suggest_bins(
    "annual_mileage",
    methods=["quantile", "equal_width", "gbm"],
    n_bins=6,
    max_splits=8,
    show_plot=True,
)

tool.add_variable("annual_mileage", breakpoints=splits["gbm"])
```

These methods are exploratory. They do not mutate `VariableConfig` objects.

### 3. Fit a Named Model Version

`fit_model` fits a GLM and stores the result as a named version.

```python
tool.fit_model(
    variables=["vehicle_age", "driver_age", "state"],
    version="baseline",
    l1_ratio=[0.1, 0.5, 0.9],
    use_cv=True,
)
```

The method builds or uses a fitted `Preprocessor`, transforms the raw data into
a numeric design matrix, calls the fitting code in `model.py`, stores the result
in `tool.model_versions`, and records the version as current.

The `ModelingTool` wrapper is normally used for its side effect of storing the
named version. Retrieve fitted versions later through `tool.model_versions`,
`tool._get_version(...)`, or the public inspection/prediction methods.

When `alpha` is supplied, `use_cv` is turned off and the model is fit with that
fixed regularization strength. Passing `alpha=0.0` fits an unpenalized GLM.

When `cv_column` was supplied to the constructor and no explicit `cv` argument
is passed, the tool uses a `PredefinedSplit` based on the fold column. Passing
`cv=<int>` overrides that for one fit.

### 4. Inspect Versions

Common version inspection methods include:

- `list_versions`
- `model_summary`
- `summary_table`
- `coefficient_plot`
- `predict`
- `set_base_version`

A fitted GLM version is represented by `ModelVersion`. An Excel factor-table
version is represented by `FactorModelVersion`. Both expose
`train_predictions`; there is no `.predictions` attribute.

### 5. Evaluate and Compare Models

`ModelingTool` wraps evaluation methods from `plots.py` and `metrics.py`.

Common methods include:

- `ae_chart`
- `ave_table`
- `decile_lift_chart`
- `residual_chart`
- `plot_all_variables`
- `compare_models`
- `overfitting_monitor`
- `vif_table`
- `bootstrap_metrics`
- `bootstrap_relativities`

`compare_models` is built around standard metrics and double-lift comparisons.
Double lift sorts observations by the ratio between two models and shows which
model is closer to actuals across the ratio-sorted buckets.

### 6. Run Discovery Diagnostics

The discovery methods use tree models or SHAP to identify nonlinear signal,
interactions, variable importance, and residual structure.

Methods exposed on `ModelingTool` include:

- `fit_shadow_gbm`
- `permutation_importance`
- `interaction_ranking`
- `partial_dependence_2d`
- `residual_gbm`
- `shap_importance`
- `shap_dependence`
- `shap_interaction_ranking`
- `tree_interaction_cooccurrence`
- `suggest_category_groups`
- `monotonicity_test`
- `boruta_select`
- `residual_heatmap`
- `regularization_path`

These are not required to fit a GLM. They are supporting analysis tools for
understanding candidate predictors, missed signal, and interactions.

### 7. Save and Load

`tool.save(...)` and `ModelingTool.load(...)` delegate to `io_utils.py`.

The persistence layer saves a snapshot of a fitted model version plus enough
tool metadata to rebuild a usable `ModelingTool` later. It also contains custom
transform serialization support. Named functions are safer than lambdas because
they can be inspected and reconstructed more reliably.

## Variable Configuration: `VariableConfig`

`VariableConfig` is defined in `elastic_net_tool/variable.py`. It is a dataclass
describing how one model variable should be prepared.

Important fields:

| Field | Meaning |
| --- | --- |
| `col` | Output variable name. Also the source column when `input_cols` is not set. |
| `input_cols` | Source columns for a derived variable. |
| `custom_transform` | Optional function applied before capping, logging, binning, or encoding. |
| `transform_kwargs` | Keyword arguments passed to `custom_transform`. |
| `cap_lower`, `cap_upper` | Numeric cap bounds. |
| `log_transform` | Apply a log transform after capping. |
| `impute_strategy` | Numeric or categorical imputation strategy. |
| `impute_value` | Constant fill value when using constant imputation. |
| `n_bins` | Number of quantile bins for numeric binning. |
| `bin_edges` | Explicit numeric breakpoints. Takes precedence over `n_bins`. |
| `standardize` | Standardize continuous, unbinned numeric values. |
| `degree` | Add polynomial powers for continuous, unbinned numeric values. |
| `encoding` | Categorical encoding. `"auto"` and `"onehot"` produce dummies. |
| `is_categorical` | Force categorical or numeric treatment instead of dtype inference. |
| `right_closed` | Controls interval closure when cutting numeric bins. |

### Custom Transform Signature

The current code calls custom transforms with a DataFrame-based API:

```python
def my_transform(df: pl.DataFrame, **kwargs) -> pl.Series:
    ...
```

For a single-column variable, `df` contains one column. For a derived variable,
`df` contains `input_cols` in the order configured.

The function should return one value per input row as a one-dimensional,
row-aligned series. A `pl.Series` is the clearest return type:

```python
def mileage_to_thousands(df: pl.DataFrame) -> pl.Series:
    return pl.Series("annual_mileage", df["annual_mileage"].to_numpy() / 1_000)
```

The implementation wraps the transform result with `pl.Series(cfg.col, result)`,
so list-like or numpy-array returns also work, but conceptually the return value
is the raw series for the output variable.

The transform is applied once in `_resolve_raw_series`, before any other
preprocessing. If the returned series contains strings, the variable can be
treated as categorical. Set `is_categorical=True` when you need to force
categorical handling, especially if the returned categories are numeric codes.

### Defaults

`default_config` inspects the polars dtype:

- string-like columns become categorical, use most-frequent imputation, and
  default to automatic/one-hot encoding
- numeric columns become continuous numeric variables

Defaults are convenient, but explicit configs are clearer for production models
because they document important modeling choices.

## Preprocessing: `Preprocessor`

`Preprocessor` is the stateful object that turns raw columns into a fitted design
matrix. It also records the learned transformation parameters needed to score
new data consistently.

It has three main public methods:

- `fit(X, y=None, weights=None)`
- `transform(X)`
- `fit_transform(X, y=None)`

After fitting, learned details are stored in `preprocessor._params`, keyed by
variable name. The feature order used by the model is stored in
`preprocessor.feature_names_`.

### Fit-Time Responsibilities

During `fit`, each configured variable goes through these steps:

1. Resolve the raw series.
2. Apply `custom_transform` if present.
3. Decide whether the result is categorical or numeric.
4. For categorical variables, learn imputation and reference level.
5. For numeric variables, learn imputation, caps, log transform behavior,
   bin edges or standardization parameters.
6. Build feature names in the exact order that `transform` will output.

### Transform-Time Responsibilities

During `transform`, the preprocessor applies the fitted parameters to any
compatible DataFrame and returns a numeric `polars.DataFrame` design matrix.
For model prediction, that matrix is converted to a float numpy array before
calling `glum`.

This fit/transform split is crucial. Training data determines bins, reference
levels, and standardization parameters. New scoring data reuses those values so
that the model sees the same schema.

### Numeric Variables

Numeric processing supports:

- null conversion to the project missing sentinel
- optional imputation
- optional capping
- optional log transform
- optional binning
- optional standardization
- optional polynomial expansion

The missing sentinel is:

```python
MISSING_SENTINEL = -999_999_999.0
```

The code checks sentinel values with `np.isclose`, not only exact equality.

For unbinned numeric variables, the output feature is usually just `col`.
If `degree > 1`, additional columns are emitted as `col^2`, `col^3`, and so on.

For binned numeric variables, the output is one dummy per retained bin, plus a
missing dummy if sentinel values remain after the fit-time numeric preprocessing
steps. One bin is dropped as the reference level.

### Numeric Binning Details

`bin_edges` stores internal breakpoints only. It does not include lower or upper
outer bounds.

For example:

```python
tool.add_variable("driver_age", breakpoints=[25, 45, 65])
```

creates four conceptual bins:

- values below 25
- values from 25 to 45
- values from 45 to 65
- values above 65

The exact displayed label text is generated by `make_bin_labels` using fitted
minimum and maximum values. Those labels are also used in dummy feature names
and later in summary tables and plots.

When `n_bins` is used instead of `bin_edges`, the preprocessor computes weighted
quantile edges during fit and stores only the internal breaks.

### Reference Level Selection

For binned numeric variables and one-hot categorical variables, one level is
dropped as the reference. With the default `drop_reference="max_weight"`, the
reference is the level with the highest total exposure weight. Without weights,
or with `drop_reference="first"`, the first sorted level is used.

This means a coefficient should be interpreted relative to the dropped level for
that variable.

### Categorical Variables

Categorical processing supports:

- dtype-based detection or `is_categorical=True`
- optional imputation
- one-hot encoding
- weighted reference-level selection
- stable dummy columns at scoring time

When a category appears at scoring time that was not present during fitting, it
does not receive a dummy column unless it was part of the fitted categories.
This keeps the model matrix schema stable.

## Model Fitting: `model.py`

`elastic_net_tool/model.py` contains the lower-level modeling functions and
model container dataclasses.

### `ModelVersion`

`ModelVersion` represents a fitted GLM version. Important attributes:

| Attribute | Meaning |
| --- | --- |
| `name` | Version name supplied to `fit_model`. |
| `variables` | Raw or derived variables included in the model. |
| `preprocessor` | Fitted `Preprocessor`. |
| `glm` | Fitted `glum` estimator. |
| `feature_names` | Design matrix feature names. |
| `coefficients` | Polars DataFrame with `feature` and `coefficient`. |
| `alpha` | Selected or fixed regularization strength. |
| `l1_ratio` | Selected or fixed elastic-net mixing value. |
| `family`, `link` | GLM family and link used by `glum`. |
| `train_predictions` | Predictions on the training rows. |
| `fit_info` | Metadata such as CV settings and fit timestamp. |
| `cv_stability` | Optional fold-stability table. |

`ModelVersion.predict(X, offset=None)` transforms `X` through the stored
preprocessor and calls the fitted `glum` model.

### `fit_model`

The module-level `fit_model` function does the actual GLM fitting:

1. Resolve the GLM family and link.
2. Build a `Preprocessor` if one was not supplied.
3. Fit the preprocessor if needed.
4. Transform the raw data into a numeric matrix.
5. Fit either `GeneralizedLinearRegressorCV` or `GeneralizedLinearRegressor`.
6. Extract coefficients.
7. Generate training predictions.
8. Return a `ModelVersion`.

If cross-validation is enabled, `glum.GeneralizedLinearRegressorCV` selects
`alpha` and possibly `l1_ratio`. The code then fits a final
`GeneralizedLinearRegressor` with the selected hyperparameters.

### `fit_cv_stability`

`fit_cv_stability` fits the same model repeatedly across user-defined folds. For
each unique fold value, that fold is held out and the model is trained on the
remaining rows. The returned table contains one row per fold plus summary rows:

- `geomean`
- `std`
- `cv_pct`

The preprocessor is fitted once on the full dataset so feature names remain
consistent across fold fits.

### `FactorModelVersion`

`FactorModelVersion` is for Excel-loaded factor tables rather than fitted GLMs.
It stores a factor table with columns like `Variable`, `Level`, and `Factor`.
Prediction multiplies together the factor for each variable level and applies an
intercept factor if present.

If a fitted preprocessor is available for some variables, the factor model uses
the same bin and category label resolution as the GLM workflow. Otherwise it
falls back to direct string matching on raw values.

## Metrics: `metrics.py`

`metrics.py` is intentionally stateless. It contains functions that operate on
arrays and return numbers or `polars.DataFrame` summaries.

Important functions:

- `gini_coefficient`
- `lift_table`
- `double_lift_table`
- `double_lift_score`
- `compute_metrics`
- `compare_metrics`
- `vif_table`
- `bootstrap_metrics`

`compute_metrics` returns MSE, RMSE, MAE, raw Gini, normalized Gini, lift range,
and lift RMSE.

`lift_table` creates equal-weight buckets sorted by model prediction.

`double_lift_table` compares two models by sorting on the ratio between their
predictions. `double_lift_score` summarizes which model is closer to actuals
across those buckets. Negative scores favor model 1. Positive scores favor
model 2.

`vif_table` is used to inspect multicollinearity in a design matrix.

`bootstrap_metrics` repeatedly resamples the data to estimate confidence
intervals for model metrics.

## Plots: `plots.py`

`plots.py` contains matplotlib plotting functions. Most accept raw data and
optional preprocessor context.

Important functions:

- `univariate_plot`
- `ae_chart`
- `residual_chart`
- `double_lift_chart`
- `decile_lift_chart`
- `lorenz_chart`
- `coefficient_plot`
- `cv_stability_plot`
- `metrics_bar_chart`
- `interaction_heatmap`
- `pd_plot_2d`
- `importance_plot`
- `residual_heatmap`
- `regularization_path_plot`
- `overfitting_plot`
- `bootstrap_ci_plot`
- `relativities_ci_plot`

Two private helpers are important for understanding how plots stay consistent
with fitted models:

- `_resolve_level`
- `_sort_labels`

`_resolve_level` uses the fitted preprocessor when available. That allows plots
to use the same bin labels and category labels as the model instead of
re-binning independently. This matters when a chart should line up with model
factor levels.

## Bin Suggestions: `bin_suggestor.py`

`bin_suggestor.py` suggests candidate breakpoints for continuous variables. It
does not modify model configuration.

Available methods:

- `suggest_bins_quantile`: weighted quantile breaks
- `suggest_bins_equal_width`: equal-width breaks
- `suggest_bins_optbin`: optimal binning through `optbinning`
- `suggest_bins_gbm`: tree-threshold-based breaks from a GBM
- `suggest_bins`: combined wrapper for multiple strategies

Sentinel missing values are excluded from breakpoint calculations.

The intended workflow is:

1. Run one or more suggestion methods.
2. Review printed splits and optional plots.
3. Choose a breakpoint list.
4. Pass it into `add_variable(..., breakpoints=chosen_splits)`.

## Discovery Diagnostics: `discovery.py`

`discovery.py` contains exploratory modeling helpers. These generally use
LightGBM, SHAP, or both.

Important functions:

- `fit_shadow_gbm`
- `permutation_importance`
- `interaction_ranking`
- `partial_dependence_2d`
- `residual_gbm`
- `shap_importance`
- `shap_dependence`
- `shap_interaction_ranking`
- `tree_interaction_cooccurrence`
- `suggest_category_groups`
- `monotonicity_test`
- `boruta_select`

Categorical variables are one-hot encoded automatically for the tree-based
helpers. Importances are reported back at the original variable level rather
than only at the dummy-column level.

These tools are useful before and after GLM fitting:

- before fitting, to identify nonlinearities, interactions, and variables worth
  adding
- after fitting, to diagnose residual signal that the GLM missed

## Persistence: `io_utils.py`

`io_utils.py` handles saving and loading model versions.

Key functions:

- `save_version`
- `load_version`

The persistence layer saves a snapshot containing the model version, variable
configs, fitted preprocessor, and relevant tool metadata. It also includes
support for serializing custom transforms by inspecting their source.

Practical guidance:

- Prefer named transform functions over lambdas.
- Keep transform dependencies importable.
- Avoid relying on notebook-only local state inside custom transforms.

Those habits make saved models easier to reload.

## Excel Factor Versions

`ModelingTool` can also load factor-table models from Excel. These models are
stored as `FactorModelVersion` objects rather than `ModelVersion` objects.

The factor workflow is useful when:

- a model is represented by relativities/factors rather than a fitted `glum`
  estimator
- factors were reviewed or edited outside Python
- the same scoring and comparison utilities should be used for an external
  rating plan

The factor model predicts by resolving each row to a level for each variable,
looking up the factor for that level, and multiplying factors together.

## Version Management

Model versions are stored in `tool.model_versions`.

Each key is a version name. Each value is either:

- `ModelVersion` for a fitted GLM
- `FactorModelVersion` for an Excel factor table

The current version is usually the most recently fitted or loaded version.
Several methods accept `version=None`, in which case they use the current
version.

The base version is used for comparison workflows. It can be another fitted
version or a prediction column in the source data, depending on the method.

## How Data Moves Through the Tool

For a GLM fit, the data path is:

```text
raw polars DataFrame
  -> ModelingTool.fit_model(...)
  -> model.fit_model(...)
  -> Preprocessor.fit(...)
  -> Preprocessor.transform(...)
  -> numeric design matrix
  -> glum GLM fit
  -> ModelVersion
  -> stored in tool.model_versions[version]
```

For GLM prediction, the path is:

```text
new polars DataFrame
  -> ModelingTool.predict(...)
  -> ModelVersion.predict(...)
  -> stored Preprocessor.transform(...)
  -> numeric design matrix with training-time schema
  -> fitted glum model
  -> prediction array
```

For Excel factor prediction, the path is:

```text
new polars DataFrame
  -> ModelingTool.predict(...)
  -> FactorModelVersion.predict(...)
  -> resolve variable levels
  -> look up factor for each level
  -> multiply factors and optional offset
  -> prediction array
```

## Common Extension Points

To add a new preprocessing behavior, start in `variable.py`. Most additions will
touch `VariableConfig`, `Preprocessor._fit_col`, `_transform_num` or
`_transform_cat`, and `_compute_feature_names`.

To add a new model diagnostic that needs fitted model context, add a method to
`ModelingTool` and delegate computation to a stateless helper module when
possible.

To add a new metric, prefer adding a stateless function to `metrics.py`, then
wire it into `ModelingTool` only if it needs version lookup or model comparison
convenience.

To add a new plot, prefer adding the plotting function to `plots.py`, then wrap
it from `ModelingTool` if users need access to model versions, weights, target
columns, or fitted preprocessors.

To add a new discovery routine, keep the core implementation in `discovery.py`
and expose a light wrapper on `ModelingTool`.

## Important Gotchas

All user-facing DataFrames are expected to be `polars.DataFrame` objects.

`ModelVersion.train_predictions` is the stored training prediction array. There
is no `.predictions` attribute.

Custom transforms use the DataFrame-based signature
`custom_transform(df: pl.DataFrame, **kwargs)`. Some older comments or examples
may describe array or per-value signatures; the implemented code uses the
DataFrame signature.

The first argument to `add_variable` is the output variable name. For derived
variables, source columns go in `input_cols`.

Numeric sentinel missing values use `-999_999_999.0`. Sentinel values are
treated differently from ordinary nulls in several places, especially numeric
binning.

`log_transform=True` requires all transformed values to be positive. Columns
with sentinel-encoded missings or zero/negative values can fail this assertion.

`bin_edges` contains internal breakpoints only. Do not include outer lower or
upper bounds.

Reference levels are dropped from one-hot encoded variables. Coefficients are
relative to the dropped level.

Discovery helpers may require optional dependencies such as `lightgbm`, `shap`,
`openpyxl`, or `optbinning`. The core package dependencies are listed in
`pyproject.toml`; optional extras are under the `full` extra.

## Minimal End-to-End Example

```python
import polars as pl
from elastic_net_tool import ModelingTool

df = pl.DataFrame({
    "earned_premium": [1000.0, 1200.0, 900.0, 1500.0],
    "loss_ratio": [0.55, 0.72, 0.40, 0.88],
    "driver_age": [22.0, 45.0, 67.0, 35.0],
    "state": ["TX", "FL", "OH", "TX"],
    "cv_fold": [1, 1, 2, 2],
})

tool = ModelingTool(
    data=df,
    target_col="loss_ratio",
    weight_col="earned_premium",
    cv_column="cv_fold",
)

tool.add_variable("driver_age", breakpoints=[30, 60])
tool.add_variable("state", encoding="onehot")

tool.fit_model(
    variables=["driver_age", "state"],
    version="baseline",
    alpha=0.01,
)

pred = tool.predict(df, version="baseline")
summary = tool.model_summary("baseline")
```

This small example omits most diagnostics, but it shows the central pattern:
construct the tool, register variables, fit a named version, and score through
that version.
