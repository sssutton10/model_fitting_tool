"""
example_usage.py
================

Demonstrates all features of elastic_net_tool using synthetic insurance data.
All DataFrames use **polars**.

Run:
    python example_usage.py

Sections
--------
 1.  Synthetic data
 2.  Initialise the tool
 3.  Bin suggestion (quantile / equal-width / GBM / combined)
 4.  Variable creation  (uses breakpoints discovered above)
 5.  Univariate exploration
 6.  Baseline model  (cv_column -> PredefinedSplit)
 7.  CV stability
 8.  Richer model
 9.  Unpenalised GLM
10.  A/E charts
11.  Residual charts
12.  Model comparison  (absolute and relative deviation options)
13.  Coefficient plot
14.  Save and load
15.  Shadow GBM + SHAP importance & dependence
16.  SHAP interaction ranking + tree co-occurrence
17.  Boruta feature selection
18.  Category grouping suggestions
19.  Monotonicity test
20.  Permutation importance
21.  Residual GBM (find missing signal — includes categoricals)
22.  2D residual heatmap
23.  Regularization path
24.  Overfitting monitor
25.  VIF (multicollinearity)
26.  Bootstrap confidence intervals
27.  Bootstrap relativities
28.  Excel factor model for prediction
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # needed on Windows for Polars table output

import numpy as np
import polars as pl


# ── Named transform functions ────────────────────────────────────────────────
# Lambdas can't be pickled; named functions can.  These receive a
# pl.DataFrame with only the relevant columns (the DataFrame-based API).

def mileage_to_thousands(df: pl.DataFrame) -> np.ndarray:
    """Scale annual_mileage from raw units to thousands."""
    return df.to_series().to_numpy() / 1_000


def veh_age_x_driver_age(df: pl.DataFrame) -> np.ndarray:
    """Interaction: vehicle_age * driver_age."""
    return (df["vehicle_age"].to_numpy() * df["driver_age"].to_numpy())


def bucketed_vehicle_age(df: pl.DataFrame) -> list:
    """Custom categorical transform: bucket vehicle_age into named bands."""
    ages = df.to_series().to_numpy()
    labels = []
    for a in ages:
        if a <= 3:
            labels.append("new_0-3")
        elif a <= 7:
            labels.append("mid_4-7")
        elif a <= 12:
            labels.append("old_8-12")
        else:
            labels.append("very_old_13+")
    return labels


def region_remap(df: pl.DataFrame) -> list:
    """Custom categorical remap: collapse states into regions."""
    mapping = {"TX": "South", "FL": "South", "CA": "West", "NY": "East", "OH": "Midwest"}
    return [mapping.get(v, "Other") for v in df.to_series().to_list()]


# ── 1. Synthetic data ─────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
n = 5_000

raw = {
    "earned_premium": rng.uniform(500, 5_000, n),
    "vehicle_age":    rng.integers(0, 20, n).astype(float),
    "driver_age":     rng.integers(16, 80, n).astype(float),
    "annual_mileage": rng.uniform(3_000, 25_000, n),
    "prior_claims":   rng.integers(0, 5, n).astype(float),
    "state":          rng.choice(["TX", "FL", "CA", "NY", "OH"], n).tolist(),
    "vehicle_class":  rng.choice(["sedan", "suv", "truck", "sports"], n).tolist(),
    # Numeric column with ~8 % sentinel-encoded missing values
    "credit_score":   np.where(rng.random(n) < 0.08, -999_999_999.0,
                               rng.uniform(300, 850, n)),
    # CV fold column — values 1-5, used for PredefinedSplit later
    "cv_fold":        (np.arange(n) % 5 + 1).tolist(),
}

true_lr = (
    0.60
    + 0.02 * (raw["vehicle_age"] / 10)
    - 0.003 * ((raw["driver_age"] - 40) / 10) ** 2
    + 0.05 * (np.array(raw["prior_claims"]) > 0).astype(float)
    + 0.08 * (np.array(raw["state"]) == "FL").astype(float)
    - 0.06 * (np.array(raw["state"]) == "OH").astype(float)
    + rng.normal(0, 0.15, n)
).clip(0.05, 3.0)
raw["loss_ratio"] = true_lr

# Simulate an existing competitor model's predictions (for double-lift comparison)
raw["existing_model_pred"] = (
    0.60
    + 0.015 * (raw["vehicle_age"] / 10)
    + 0.03 * (np.array(raw["state"]) == "FL").astype(float)
    + rng.normal(0, 0.05, n)
).clip(0.1, 2.5)

df = pl.DataFrame(raw)
print("Dataset shape:", df.shape)
print(df.head(3))


# ── 2. Initialise the tool ────────────────────────────────────────────────────

from elastic_net_tool import ModelingTool, VariableConfig

# cv_column tells the tool to use the "cv_fold" column as a PredefinedSplit
# whenever fit_model is called without an explicit cv= argument.
tool = ModelingTool(
    data=df,
    target_col="loss_ratio",
    weight_col="earned_premium",
    cv_column="cv_fold",          # fold 1-5 -> PredefinedSplit automatically
    # family / link / tweedie_power default to Tweedie(1.5) / log
)


# ── 3. Bin suggestion ─────────────────────────────────────────────────────────
#
# Use the suggest_bins_* methods to explore candidate breakpoints BEFORE
# committing to any variable configuration.  The tool needs no fitted model
# for these — they only look at the raw column distribution.

print("\n--- Bin suggestion: driver_age ---")

# Individual strategies — each prints its splits and returns a list of floats
q_splits   = tool.suggest_bins_quantile("driver_age",   n_bins=7)
ew_splits  = tool.suggest_bins_equal_width("driver_age", n_bins=7)
gbm_splits = tool.suggest_bins_gbm("driver_age",        max_splits=6)

# Combined view: run multiple strategies at once and overlay them on a
# weighted histogram so you can compare visually in one call.
print("\n--- Bin suggestion: annual_mileage (combined view) ---")
am_splits = tool.suggest_bins(
    "annual_mileage",
    methods=["quantile", "equal_width", "gbm"],
    n_bins=6,
    max_splits=8,
    show_plot=True,   # weighted histogram with colour-coded split lines
)
# am_splits is a dict: {"quantile": [...], "equal_width": [...], "gbm": [...]}
print("Chosen GBM splits for annual_mileage:", am_splits["gbm"])

print("\n--- Bin suggestion: credit_score (sentinel values excluded automatically) ---")
cs_splits = tool.suggest_bins(
    "credit_score",
    methods=["quantile", "gbm"],
    n_bins=8,
    max_splits=7,
    show_plot=True,
)


# ── 4. Variable creation ──────────────────────────────────────────────────────
#
# custom_transform receives a pl.DataFrame with only the relevant columns.
# For single-column variables the DataFrame has one column; for multi-input
# variables it contains all input_cols.

# Numeric: 99th-percentile cap (cap_upper < 1.0 is treated as a quantile)
tool.add_variable("vehicle_age", cap_upper=0.99)

# Numeric: use breakpoints discovered by suggest_bins above.
# 'breakpoints' is an alias for 'bin_edges' — both are accepted.
tool.add_variable("driver_age", breakpoints=q_splits)

# Numeric: custom_transform scales miles to thousands, then cap.
# The function receives a 1-column pl.DataFrame and returns an array.
tool.add_variable(
    "annual_mileage",
    cap_upper=0.99,
    custom_transform=mileage_to_thousands,
)

# Numeric with sentinel missing values: -999999999 -> 'Missing' bin.
# n_bins builds the bins and a separate _missing dummy automatically.
# Use the breakpoints returned by suggest_bins for full control:
tool.add_variable("credit_score", cap_lower=0.01, cap_upper=0.99,
                  breakpoints=cs_splits.get("quantile") or [400, 550, 650, 750])

# Categorical: simple one-hot (reference level dropped by max exposure weight)
tool.add_variable("state", encoding="onehot")

# Categorical: custom_transform remaps state -> region BEFORE one-hot encoding.
# The function receives a 1-column DataFrame of the raw state values and
# returns a list of strings.  These become the categorical levels.
tool.add_variable(
    "region",
    input_cols=["state"],
    custom_transform=region_remap,
    is_categorical=True,
    encoding="onehot",
)

# Categorical: custom_transform that creates category bands from a numeric column.
tool.add_variable(
    "veh_age_band",
    input_cols=["vehicle_age"],
    custom_transform=bucketed_vehicle_age,
    is_categorical=True,
    encoding="onehot",
)

tool.add_variable("vehicle_class", encoding="onehot")

# Plain numeric (no special transforms)
tool.add_variable("prior_claims", cap_upper=0.99)

# Multi-input derived variable (vehicle_age x driver_age interaction).
# custom_transform receives a pl.DataFrame with columns [vehicle_age, driver_age].
tool.add_variable(
    "veh_x_age",
    input_cols=["vehicle_age", "driver_age"],
    custom_transform=veh_age_x_driver_age,
    cap_upper=0.99,
)

print("\nRegistered variable configs:")
print(tool.list_variables())


# ── 5. Univariate exploration ─────────────────────────────────────────────────

print("\n--- Univariate plots ---")
tool.univariate_plot("driver_age",   n_bins=10)  # continuous -> binned
tool.univariate_plot("state")                    # categorical
tool.univariate_plot("credit_score", n_bins=8)   # sentinel shown as 'Missing'


# ── 6. Baseline model (cv_column -> PredefinedSplit) ─────────────────────────
#
# Because cv_column="cv_fold" was set at construction, fit_model automatically
# builds a PredefinedSplit from that column — no cv= argument needed.

print("\n--- Fitting baseline model (v1) via PredefinedSplit ---")
v1_vars = ["vehicle_age", "driver_age", "state", "prior_claims"]

tool.fit_model(
    variables=v1_vars,
    version="v1",
    use_cv=True,
    # cv=None (default) -> PredefinedSplit from cv_fold column
    l1_ratio=[0.1, 0.5, 0.9, 1.0],
)

print("\n--- Model summary (direct call) ---")
tool.model_summary("v1")


# ── 7. CV stability using the fold column ─────────────────────────────────────

print("\n--- CV stability for v1 ---")
stability = tool.fit_cv_stability(
    variables=v1_vars,
    fold_col="cv_fold",
    version="v1",   # borrows alpha / l1_ratio from v1
    plot=True,
)
# Summary rows appended automatically: geomean, std, cv_pct
print(stability.filter(pl.col("fold").is_in(["geomean", "std", "cv_pct"])))


# ── 8. Richer model ───────────────────────────────────────────────────────────

print("\n--- Fitting richer model (v2) ---")
v2_vars = v1_vars + ["annual_mileage", "credit_score", "vehicle_class", "veh_x_age"]

tool.fit_model(
    variables=v2_vars,
    version="v2",
    use_cv=True,
    l1_ratio=[0.1, 0.5, 0.9, 1.0],
)


# ── 9. Near-unpenalised GLM ───────────────────────────────────────────────────
#
# alpha=0.0 gives a pure MLE GLM; on datasets with many dummy variables it can
# produce a singular design matrix.  alpha=1e-6 is practically unpenalised while
# remaining numerically stable.

print("\n--- Near-unpenalised GLM (v_glm) ---")
tool.fit_model(
    variables=v1_vars,
    version="v_glm",
    use_cv=False,
    alpha=1e-6,
)

print("\nAll versions:")
print(tool.list_versions())


# ── 10. A/E charts ────────────────────────────────────────────────────────────

print("\n--- Actual vs Expected ---")
tool.ae_chart("state",          version="v2")               # in-model variable
tool.ae_chart("vehicle_class",  version="v1")               # out-of-model variable
tool.ae_chart("annual_mileage", version="v1", n_bins=10)
tool.ae_chart("credit_score",   version="v2", n_bins=8)     # 'Missing' bucket shown


# ── 11. Residual charts ───────────────────────────────────────────────────────
#
# Residual chart shows mean_actual / mean_predicted per level.
# A ratio of 1.0 means perfect fit; > 1.0 means under-predicting for that group.

print("\n--- Residual charts ---")
tool.residual_chart("driver_age",     version="v2", n_bins=10)
tool.residual_chart("state",          version="v2")
tool.residual_chart("annual_mileage", version="v1", n_bins=10)
tool.residual_chart("vehicle_class",  version="v1")
tool.residual_chart("credit_score",   version="v1", n_bins=8)
tool.residual_chart("credit_score",   version="v2", n_bins=8)


# ── 12. Model comparison ──────────────────────────────────────────────────────
#
# compare_models produces a side-by-side metrics table (RMSE, MAE, Gini,
# double-lift score) and a double-lift chart.
#
# deviation='absolute'  (default) — score = Σ(|m1−a| − |m2−a|) per bucket.
#   Intuitive when both models are similarly scaled.
# deviation='relative'            — score = Σ(|a/m1−1| − |a/m2−1|) per bucket.
#   Scale-free: a 5% miss vs 0.50 actual counts the same as 10% vs 1.00.
#   Better when predictions span very different magnitudes.

print("\n--- v1 vs v2 (absolute deviation, default) ---")
results = tool.compare_models("v1", "v2", n_buckets=10)
print("\nDouble-lift table (first 5 rows):")
print(results["double_lift"].head(5))

print("\n--- v1 vs v2 (relative deviation) ---")
results_rel = tool.compare_models("v1", "v2", n_buckets=10, deviation="relative")

print("\n--- v1 vs v_glm ---")
tool.compare_models("v1", "v_glm", n_buckets=10)

# Compare against a column of predictions already in the dataset
# (version2 can be a column name as well as a registered version name)
print("\n--- v2 vs existing_model_pred column ---")
tool.compare_models("v2", "existing_model_pred", n_buckets=10)


# ── 13. Coefficient plot ──────────────────────────────────────────────────────

tool.coefficient_plot("v2", top_n=20)


# ── 14. Save and load ─────────────────────────────────────────────────────────

print("\n--- Save v2, reload, frozen predict ---")
import os
os.makedirs("models", exist_ok=True)
tool.save("v2", "models/v2.pkl")

# Load onto a new dataset: variable configs and hyperparameters are restored,
# the model is refit from scratch, and stored as version 'v1'.
df_test = df.sample(500, seed=99)
tool_loaded = ModelingTool.load("models/v2.pkl", data=df_test)
print(tool_loaded.list_versions())

# Frozen load: no data needed — predictions use the stored model weights directly.
# Useful for scoring new data without access to the training frame.
# drop_reference and cv_column are now correctly restored from the snapshot.
tool_frozen = ModelingTool.load_frozen("models/v2.pkl")
new_preds = tool_frozen.model_versions["v1"].predict(df_test)
print(f"Frozen predictions (first 5): {new_preds[:5]}")


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════


# ── 15. Shadow GBM + SHAP importance & dependence ───────────────────────────
#
# Fit a LightGBM as a diagnostic lens — never the final model.
# fit_shadow_gbm automatically excludes target, weight, and cv_column so the
# fold column never leaks into the feature matrix.
#
# Two encoding modes:
#   use_categorical=False (default) — one-hot encode all string/categorical
#     columns; each dummy is a separate feature; importances are summed per
#     original variable.
#   use_categorical=True            — pass integer codes directly with LightGBM's
#     native categorical handling; optimal partition splits on all levels at once.

print("\n" + "=" * 62)
print("  SHADOW GBM + SHAP IMPORTANCE & DEPENDENCE")
print("=" * 62)

# fit_shadow_gbm with default (one-hot) encoding
# feature_cols omitted -> all numeric + categorical columns except target/weight/cv
shadow = tool.fit_shadow_gbm(
    n_estimators=200,
    max_depth=5,
)

# SHAP importance: Mean |SHAP value| per original variable.
# More stable than permutation importance — no repeated shuffling needed.
# Importances for categorical dummies are summed back to the original variable.
# Requires: pip install shap
print("\n--- SHAP importance ---")
try:
    shap_imp = tool.shap_importance(sample_size=500, random_state=42)
    print(shap_imp)

    from elastic_net_tool import importance_plot
    fig = importance_plot(shap_imp, top_n=10)
except ImportError:
    print("  shap not installed — skipping (pip install shap)")

# SHAP dependence: how a variable's SHAP value varies with its raw value.
# The scatter shape directly reveals the best transform and where breakpoints
# should go — no need for trial-and-error with n_bins.
# color_var adds a second dimension to reveal pairwise interactions (pink/blue
# coloring shows whether the interaction is additive or multiplicative).
print("\n--- SHAP dependence: driver_age (coloured by vehicle_age) ---")
try:
    dep_data = tool.shap_dependence(
        "driver_age", color_var="vehicle_age", sample_size=500
    )
    print(dep_data.head(5))
    # dep_data has columns: driver_age, shap_value, vehicle_age
    # Plot with matplotlib: scatter(dep_data["driver_age"], dep_data["shap_value"])

    print("\n--- SHAP dependence: credit_score ---")
    dep_cs = tool.shap_dependence("credit_score", sample_size=500)
    # A hinge shape at ~650 confirms that breakpoint; a plateau at the low end
    # suggests the existing sentinel binning is appropriate.
    print(dep_cs.head(5))
except ImportError:
    print("  shap not installed — skipping (pip install shap)")


# ── 16. SHAP interaction ranking + tree co-occurrence ────────────────────────
#
# Two methods for interaction discovery — use both for triangulation.
#
# tree_interaction_cooccurrence: fast heuristic, no extra library needed.
#   Counts how often two variables appear in the same tree, weighted by split
#   gain.  Use as a cheap pre-screen — O(n_trees × n_features).
#
# shap_interaction_ranking: exact TreeSHAP interactions, more accurate.
#   Covers categorical variables.  Requires pip install shap.
#   More expensive than co-occurrence: O(sample × n_features²).

print("\n" + "=" * 62)
print("  INTERACTION DISCOVERY")
print("=" * 62)

# Fast tree co-occurrence: no extra dependencies
print("\n--- Tree co-occurrence (fast, no shap needed) ---")
cooc = tool.tree_interaction_cooccurrence(top_n=15)
print(cooc)

# Friedman H-statistic via PDP grid (numeric pairs only)
print("\n--- Friedman H-statistic interaction ranking (numeric pairs) ---")
interactions = tool.interaction_ranking(top_n=10, grid_resolution=10, sample_size=500)
print(interactions)

# SHAP interaction ranking (covers numeric AND categorical variable pairs)
print("\n--- SHAP interaction ranking ---")
try:
    shap_ir = tool.shap_interaction_ranking(sample_size=200, top_n=15)
    print(shap_ir)
except ImportError:
    print("  shap not installed — skipping (pip install shap)")

# 2D partial dependence: visualise HOW two variables interact
print("\n--- 2D Partial Dependence: driver_age x vehicle_age ---")
pd_data = tool.partial_dependence_2d("driver_age", "vehicle_age",
                                      grid_resolution=15, sample_size=500)
from elastic_net_tool import pd_plot_2d, interaction_heatmap
fig = pd_plot_2d(pd_data, "driver_age", "vehicle_age")
fig = interaction_heatmap(interactions, top_n=10)


# ── 17. Boruta feature selection ─────────────────────────────────────────────
#
# Boruta-style selection: in each iteration, shuffled shadow copies of every
# feature are appended to the matrix and LightGBM is fit.  A real feature
# "wins" if its importance exceeds the maximum shadow feature importance.
# Features that win in >= (1 - threshold) of iterations are selected.
#
# This provides an objective cut-off rather than eyeballing the importance chart.
# Use it to trim a wide candidate list before building the GLM.

print("\n" + "=" * 62)
print("  BORUTA FEATURE SELECTION")
print("=" * 62)

boruta_result = tool.boruta_select(
    n_estimators=100,
    n_iterations=20,
    threshold=0.05,   # must win 95% of iterations to be selected
    random_state=42,
)
print(boruta_result)
selected_vars = boruta_result.filter(pl.col("selected"))["variable"].to_list()
print(f"\nSelected variables ({len(selected_vars)}): {selected_vars}")
# Use selected_vars as the starting point for your GLM variable list


# ── 18. Category grouping suggestions ────────────────────────────────────────
#
# High-cardinality categoricals (50+ levels) often need manual grouping before
# one-hot encoding.  suggest_category_groups sorts levels by exposure-weighted
# mean target and merges adjacent groups greedily until max_groups remain.
# Levels below min_exposure_pct of total exposure are merged first.
#
# Returns:
#   level_to_group: dict mapping each original level to a group label (G01, G02…)
#   summary: DataFrame with group, levels, exposure, mean_target

print("\n" + "=" * 62)
print("  CATEGORY GROUPING SUGGESTIONS")
print("=" * 62)

# state has 5 levels — suggest how to merge them down to 3 groups
print("\n--- Suggest groupings for state (max_groups=3) ---")
level_map, group_summary = tool.suggest_category_groups(
    "state",
    max_groups=3,
    min_exposure_pct=0.02,  # merge levels with < 2% of total exposure
    verbose=True,
)
print("\nLevel → group mapping:")
print(level_map)
print("\nGroup summary:")
print(group_summary)

# You can then define a custom_transform using this mapping:
# def state_grouped(df):
#     return [level_map.get(v, "Other") for v in df.to_series().to_list()]
# tool.add_variable("state_grouped", input_cols=["state"],
#                   custom_transform=state_grouped, is_categorical=True)

# vehicle_class: suggest 2 groups
print("\n--- Suggest groupings for vehicle_class (max_groups=2) ---")
vc_map, vc_summary = tool.suggest_category_groups("vehicle_class", max_groups=2)
print(vc_map)
print(vc_summary)


# ── 19. Monotonicity test ─────────────────────────────────────────────────────
#
# Before enforcing a monotone constraint in the GLM (e.g. "older vehicles must
# have higher loss ratios"), measure how much predictive accuracy you sacrifice.
#
# Fits three LightGBM models: unconstrained, monotone-increasing, monotone-
# decreasing.  Reports RMSE cost vs baseline and recommends a direction.
#
# cost < ~1%  → safe to enforce the constraint
# cost > ~5%  → the data does not support monotonicity; consider keeping the
#               unconstrained binning or relaxing the constraint

print("\n" + "=" * 62)
print("  MONOTONICITY TEST")
print("=" * 62)

print("\n--- Monotonicity test: driver_age ---")
mono_age = tool.monotonicity_test("driver_age", verbose=True)
# mono_age["recommended"] is 'increasing', 'decreasing', or 'no_constraint'
print(f"\nRecommended constraint: {mono_age['recommended']}")
print(f"  Unconstrained RMSE : {mono_age['unconstrained_rmse']:.6f}")
print(f"  Cost (increasing)  : {mono_age['cost_pos']:.2%}")
print(f"  Cost (decreasing)  : {mono_age['cost_neg']:.2%}")

print("\n--- Monotonicity test: vehicle_age ---")
mono_veh = tool.monotonicity_test("vehicle_age", verbose=True)
print(f"Recommended: {mono_veh['recommended']}")

print("\n--- Monotonicity test: credit_score ---")
mono_cs = tool.monotonicity_test("credit_score", verbose=True)
print(f"Recommended: {mono_cs['recommended']}")


# ── 20. Permutation importance ───────────────────────────────────────────────
#
# Model-agnostic variable screening.  For categorical variables, all dummy
# columns are shuffled together so importance is reported per original column.
# SHAP importance (section 15) is generally preferred — use permutation
# importance when shap is not available or for a quick sanity check.

print("\n" + "=" * 62)
print("  PERMUTATION IMPORTANCE")
print("=" * 62)

print("\n--- Permutation importance (shadow GBM) ---")
importance = tool.permutation_importance(n_repeats=5)
print(importance)

from elastic_net_tool import importance_plot
fig = importance_plot(importance, top_n=10)


# ── 21. Residual GBM (find missing signal) ───────────────────────────────────
#
# Fit a GBM on the GLM's residuals (actual / predicted).  The top split
# variables and thresholds tell you exactly what signal the GLM is missing.
#
# feature_cols now defaults to ALL numeric + categorical columns (excluding
# target, weight, and cv_column), so string/categorical features are included
# automatically via one-hot encoding inside the function.

print("\n" + "=" * 62)
print("  RESIDUAL GBM")
print("=" * 62)

print("\n--- Residual GBM: what is v1 missing? ---")
# feature_cols omitted -> all numeric + categorical columns (excl. cv_fold)
resid_df = tool.residual_gbm(version="v1", top_n=7)
print(resid_df)

# Explicit feature list if you want to restrict the search:
# resid_df = tool.residual_gbm(
#     version="v1",
#     feature_cols=["driver_age", "vehicle_age", "annual_mileage",
#                   "prior_claims", "credit_score", "state", "vehicle_class"],
#     top_n=7,
# )


# ── 22. 2D residual heatmap ─────────────────────────────────────────────────
#
# Cross-tabulate actual/expected ratio across two variable dimensions.
# Cells deviating from 1.0 with meaningful exposure reveal interactions.

print("\n" + "=" * 62)
print("  2D RESIDUAL HEATMAP")
print("=" * 62)

print("\n--- 2D Residual Heatmap: state x vehicle_class (v1) ---")
fig, heatmap_data = tool.residual_heatmap("v1", "state", "vehicle_class", n_bins=8)
print(heatmap_data.head(10))

print("\n--- 2D Residual Heatmap: driver_age x credit_score (v1) ---")
fig, heatmap_data2 = tool.residual_heatmap("v1", "driver_age", "credit_score", n_bins=6)


# ── 23. Regularization path ─────────────────────────────────────────────────
#
# Fit the GLM at a sequence of alpha values and track how each coefficient
# evolves.  Variables that "enter" at high alpha are robust predictors;
# those that only appear at low alpha are fragile or overfitting.

print("\n" + "=" * 62)
print("  REGULARIZATION PATH")
print("=" * 62)

print("\n--- Regularization path (v2 variables) ---")
path_df = tool.regularization_path(
    version="v2",
    l1_ratio=0.5,
    n_alphas=30,
    alpha_min=1e-4,
    alpha_max=5.0,
)
print(f"Path shape: {path_df.shape}")


# ── 24. Overfitting monitor ─────────────────────────────────────────────────
#
# Track train vs CV metric as variables are added.  A widening gap between
# train and CV performance signals overfitting.

print("\n" + "=" * 62)
print("  OVERFITTING MONITOR")
print("=" * 62)

print("\n--- Overfitting monitor: v1 → v2 ---")
monitor_df = tool.overfitting_monitor(["v1", "v2", "v_glm"])
print(monitor_df)


# ── 25. VIF (multicollinearity) ──────────────────────────────────────────────
#
# Variance Inflation Factor for each feature in the design matrix.
# VIF > 5 = moderate concern; VIF > 10 = serious multicollinearity.

print("\n" + "=" * 62)
print("  VIF (MULTICOLLINEARITY)")
print("=" * 62)

print("\n--- VIF table (v2) ---")
vif_df = tool.vif_table("v2")
print(vif_df.head(15))

high_vif = vif_df.filter(pl.col("vif") > 5.0)
if len(high_vif) > 0:
    print(f"\nWarning: {len(high_vif)} features with VIF > 5:")
    print(high_vif)


# ── 26. Bootstrap confidence intervals ──────────────────────────────────────
#
# Bootstrap CIs on Gini and MSE tell you whether a model improvement
# is statistically meaningful or within noise.

print("\n" + "=" * 62)
print("  BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 62)

print("\n--- Bootstrap CIs on v2 metrics ---")
from elastic_net_tool import gini_coefficient

boot_df = tool.bootstrap_metrics(
    "v2",
    metric_fns={
        "gini_norm": lambda yt, yp, w: gini_coefficient(yt, yp, w, normalize=True),
        "rmse": lambda yt, yp, w: -float(np.average((yt - yp) ** 2, weights=w)) ** 0.5,
    },
    n_bootstrap=200,
    ci=0.95,
)
print(boot_df)


# ── 27. Bootstrap relativities ──────────────────────────────────────────────
#
# Resample training data, refit the GLM, and compute CIs on each factor
# relativity.  Failed bootstrap samples (e.g. dropped level) contribute NaN
# and are excluded from the quantile computation rather than inflating the
# point estimate.

print("\n" + "=" * 62)
print("  BOOTSTRAP RELATIVITIES")
print("=" * 62)

print("\n--- Bootstrap relativity CIs (v2, 100 resamples) ---")
rel_ci = tool.bootstrap_relativities("v2", n_bootstrap=100, ci=0.95)
print(rel_ci.head(15))

# Plot relativity CIs for a specific variable
from elastic_net_tool import relativities_ci_plot
fig = relativities_ci_plot(rel_ci, "state")


# ══════════════════════════════════════════════════════════════════════════════
# 28. EXCEL FACTOR MODEL FOR PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
#
# A common workflow: someone builds a model externally (or you make manual
# selections in Excel), and you want to bring those factors back into Python
# to score data and compare against your GLM.
#
# The Excel file needs columns: Variable, Level, Factor.
# The "Factor" is a multiplicative relativity (1.0 = base level).

print("\n" + "=" * 62)
print("  EXCEL FACTOR MODEL FOR PREDICTION")
print("=" * 62)

import os
os.makedirs("models", exist_ok=True)

factor_data = pl.DataFrame({
    "Variable": [
        "intercept",
        "state", "state", "state", "state", "state",
        "vehicle_class", "vehicle_class", "vehicle_class", "vehicle_class",
    ],
    "Level": [
        "intercept",
        "CA", "TX", "FL", "NY", "OH",
        "sedan", "suv", "truck", "sports",
    ],
    "Factor": [
        0.65,                          # global intercept
        1.00, 1.05, 1.15, 0.98, 0.90,  # state relativities
        1.00, 1.08, 1.12, 1.25,        # vehicle_class relativities
    ],
})
factor_data.write_excel("models/competitor_factors.xlsx", worksheet="Factors")
print("\nCreated sample Excel factor table: models/competitor_factors.xlsx")

# ── Method 1: add_excel_version on an existing tool ─────────────────────────
tool.add_excel_version(
    filepath="models/competitor_factors.xlsx",
    sheet_name="Factors",
    version="competitor",
    missing_factor=1.0,
)

comp_preds = tool.model_versions["competitor"].train_predictions
print(f"Competitor predictions (first 5): {comp_preds[:5]}")

print("\n--- Your model (v2) vs Competitor (Excel factors) ---")
results = tool.compare_models("v2", "competitor", n_buckets=10)
print(results["metrics"])

# ── Method 2: load_from_excel — build a fresh tool from an Excel file ────────
print("\n--- Building standalone tool from Excel factors ---")
tool_excel = ModelingTool.load_from_excel(
    excel_path="models/competitor_factors.xlsx",
    sheet_name="Factors",
    data=df,
    target_col="loss_ratio",
    weight_col="earned_premium",
    version="competitor_standalone",
    # pkl_path="models/v2.pkl",  # uncomment if Excel has binned numeric variables
)

from elastic_net_tool import compute_metrics
excel_preds = tool_excel.model_versions["competitor_standalone"].train_predictions
metrics = compute_metrics(
    df["loss_ratio"].to_numpy(),
    excel_preds,
    weights=df["earned_premium"].to_numpy(),
    version_name="competitor",
)
print(metrics)

# ── Standalone double-lift using the metrics module directly ─────────────────
from elastic_net_tool import double_lift_table, double_lift_score, double_lift_chart

v2_preds = tool.model_versions["v2"].train_predictions
existing_preds = df["existing_model_pred"].to_numpy()
y_true = df["loss_ratio"].to_numpy()
w = df["earned_premium"].to_numpy()

dl_table = double_lift_table(y_true, v2_preds, existing_preds, weights=w, n_buckets=10)

# Absolute deviation (default) — straightforward sum of |m1−a| − |m2−a|
dl_score_abs = double_lift_score(dl_table, deviation="absolute")
# Relative deviation — scale-free, useful when predictions span different ranges
dl_score_rel = double_lift_score(dl_table, deviation="relative")

print(f"\nDouble-lift score (v2 vs existing):")
print(f"  absolute: {dl_score_abs:.4f}  | relative: {dl_score_rel:.4f}")
print("  Negative = v2 wins, Positive = existing wins")
print(dl_table)

fig = double_lift_chart(y_true, v2_preds, existing_preds, weights=w,
                        name1="v2 (yours)", name2="Existing Model")


# ══════════════════════════════════════════════════════════════════════════════
# SUGGESTED MODELLING WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. Screen variables (objective cut-off):
#       tool.fit_shadow_gbm(...)
#       tool.boruta_select(...)           ← new: objective pass/fail threshold
#       tool.shap_importance(...)         ← new: stable, no repeated shuffling
#    → Focus on selected / high-importance variables
#
# 2. Decide bin placements:
#       tool.suggest_bins(col, ...)       ← quantile / GBM / optbin
#       tool.shap_dependence(col, ...)    ← new: transform shape & breakpoints
#       tool.monotonicity_test(col, ...)  ← new: safe to enforce monotone?
#
# 3. Group high-cardinality categoricals:
#       tool.suggest_category_groups(col, max_groups=5)  ← new
#    → Build a custom_transform from the returned level_to_group dict
#
# 4. Discover interactions:
#       tool.tree_interaction_cooccurrence(...)    ← new: fast pre-screen
#       tool.shap_interaction_ranking(...)         ← new: exact, covers cats
#       tool.interaction_ranking(...)              ← Friedman H (numeric only)
#       tool.partial_dependence_2d(var1, var2)
#
# 5. Build base GLM:
#       tool.fit_model(top_vars, version="v1", ...)
#
# 6. Check multicollinearity:
#       tool.vif_table("v1")
#    → Drop or combine features with VIF > 10
#
# 7. Find missing signal:
#       tool.residual_gbm("v1", ...)     ← includes categoricals automatically
#       tool.residual_heatmap("v1", col1, col2)
#
# 8. Add interactions / variables, refit:
#       tool.add_variable("var1_x_var2", input_cols=[...], custom_transform=...)
#       tool.fit_model([...], version="v2", ...)
#
# 9. Monitor overfitting:
#       tool.overfitting_monitor(["v1", "v2", "v3"])
#    → Stop when train/CV gap widens faster than CV improves
#
# 10. Check coefficient stability:
#       tool.fit_cv_stability(...)
#       tool.regularization_path(version="v2")
#
# 11. Quantify improvement:
#       tool.bootstrap_metrics("v2", ...)
#       tool.compare_models("v1", "v2", deviation="relative")
#    → Is the double-lift improvement real or within noise?
#
# 12. Export for manual selection:
#       rel_table = tool.relativities_table("v2")
#       rel_ci = tool.bootstrap_relativities("v2")
#       # Write to Excel; CIs show which relativities can be safely rounded
#
# 13. Load adjusted factors and compare:
#       tool.add_excel_version("adjusted_factors.xlsx", ...)
#       tool.compare_models("v2", "adjusted", deviation="relative")
#    → Final comparison on holdout set

print("\n--- Done ---")
