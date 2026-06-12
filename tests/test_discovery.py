"""Tests for the discovery module (shadow GBM interaction discovery)."""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest

_HAS_LIGHTGBM: bool = importlib.util.find_spec("lightgbm") is not None
pytestmark = pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")


@pytest.fixture(scope="module")
def disco_df() -> pl.DataFrame:
    """Small synthetic dataset for discovery tests (numeric only)."""
    rng = np.random.default_rng(123)
    n = 200
    x1 = rng.uniform(0, 100, n)
    x2 = rng.uniform(0, 50, n)
    x3 = rng.normal(10, 3, n)
    # y has a main effect from x1 and an interaction between x1 and x2
    y = 0.5 + 0.01 * x1 + 0.02 * x2 + 0.001 * x1 * x2 + rng.normal(0, 0.1, n)
    w = rng.uniform(100, 1000, n)
    return pl.DataFrame({
        "x1": x1, "x2": x2, "x3": x3,
        "target": y, "weight": w,
    })


@pytest.fixture(scope="module")
def disco_df_with_cats() -> pl.DataFrame:
    """Synthetic dataset with both numeric and categorical features."""
    rng = np.random.default_rng(456)
    n = 200
    x1 = rng.uniform(0, 100, n)
    x2 = rng.uniform(0, 50, n)
    # Categorical with signal: group A has higher target
    cat1 = rng.choice(["A", "B", "C"], n, p=[0.3, 0.4, 0.3]).tolist()
    cat_effect = np.array([0.5 if c == "A" else (0.0 if c == "B" else -0.3) for c in cat1])
    y = 1.0 + 0.01 * x1 + cat_effect + rng.normal(0, 0.2, n)
    w = rng.uniform(100, 1000, n)
    return pl.DataFrame({
        "x1": x1, "x2": x2, "cat1": cat1,
        "target": y, "weight": w,
    })


class TestFitShadowGBM:
    def test_basic_fit(self, disco_df):
        from elastic_net_tool.discovery import fit_shadow_gbm

        model = fit_shadow_gbm(
            disco_df, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "x3"],
            n_estimators=20, max_depth=3,
        )
        assert hasattr(model, "predict")
        assert model._shadow_feature_cols == ["x1", "x2", "x3"]

    def test_auto_feature_cols(self, disco_df):
        from elastic_net_tool.discovery import fit_shadow_gbm

        model = fit_shadow_gbm(
            disco_df, target_col="target", weight_col="weight",
            n_estimators=10, max_depth=2,
        )
        # Should auto-detect x1, x2, x3 (exclude target and weight)
        assert "x1" in model._shadow_feature_cols
        assert "target" not in model._shadow_feature_cols
        assert "weight" not in model._shadow_feature_cols


class TestPermutationImportance:
    def test_returns_ranked_df(self, disco_df):
        from elastic_net_tool.discovery import fit_shadow_gbm, permutation_importance

        model = fit_shadow_gbm(
            disco_df, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "x3"],
            n_estimators=30, max_depth=3,
        )
        result = permutation_importance(
            model, disco_df, target_col="target", weight_col="weight",
            n_repeats=3,
        )
        assert isinstance(result, pl.DataFrame)
        assert "variable" in result.columns
        assert "importance_mean" in result.columns
        assert "importance_std" in result.columns
        assert len(result) == 3

    def test_x1_x2_rank_higher_than_x3(self, disco_df):
        """x1 and x2 have real signal; x3 is noise."""
        from elastic_net_tool.discovery import fit_shadow_gbm, permutation_importance

        model = fit_shadow_gbm(
            disco_df, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "x3"],
            n_estimators=50, max_depth=4,
        )
        result = permutation_importance(
            model, disco_df, target_col="target", weight_col="weight",
            n_repeats=5,
        )
        top_var = result["variable"][0]
        assert top_var in ("x1", "x2"), f"Expected x1 or x2 at top, got {top_var}"


class TestPartialDependence2D:
    def test_returns_grid_df(self, disco_df):
        from elastic_net_tool.discovery import fit_shadow_gbm, partial_dependence_2d

        model = fit_shadow_gbm(
            disco_df, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "x3"],
            n_estimators=20, max_depth=3,
        )
        result = partial_dependence_2d(
            model, disco_df, "x1", "x2",
            grid_resolution=5, sample_size=50,
        )
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"var1_value", "var2_value", "pd_value"}
        assert len(result) > 0


class TestInteractionRanking:
    def test_returns_pairs(self, disco_df):
        from elastic_net_tool.discovery import fit_shadow_gbm, interaction_ranking

        model = fit_shadow_gbm(
            disco_df, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "x3"],
            n_estimators=30, max_depth=4,
        )
        result = interaction_ranking(
            model, disco_df,
            top_n=3, grid_resolution=5, sample_size=50,
        )
        assert isinstance(result, pl.DataFrame)
        assert "var1" in result.columns
        assert "var2" in result.columns
        assert "h_statistic" in result.columns
        # 3 variables → 3 pairs
        assert len(result) == 3

    def test_x1_x2_has_highest_h(self, disco_df):
        """The x1*x2 interaction should have the highest H-statistic."""
        from elastic_net_tool.discovery import fit_shadow_gbm, interaction_ranking

        model = fit_shadow_gbm(
            disco_df, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "x3"],
            n_estimators=50, max_depth=5,
        )
        result = interaction_ranking(
            model, disco_df,
            top_n=3, grid_resolution=8, sample_size=100,
        )
        top_pair = {result["var1"][0], result["var2"][0]}
        assert top_pair == {"x1", "x2"}, f"Expected {{x1, x2}} at top, got {top_pair}"


class TestResidualGBM:
    def test_returns_importance(self, disco_df):
        from elastic_net_tool.discovery import residual_gbm

        # Simulate residuals: model that only uses x1
        residuals = disco_df["target"].to_numpy() / (0.5 + 0.01 * disco_df["x1"].to_numpy())
        result = residual_gbm(
            disco_df, residuals, feature_cols=["x1", "x2", "x3"],
            weight_col="weight", top_n=3,
            n_estimators=20, max_depth=3,
        )
        assert isinstance(result, pl.DataFrame)
        assert "variable" in result.columns
        assert "importance" in result.columns
        assert len(result) == 3


class TestCategoricalSupport:
    """Tests for categorical variable handling in shadow GBM."""

    def test_fit_with_categoricals(self, disco_df_with_cats):
        from elastic_net_tool.discovery import fit_shadow_gbm

        model = fit_shadow_gbm(
            disco_df_with_cats, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "cat1"],
            n_estimators=20, max_depth=3,
        )
        assert hasattr(model, "predict")
        assert model._shadow_feature_cols == ["x1", "x2", "cat1"]
        # cat1 has 3 levels → 3 dummy columns + 2 numeric = 5 encoded columns
        assert len(model._shadow_encoded_names) == 5
        assert "cat1_A" in model._shadow_encoded_names
        assert "cat1_B" in model._shadow_encoded_names
        assert "cat1_C" in model._shadow_encoded_names

    def test_auto_detects_categoricals(self, disco_df_with_cats):
        from elastic_net_tool.discovery import fit_shadow_gbm

        model = fit_shadow_gbm(
            disco_df_with_cats, target_col="target", weight_col="weight",
            n_estimators=10, max_depth=2,
        )
        assert "cat1" in model._shadow_feature_cols

    def test_col_index_map(self, disco_df_with_cats):
        from elastic_net_tool.discovery import fit_shadow_gbm

        model = fit_shadow_gbm(
            disco_df_with_cats, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "cat1"],
            n_estimators=10, max_depth=2,
        )
        cim = model._shadow_col_index_map
        assert len(cim["x1"]) == 1
        assert len(cim["x2"]) == 1
        assert len(cim["cat1"]) == 3  # 3 dummy columns

    def test_permutation_importance_with_cats(self, disco_df_with_cats):
        """Permutation importance reports one row per original column."""
        from elastic_net_tool.discovery import fit_shadow_gbm, permutation_importance

        model = fit_shadow_gbm(
            disco_df_with_cats, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "cat1"],
            n_estimators=30, max_depth=3,
        )
        result = permutation_importance(
            model, disco_df_with_cats,
            target_col="target", weight_col="weight", n_repeats=3,
        )
        assert len(result) == 3  # 3 original columns, not 5 encoded
        assert set(result["variable"].to_list()) == {"x1", "x2", "cat1"}

    def test_cat_has_signal(self, disco_df_with_cats):
        """cat1 has real signal and should rank above x2 (noise)."""
        from elastic_net_tool.discovery import fit_shadow_gbm, permutation_importance

        model = fit_shadow_gbm(
            disco_df_with_cats, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "cat1"],
            n_estimators=50, max_depth=4,
        )
        result = permutation_importance(
            model, disco_df_with_cats,
            target_col="target", weight_col="weight", n_repeats=5,
        )
        # x1 and cat1 both have signal; x2 is noise
        bottom_var = result["variable"][-1]
        assert bottom_var == "x2", f"Expected x2 at bottom, got {bottom_var}"

    def test_interaction_ranking_with_cats(self, disco_df_with_cats):
        """Interaction ranking only includes numeric pairs (skips categoricals)."""
        from elastic_net_tool.discovery import fit_shadow_gbm, interaction_ranking

        model = fit_shadow_gbm(
            disco_df_with_cats, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "cat1"],
            n_estimators=30, max_depth=4,
        )
        result = interaction_ranking(
            model, disco_df_with_cats,
            top_n=5, grid_resolution=5, sample_size=50,
        )
        # Only x1 and x2 are numeric → 1 pair
        assert len(result) == 1
        pair = {result["var1"][0], result["var2"][0]}
        assert pair == {"x1", "x2"}

    def test_pdp_rejects_categorical(self, disco_df_with_cats):
        """partial_dependence_2d raises ValueError for categorical variables."""
        from elastic_net_tool.discovery import fit_shadow_gbm, partial_dependence_2d

        model = fit_shadow_gbm(
            disco_df_with_cats, target_col="target", weight_col="weight",
            feature_cols=["x1", "x2", "cat1"],
            n_estimators=10, max_depth=2,
        )
        with pytest.raises(ValueError, match="categorical"):
            partial_dependence_2d(model, disco_df_with_cats, "x1", "cat1")

    def test_residual_gbm_with_cats(self, disco_df_with_cats):
        """residual_gbm handles categorical columns and aggregates importance."""
        from elastic_net_tool.discovery import residual_gbm

        residuals = np.ones(len(disco_df_with_cats))
        result = residual_gbm(
            disco_df_with_cats, residuals,
            feature_cols=["x1", "x2", "cat1"],
            weight_col="weight", top_n=3,
            n_estimators=20, max_depth=3,
        )
        assert len(result) == 3  # 3 original columns
        assert set(result["variable"].to_list()) == {"x1", "x2", "cat1"}
        # cat1's top_split_value should be NaN (not meaningful for categoricals)
        cat_row = result.filter(pl.col("variable") == "cat1")
        assert cat_row["top_split_value"].is_nan()[0]
