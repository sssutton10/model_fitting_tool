"""
Tests for tool.py — ModelingTool orchestration class.

Chart-returning methods are tested by suppressing plt.show() via monkeypatch
and asserting that a matplotlib Figure is returned.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest
from unittest.mock import patch
import matplotlib.figure


pytestmark = pytest.mark.requires_glum

from elastic_net_tool import ModelingTool, VariableConfig

_HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None


# ── Instantiation ─────────────────────────────────────────────────────────────

class TestInstantiation:
    def test_basic_instantiation(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        assert tool.target_col == "loss_ratio"
        assert tool.weight_col is None

    def test_with_weight_col(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio",
                            weight_col="earned_premium")
        assert tool.weight_col == "earned_premium"

    def test_with_cv_column(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio",
                            cv_column="cv_fold")
        assert tool.cv_column == "cv_fold"

    def test_invalid_cv_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="cv_column"):
            ModelingTool(sample_df, target_col="loss_ratio",
                         cv_column="nonexistent_col")

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="polars DataFrame"):
            ModelingTool({"a": [1, 2]}, target_col="a")

    def test_tweedie_power_stored(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio", tweedie_power=1.0)
        assert tool.tweedie_power == 1.0

    def test_drop_reference_stored(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio", drop_reference="first")
        assert tool.drop_reference == "first"

    def test_default_drop_reference_is_max_weight(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        assert tool.drop_reference == "max_weight"

    def test_empty_model_versions_on_init(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        assert tool.model_versions == {}

    def test_empty_variable_configs_on_init(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        assert tool.variable_configs == {}


# ── add_variable ──────────────────────────────────────────────────────────────

class TestAddVariable:
    def test_registers_config(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        tool.add_variable("driver_age", cap_upper=0.95)
        assert "driver_age" in tool.variable_configs

    def test_kwargs_passed_to_config(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        tool.add_variable("driver_age", cap_upper=0.95, log_transform=True)
        cfg = tool.variable_configs["driver_age"]
        assert cfg.cap_upper == 0.95
        assert cfg.log_transform is True

    def test_explicit_config_object_accepted(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        cfg = VariableConfig("driver_age", cap_upper=0.90, n_bins=5)
        tool.add_variable("driver_age", config=cfg)
        assert tool.variable_configs["driver_age"].n_bins == 5

    def test_overwrite_existing_config(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        tool.add_variable("driver_age", cap_upper=0.99)
        tool.add_variable("driver_age", cap_upper=0.90)   # overwrite
        assert tool.variable_configs["driver_age"].cap_upper == 0.90

    def test_returns_self_for_chaining(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        result = tool.add_variable("driver_age")
        assert result is tool

    def test_custom_transform_registered(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        fn = lambda a: a / 100
        tool.add_variable("driver_age", custom_transform=fn)
        assert tool.variable_configs["driver_age"].custom_transform is fn

    def test_transform_kwargs_registered(self, sample_df):
        def scale(arr, factor=1.0):
            return arr / factor

        tool = ModelingTool(sample_df, target_col="loss_ratio")
        tool.add_variable("driver_age", custom_transform=scale,
                          transform_kwargs={"factor": 10.0})
        assert tool.variable_configs["driver_age"].transform_kwargs == {"factor": 10.0}

    def test_multi_input_variable(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        tool.add_variable(
            "age_x_value",
            input_cols=["driver_age", "vehicle_value"],
            custom_transform=lambda a, v: a * v / 1e6,
        )
        assert "age_x_value" in tool.variable_configs
        assert tool.variable_configs["age_x_value"].input_cols == ["driver_age", "vehicle_value"]

    def test_breakpoints_kwarg_sets_bin_edges(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        tool.add_variable("driver_age", breakpoints=[25.0, 45.0, 65.0])
        assert tool.variable_configs["driver_age"].bin_edges == [25.0, 45.0, 65.0]


# ── fit_model ─────────────────────────────────────────────────────────────────

class TestFitModel:
    def test_version_stored_in_model_versions(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio",
                            weight_col="earned_premium")
        tool.fit_model(["driver_age"], version="v1", use_cv=False, alpha=0.01)
        assert "v1" in tool.model_versions

    def test_multiple_versions_tracked(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        tool.fit_model(["driver_age"], version="v1", use_cv=False, alpha=0.01)
        tool.fit_model(["vehicle_value"], version="v2", use_cv=False, alpha=0.01)
        assert "v1" in tool.model_versions
        assert "v2" in tool.model_versions

    def test_returns_model_version(self, sample_df):
        from elastic_net_tool.model import ModelVersion
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        mv = tool.fit_model(["driver_age"], version="v1", use_cv=False, alpha=0.01,
                            print_summary=False)
        assert isinstance(mv, ModelVersion)

    def test_default_config_applied_for_unregistered_variable(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        mv = tool.fit_model(["state"], version="v1", use_cv=False, alpha=0.01,
                            print_summary=False)
        assert any(f.startswith("state_") for f in mv.feature_names)

    def test_predefined_split_used_when_cv_column_set(self, sample_df, capsys):
        tool = ModelingTool(sample_df, target_col="loss_ratio",
                            weight_col="earned_premium", cv_column="cv_fold")
        tool.fit_model(["driver_age"], version="v1",
                       alphas=np.array([0.01, 0.1]), print_summary=False)
        captured = capsys.readouterr()
        assert "PredefinedSplit" in captured.out

    def test_explicit_cv_int_overrides_cv_column(self, sample_df, capsys):
        tool = ModelingTool(sample_df, target_col="loss_ratio",
                            cv_column="cv_fold")
        tool.fit_model(["driver_age"], version="v1", cv=3,
                       alphas=np.array([0.01, 0.1]), print_summary=False)
        captured = capsys.readouterr()
        # PredefinedSplit message should NOT appear
        assert "PredefinedSplit" not in captured.out

    def test_no_cv_column_and_no_cv_arg_defaults_to_five_fold(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        mv = tool.fit_model(["driver_age"], version="v1",
                            alphas=np.array([0.01, 0.1]), print_summary=False)
        # fit_info should record 5 as the fold count
        assert mv.fit_info.get("cv_folds") == 5

    def test_unpenalised_alpha_zero(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        mv = tool.fit_model(["driver_age"], version="v1",
                            use_cv=False, alpha=0.0, print_summary=False)
        assert mv.alpha == 0.0

    def test_unknown_variable_raises(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio")
        with pytest.raises(KeyError):
            tool.fit_model(["nonexistent_variable"], version="v1",
                           use_cv=False, alpha=0.01, print_summary=False)


# ── fit_cv_stability ──────────────────────────────────────────────────────────

class TestFitCVStabilityTool:
    def test_returns_dataframe(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio",
                            weight_col="earned_premium")
        # First fit a reference model so alpha/l1 are sensible
        tool.fit_model(["driver_age"], version="v1", use_cv=False,
                       alpha=0.01, print_summary=False)
        result = tool.fit_cv_stability(
            ["driver_age"], fold_col="cv_fold", version="v1",
            alpha=0.01, l1_ratio=0.5, plot=False, show=False,
        )
        assert isinstance(result, pl.DataFrame)

    def test_summary_rows_present(self, sample_df):
        tool = ModelingTool(sample_df, target_col="loss_ratio",
                            weight_col="earned_premium")
        tool.fit_model(["driver_age"], version="v1", use_cv=False,
                       alpha=0.01, print_summary=False)
        result = tool.fit_cv_stability(
            ["driver_age"], fold_col="cv_fold", version="v1",
            alpha=0.01, l1_ratio=0.5, plot=False, show=False,
        )
        assert "geomean" in result["fold"].to_list()


# ── model_summary / list_versions ─────────────────────────────────────────────

class TestVersionManagement:
    def test_list_versions_returns_dataframe(self, fitted_tool):
        result = fitted_tool.list_versions()
        assert isinstance(result, pl.DataFrame)

    def test_list_versions_has_both_versions(self, fitted_tool):
        result = fitted_tool.list_versions()
        versions = result["version"].to_list()
        assert "v1" in versions
        assert "v2" in versions

    def test_list_versions_has_metric_columns(self, fitted_tool):
        result = fitted_tool.list_versions()
        for col in ("rmse", "mae", "gini_norm"):
            assert col in result.columns

    def test_get_version_raises_for_missing(self, fitted_tool):
        with pytest.raises(KeyError, match="not found"):
            fitted_tool._get_version("v999")


# ── compare_models ────────────────────────────────────────────────────────────

class TestCompareModels:
    def test_returns_dict_with_expected_keys(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            result = fitted_tool.compare_models("v1", "v2", show=False)
        assert "metrics" in result
        assert "double_lift" in result

    def test_metrics_is_dataframe(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            result = fitted_tool.compare_models("v1", "v2", show=False)
        assert isinstance(result["metrics"], pl.DataFrame)

    def test_metrics_has_winner_column(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            result = fitted_tool.compare_models("v1", "v2", show=False)
        assert "winner" in result["metrics"].columns

    def test_metrics_has_double_lift_score_row(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            result = fitted_tool.compare_models("v1", "v2", show=False)
        metrics = result["metrics"]
        assert "double_lift_score" in metrics["metric"].to_list()

    def test_double_lift_is_dataframe(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            result = fitted_tool.compare_models("v1", "v2", show=False)
        assert isinstance(result["double_lift"], pl.DataFrame)

    def test_missing_version_raises(self, fitted_tool):
        with pytest.raises(KeyError):
            fitted_tool.compare_models("v1", "v_missing", show=False)


# ── ae_chart / residual_chart / univariate_plot ───────────────────────────────

class TestCharts:
    def test_ae_chart_returns_figure(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            fig = fitted_tool.ae_chart("driver_age", version="v1", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_ae_chart_non_model_variable(self, fitted_tool):
        """ae_chart should work for variables not in the model."""
        with patch("matplotlib.pyplot.show"):
            fig = fitted_tool.ae_chart("vehicle_value", version="v1", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_residual_chart_returns_figure(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            fig = fitted_tool.residual_chart("driver_age", version="v1", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_residual_chart_categorical_variable(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            fig = fitted_tool.residual_chart("state", version="v1", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_univariate_plot_returns_figure(self, base_tool):
        with patch("matplotlib.pyplot.show"):
            fig = base_tool.univariate_plot("driver_age", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_univariate_plot_categorical(self, base_tool):
        with patch("matplotlib.pyplot.show"):
            fig = base_tool.univariate_plot("state", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_univariate_plot_column_with_sentinel(self, base_tool):
        """annual_mileage has ~10% sentinel values; should not error."""
        with patch("matplotlib.pyplot.show"):
            fig = base_tool.univariate_plot("annual_mileage", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)


# ── suggest_bins (tool-level) ─────────────────────────────────────────────────

class TestToolSuggestBins:
    def test_suggest_bins_quantile_shortcut(self, base_tool):
        splits = base_tool.suggest_bins_quantile("driver_age", n_bins=5, verbose=False)
        assert isinstance(splits, list)
        assert len(splits) == 4   # n_bins - 1 interior splits

    def test_suggest_bins_equal_width_shortcut(self, base_tool):
        splits = base_tool.suggest_bins_equal_width("driver_age", n_bins=5, verbose=False)
        assert isinstance(splits, list)
        assert len(splits) == 4

    def test_suggest_bins_gbm_shortcut(self, base_tool):
        splits = base_tool.suggest_bins_gbm("driver_age", max_splits=10, verbose=False)
        assert isinstance(splits, list)
        assert len(splits) <= 10

    def test_suggest_bins_combined_returns_dict(self, base_tool):
        with patch("matplotlib.pyplot.show"):
            result = base_tool.suggest_bins(
                "driver_age",
                methods=["quantile", "equal_width"],
                n_bins=5,
                show_plot=False,
            )
        assert "quantile" in result
        assert "equal_width" in result

    def test_suggest_bins_splits_are_sorted(self, base_tool):
        splits = base_tool.suggest_bins_quantile("vehicle_value", n_bins=5, verbose=False)
        assert splits == sorted(splits)


# ── relativities_table ────────────────────────────────────────────────────────

class TestRelativitiesTable:
    def test_returns_dataframe(self, fitted_tool):
        result = fitted_tool.relativities_table("v2")
        assert isinstance(result, pl.DataFrame)

    def test_has_required_columns(self, fitted_tool):
        result = fitted_tool.relativities_table("v2")
        for col in ("variable", "level", "weight", "train_coef"):
            assert col in result.columns

    def test_excludes_pure_continuous_variables(self, fitted_tool):
        """driver_age is binned (has bin_edges) so it should appear;
        a plain continuous variable should not appear as a variable name."""
        result = fitted_tool.relativities_table("v2")
        variables_present = result["variable"].unique().to_list()
        # state and driver_age (binned) are in v2 — both should be present
        assert "state" in variables_present

    def test_base_level_has_zero_train_coef(self, fitted_tool):
        result = fitted_tool.relativities_table("v2")
        base_rows = result.filter(pl.col("level").str.ends_with("(base)"))
        assert len(base_rows) > 0
        assert (base_rows["train_coef"] == 0.0).all()

    def test_base_level_present_for_categorical(self, fitted_tool):
        result = fitted_tool.relativities_table("v2")
        state_rows = result.filter(pl.col("variable") == "state")
        assert state_rows.filter(pl.col("level").str.ends_with("(base)")).height == 1

    def test_weight_column_positive(self, fitted_tool):
        result = fitted_tool.relativities_table("v2")
        assert (result["weight"] >= 0).all()

    def test_weight_sums_to_approx_total_per_variable(self, fitted_tool, sample_df):
        """Weights across all levels of one variable should sum to total weight."""
        result = fitted_tool.relativities_table("v2")
        total_w = float(sample_df["earned_premium"].sum())
        state_w = float(result.filter(pl.col("variable") == "state")["weight"].sum())
        assert abs(state_w - total_w) < 1.0

    def test_with_fold_col_adds_fold_columns(self, fitted_tool):
        result = fitted_tool.relativities_table("v2", fold_col="cv_fold")
        fold_cols = [c for c in result.columns if c.startswith("fold_")]
        assert len(fold_cols) > 0

    def test_fold_base_levels_are_zero(self, fitted_tool):
        result = fitted_tool.relativities_table("v2", fold_col="cv_fold")
        fold_cols = [c for c in result.columns if c.startswith("fold_")]
        base_rows = result.filter(pl.col("level").str.ends_with("(base)"))
        for fc in fold_cols:
            assert (base_rows[fc] == 0.0).all()

    def test_no_discrete_variables_returns_empty(self, sample_df):
        """A model with only a plain continuous variable should return empty."""
        from elastic_net_tool import ModelingTool
        tool = ModelingTool(sample_df, target_col="loss_ratio",
                            weight_col="earned_premium")
        # driver_age with no bins → pure continuous
        tool.fit_model(["driver_age"], version="v1", use_cv=False,
                       alpha=0.01, print_summary=False)
        result = tool.relativities_table("v1")
        assert result.is_empty()

    def test_calib_df_adds_calib_weight_column(self, fitted_tool, sample_df):
        """Passing calib_df should add a calib_weight column."""
        calib = sample_df.sample(50, seed=7)
        result = fitted_tool.relativities_table("v2", calib_df=calib)
        assert "calib_weight" in result.columns

    def test_calib_df_weights_non_negative(self, fitted_tool, sample_df):
        calib = sample_df.sample(50, seed=7)
        result = fitted_tool.relativities_table("v2", calib_df=calib)
        assert (result["calib_weight"] >= 0).all()

    def test_calib_df_weight_sums_to_approx_calib_total(self, fitted_tool, sample_df):
        """calib_weight per variable should sum to ~total calib exposure."""
        calib = sample_df.sample(80, seed=11)
        result = fitted_tool.relativities_table("v2", calib_df=calib)
        total_calib_w = float(calib["earned_premium"].sum())
        state_cw = float(result.filter(pl.col("variable") == "state")["calib_weight"].sum())
        assert abs(state_cw - total_calib_w) < 1.0

    def test_no_calib_df_no_calib_weight_column(self, fitted_tool):
        """Without calib_df, calib_weight column must not be present."""
        result = fitted_tool.relativities_table("v2")
        assert "calib_weight" not in result.columns

    def test_calib_df_with_fold_col_both_present(self, fitted_tool, sample_df):
        """calib_weight and fold columns can coexist."""
        calib = sample_df.sample(60, seed=13)
        result = fitted_tool.relativities_table("v2", fold_col="cv_fold", calib_df=calib)
        assert "calib_weight" in result.columns
        fold_cols = [c for c in result.columns if c.startswith("fold_")]
        assert len(fold_cols) > 0


# ── plot_all_variables ────────────────────────────────────────────────────────

class TestPlotAllVariables:
    def test_returns_list_of_figures(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            figs = fitted_tool.plot_all_variables("v1", show=False)
        assert isinstance(figs, list)
        assert len(figs) == len(fitted_tool.model_versions["v1"].variables)

    def test_residual_chart_type(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            figs = fitted_tool.plot_all_variables("v1", chart="residual", show=False)
        import matplotlib.figure
        assert all(isinstance(f, matplotlib.figure.Figure) for f in figs)

    def test_ae_chart_type(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            figs = fitted_tool.plot_all_variables("v1", chart="ae", show=False)
        import matplotlib.figure
        assert all(isinstance(f, matplotlib.figure.Figure) for f in figs)

    def test_invalid_chart_raises(self, fitted_tool):
        with pytest.raises(ValueError, match="chart must be"):
            fitted_tool.plot_all_variables("v1", chart="lorenz", show=False)

    def test_figure_count_matches_v2_variables(self, fitted_tool):
        with patch("matplotlib.pyplot.show"):
            figs = fitted_tool.plot_all_variables("v2", chart="ae", show=False)
        assert len(figs) == len(fitted_tool.model_versions["v2"].variables)


# ── Excel factor version ───────────────────────────────────────────────────────

def _write_excel(path: str, rows: list, sheet: str = "Factors") -> None:
    """Write Variable/Level/Factor rows to an xlsx file via openpyxl."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(["Variable", "Level", "Factor"])
    for row in rows:
        ws.append(row)
    wb.save(path)


@pytest.mark.skipif(not _HAS_OPENPYXL, reason="openpyxl not installed")
class TestExcelVersion:
    """Tests for add_excel_version and load_from_excel."""

    # State levels as produced by relativities_table (preprocessor-based labels)
    _STATE_ROWS = [
        ["state", "CA (base)", 1.00],
        ["state", "TX",        1.05],
        ["state", "FL",        0.95],
        ["state", "NY",        1.10],
        ["state", "OH",        0.92],
    ]

    # State levels using raw column values (for standalone / direct-lookup mode)
    _STATE_ROWS_RAW = [
        ["state", "CA", 1.00],
        ["state", "TX", 1.05],
        ["state", "FL", 0.95],
        ["state", "NY", 1.10],
        ["state", "OH", 0.92],
    ]

    def test_add_returns_self(self, fitted_tool, tmp_path):
        path = str(tmp_path / "f.xlsx")
        _write_excel(path, self._STATE_ROWS)
        result = fitted_tool.add_excel_version(path, "Factors", version="ex_test")
        # Clean up so other tests are unaffected
        fitted_tool.model_versions.pop("ex_test", None)
        assert result is fitted_tool

    def test_version_registered(self, fitted_tool, tmp_path):
        path = str(tmp_path / "f.xlsx")
        _write_excel(path, self._STATE_ROWS)
        fitted_tool.add_excel_version(path, "Factors", version="ex_reg")
        assert "ex_reg" in fitted_tool.model_versions
        fitted_tool.model_versions.pop("ex_reg", None)

    def test_predictions_length(self, fitted_tool, sample_df, tmp_path):
        path = str(tmp_path / "f.xlsx")
        _write_excel(path, self._STATE_ROWS)
        fitted_tool.add_excel_version(path, "Factors", version="ex_len")
        preds = fitted_tool.model_versions["ex_len"].train_predictions
        fitted_tool.model_versions.pop("ex_len", None)
        assert len(preds) == len(sample_df)

    def test_predictions_positive(self, fitted_tool, tmp_path):
        path = str(tmp_path / "f.xlsx")
        _write_excel(path, self._STATE_ROWS)
        fitted_tool.add_excel_version(path, "Factors", version="ex_pos")
        preds = fitted_tool.model_versions["ex_pos"].train_predictions
        fitted_tool.model_versions.pop("ex_pos", None)
        assert (preds > 0).all()

    def test_ae_chart_works(self, fitted_tool, tmp_path):
        path = str(tmp_path / "f.xlsx")
        _write_excel(path, self._STATE_ROWS)
        fitted_tool.add_excel_version(path, "Factors", version="ex_ae")
        try:
            with patch("matplotlib.pyplot.show"):
                fig = fitted_tool.ae_chart("state", version="ex_ae", show=False)
            assert isinstance(fig, matplotlib.figure.Figure)
        finally:
            fitted_tool.model_versions.pop("ex_ae", None)

    def test_compare_models_works(self, fitted_tool, tmp_path):
        path = str(tmp_path / "f.xlsx")
        _write_excel(path, self._STATE_ROWS)
        fitted_tool.add_excel_version(path, "Factors", version="ex_cmp")
        try:
            with patch("matplotlib.pyplot.show"):
                result = fitted_tool.compare_models("v2", "ex_cmp", show=False)
            assert "metrics" in result
            assert "double_lift" in result
        finally:
            fitted_tool.model_versions.pop("ex_cmp", None)

    def test_missing_columns_raises(self, fitted_tool, tmp_path):
        """Excel without Factor column should raise ValueError."""
        import openpyxl
        path = str(tmp_path / "bad.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["Variable", "Level"])   # missing Factor
        ws.append(["state", "CA"])
        wb.save(path)
        with pytest.raises(ValueError, match="Factor"):
            fitted_tool.add_excel_version(path, "Sheet1", version="bad_ver")

    def test_unknown_variable_raises(self, fitted_tool, tmp_path):
        """Variable in Excel not in the data should raise ValueError."""
        path = str(tmp_path / "unknown.xlsx")
        _write_excel(path, [["ghost_col", "X", 1.0]])
        with pytest.raises(ValueError):
            fitted_tool.add_excel_version(path, "Factors", version="bad_var")

    def test_model_summary_raises_for_excel_version(self, fitted_tool, tmp_path):
        path = str(tmp_path / "f.xlsx")
        _write_excel(path, self._STATE_ROWS)
        fitted_tool.add_excel_version(path, "Factors", version="ex_ms")
        try:
            with pytest.raises(TypeError, match="Excel factor model"):
                fitted_tool.model_summary("ex_ms")
        finally:
            fitted_tool.model_versions.pop("ex_ms", None)

    def test_unseen_level_gets_neutral_factor(self, fitted_tool, sample_df, tmp_path):
        """Rows whose level isn't in the table should not crash; factor = 1.0."""
        # Only include 3 of the 5 state levels; the other 2 → missing_factor=1.0
        partial_rows = [
            ["state", "CA (base)", 1.0],
            ["state", "TX",        1.1],
            ["state", "FL",        0.9],
        ]
        path = str(tmp_path / "partial.xlsx")
        _write_excel(path, partial_rows)
        fitted_tool.add_excel_version(
            path, "Factors", version="ex_unseen", missing_factor=1.0
        )
        preds = fitted_tool.model_versions["ex_unseen"].train_predictions
        fitted_tool.model_versions.pop("ex_unseen", None)
        # Predictions still computed for all rows
        assert len(preds) == len(sample_df)
        assert (preds > 0).all()

    def test_intercept_row_applied(self, fitted_tool, sample_df, tmp_path):
        """An intercept row should scale all predictions by its Factor."""
        rows_no_icept = self._STATE_ROWS[:]
        rows_with_icept = self._STATE_ROWS[:] + [["intercept", "intercept", 2.0]]

        p1 = str(tmp_path / "no_icept.xlsx")
        p2 = str(tmp_path / "with_icept.xlsx")
        _write_excel(p1, rows_no_icept)
        _write_excel(p2, rows_with_icept)

        fitted_tool.add_excel_version(p1, "Factors", version="ex_ni")
        fitted_tool.add_excel_version(p2, "Factors", version="ex_wi")
        preds_ni = fitted_tool.model_versions["ex_ni"].train_predictions.copy()
        preds_wi = fitted_tool.model_versions["ex_wi"].train_predictions.copy()
        fitted_tool.model_versions.pop("ex_ni", None)
        fitted_tool.model_versions.pop("ex_wi", None)

        np.testing.assert_allclose(preds_wi, preds_ni * 2.0, rtol=1e-6)

    def test_load_from_excel_standalone(self, sample_df, tmp_path):
        """Standalone classmethod with no pkl: direct string lookup on 'state'."""
        path = str(tmp_path / "raw.xlsx")
        _write_excel(path, self._STATE_ROWS_RAW)
        tool = ModelingTool.load_from_excel(
            excel_path=path,
            sheet_name="Factors",
            data=sample_df,
            target_col="loss_ratio",
            weight_col="earned_premium",
            version="excel",
        )
        assert "excel" in tool.model_versions
        preds = tool.model_versions["excel"].train_predictions
        assert len(preds) == len(sample_df)
        assert (preds > 0).all()

    def test_load_from_excel_with_pkl(self, fitted_tool, sample_df, tmp_path):
        """Classmethod with pkl_path: preprocessor-based level resolution."""
        pkl_path = str(tmp_path / "v2.pkl")
        fitted_tool.save("v2", pkl_path)

        excel_path = str(tmp_path / "factors.xlsx")
        _write_excel(excel_path, self._STATE_ROWS)

        tool2 = ModelingTool.load_from_excel(
            excel_path=excel_path,
            sheet_name="Factors",
            data=sample_df,
            target_col="loss_ratio",
            weight_col="earned_premium",
            pkl_path=pkl_path,
            version="excel",
        )
        assert "excel" in tool2.model_versions
        preds = tool2.model_versions["excel"].train_predictions
        assert len(preds) == len(sample_df)
        assert (preds > 0).all()

    def test_base_version_param_uses_named_preprocessor(self, fitted_tool, sample_df, tmp_path):
        """base_version= explicitly pins which version's preprocessor is used."""
        path = str(tmp_path / "f.xlsx")
        _write_excel(path, self._STATE_ROWS)
        fitted_tool.add_excel_version(
            path, "Factors", version="ex_bv", base_version="v2"
        )
        preds = fitted_tool.model_versions["ex_bv"].train_predictions
        fitted_tool.model_versions.pop("ex_bv", None)
        assert len(preds) == len(sample_df)
