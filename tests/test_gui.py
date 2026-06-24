"""
Headless smoke / regression tests for the Panel GUI (elastic_net_tool/gui.py).

These tests drive the tab callbacks directly (no browser, no server):
widgets are plain Python objects, so setting ``widget.value`` and invoking
``_on_*`` handlers exercises the same code paths as clicks in the UI.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import pytest

pn = pytest.importorskip("panel")

from elastic_net_tool.gui import (  # noqa: E402
    DataTab,
    DiagnosticsTab,
    DiscoveryTab,
    EvaluationTab,
    ModelingApp,
    ModelTab,
    VariablesTab,
    _parse_float_text,
    create_app,
)

_NONE_OPTION = "— None —"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def csv_path(tmp_path_factory, sample_df: pl.DataFrame) -> str:
    path = tmp_path_factory.mktemp("gui_data") / "sample.csv"
    sample_df.write_csv(path)
    return str(path)


@pytest.fixture()
def gui(csv_path: str):
    """All six tabs wired to a shared ModelingApp, with data loaded and tool created."""
    app = ModelingApp()
    tabs = {
        "data": DataTab(app=app),
        "vars": VariablesTab(app=app),
        "model": ModelTab(app=app),
        "eval": EvaluationTab(app=app),
        "diag": DiagnosticsTab(app=app),
        "disc": DiscoveryTab(app=app),
    }
    data = tabs["data"]
    data.file_path.value = csv_path
    data._on_load(None)
    assert app.data is not None, data.status.object

    data.target_col.value = "loss_ratio"
    data.weight_col.value = "earned_premium"
    data._on_create(None)
    assert app.tool is not None, data.status.object
    return app, tabs


@pytest.fixture()
def fitted_gui(gui):
    """GUI with driver_age + state registered and two versions fitted."""
    app, tabs = gui
    vt = tabs["vars"]
    for col in ("driver_age", "state"):
        vt.col_select.value = col
        vt._on_add_variable(None)
        assert col in app.tool.variable_configs, vt.status.object
        vt.cap_lower.value = ""  # reset form between variables

    mt = tabs["model"]
    mt.use_cv.value = False
    mt.alpha.value = "0.01"
    mt.var_select.value = ["driver_age"]
    mt.version_name.value = "v1"
    mt._on_fit(None)
    assert "v1" in app.tool.model_versions, mt.status.object

    mt.var_select.value = ["driver_age", "state"]
    mt.version_name.value = "v2"
    mt._on_fit(None)
    assert "v2" in app.tool.model_versions, mt.status.object
    return app, tabs


# ── Construction smoke tests ─────────────────────────────────────────────────

class TestConstruction:
    def test_create_app_builds_template(self):
        template = create_app()
        assert template is not None

    def test_all_tabs_build_panels(self, gui):
        _app, tabs = gui
        for tab in tabs.values():
            assert tab.panel() is not None


# ── Helper regression: 0 vs blank (bug 2) ────────────────────────────────────

class TestParseFloatText:
    def test_blank_is_none(self):
        assert _parse_float_text("") is None
        assert _parse_float_text("   ") is None
        assert _parse_float_text(None) is None

    def test_zero_is_zero(self):
        assert _parse_float_text("0") == 0.0

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            _parse_float_text("abc")


class TestDataExplore:
    def test_value_counts_categorical(self, gui):
        app, tabs = gui
        dt = tabs["data"]
        dt.explore_col.value = "state"
        dt._on_explore(None)
        vc = dt.value_counts_table.value
        assert vc is not None and len(vc) == 5  # CA, TX, FL, NY, OH
        assert "count" in vc.columns and "percent" in vc.columns
        assert vc["percent"].sum() == pytest.approx(100.0, abs=0.5)
        assert vc["count"].sum() == len(app.data)

    def test_summary_stats_numeric(self, gui):
        _app, tabs = gui
        dt = tabs["data"]
        dt.explore_col.value = "driver_age"
        dt._on_explore(None)
        stats = dt.stats_table.value
        assert stats is not None and len(stats) > 0
        described = set(stats["statistic"])
        assert {"mean", "min", "max", "n_unique"} <= described

    def test_high_cardinality_truncates_to_top_50(self, gui):
        _app, tabs = gui
        dt = tabs["data"]
        dt.explore_col.value = "driver_age"  # 62 unique ages in sample data
        dt._on_explore(None)
        assert len(dt.value_counts_table.value) == 50
        assert "top 50" in dt.explore_status.object

    def test_select_change_triggers_refresh(self, gui):
        _app, tabs = gui
        dt = tabs["data"]
        dt.explore_col.value = "region"  # watcher fires, no explicit call
        vc = dt.value_counts_table.value
        assert vc is not None and len(vc) == 3  # East, West, South

    def test_no_data_is_a_noop(self):
        dt = DataTab(app=ModelingApp())
        dt._on_explore(None)  # must not raise
        assert dt.value_counts_table.value is None or len(dt.value_counts_table.value) == 0


class TestVariablesTab:
    def test_cap_of_zero_is_kept(self, gui):
        app, tabs = gui
        vt = tabs["vars"]
        vt.col_select.value = "driver_age"
        vt.cap_lower.value = "0"
        vt._on_add_variable(None)
        cfg = app.tool.variable_configs["driver_age"]
        assert cfg.cap_lower == 0.0, vt.status.object

    def test_blank_cap_is_none(self, gui):
        app, tabs = gui
        vt = tabs["vars"]
        vt.col_select.value = "driver_age"
        vt.cap_lower.value = ""
        vt._on_add_variable(None)
        assert app.tool.variable_configs["driver_age"].cap_lower is None

    def test_degree_and_transform_kwargs_round_trip(self, gui):
        # Regression for bug 3: degree / transform_kwargs / impute_value
        # had no widgets or were dropped by the Edit button.
        app, tabs = gui
        vt = tabs["vars"]
        vt.col_select.value = "driver_age"
        vt.degree.value = 2
        vt.transform_kwargs_text.value = '{"factor": 10}'
        vt.impute_strategy.value = "constant"
        vt.impute_value.value = "5"
        vt._on_add_variable(None)

        cfg = app.tool.variable_configs["driver_age"]
        assert cfg.degree == 2
        assert cfg.transform_kwargs == {"factor": 10}
        assert cfg.impute_value == "5"

        # Reset form, then Edit must restore everything.
        vt.degree.value = 1
        vt.transform_kwargs_text.value = ""
        vt.impute_value.value = ""
        vt._edit_variable("driver_age")
        assert vt.degree.value == 2
        assert vt.transform_kwargs_text.value == '{"factor": 10}'
        assert vt.impute_value.value == "5"

    def test_edit_restores_blank_caps_as_blank(self, gui):
        app, tabs = gui
        vt = tabs["vars"]
        vt.col_select.value = "driver_age"
        vt._on_add_variable(None)
        vt.cap_lower.value = "999"
        vt._edit_variable("driver_age")
        assert vt.cap_lower.value == ""

    def test_suggest_bins_renders_overlay_figure(self, gui):
        _app, tabs = gui
        vt = tabs["vars"]
        vt.col_select.value = "driver_age"
        vt.bin_methods.value = ["quantile"]
        vt._on_suggest_bins(None)
        assert isinstance(vt.suggest_plot_pane.object, plt.Figure), vt.status.object
        assert "quantile" in vt.suggest_result.object


# ── Fit + evaluation (need real glum) ────────────────────────────────────────

@pytest.mark.requires_glum
class TestModelTab:
    def test_fit_with_alpha_zero(self, gui):
        # Regression for bug 2: alpha "0" used to be coerced to None (=CV).
        app, tabs = gui
        vt, mt = tabs["vars"], tabs["model"]
        vt.col_select.value = "driver_age"
        vt._on_add_variable(None)

        mt.use_cv.value = False
        mt.alpha.value = "0"
        mt.var_select.value = ["driver_age"]
        mt.version_name.value = "v0"
        mt._on_fit(None)
        assert "v0" in app.tool.model_versions, mt.status.object
        assert app.tool.model_versions["v0"].alpha == 0

    def test_inspection_panes_populate(self, fitted_gui):
        _app, tabs = fitted_gui
        mt = tabs["model"]
        mt.version_select.value = "v2"
        mt._refresh_inspection()
        assert mt.summary_table.value is not None
        assert mt.relat_table.value is not None


@pytest.mark.requires_glum
class TestEvaluationTab:
    def test_compare_shows_chart_without_leaking(self, fitted_gui):
        # Regression for bug 1: chart was stubbed to None and the internal
        # double-lift figure leaked on every click.
        _app, tabs = fitted_gui
        et = tabs["eval"]
        et.cmp_v1.value = "v1"
        et.cmp_v2.value = "v2"
        n_figs_before = len(plt.get_fignums())
        et._on_compare(None)
        assert isinstance(et.cmp_chart.object, plt.Figure), et.status.object
        assert et.cmp_metrics.value is not None
        assert len(plt.get_fignums()) == n_figs_before

    def test_charts_render(self, fitted_gui):
        _app, tabs = fitted_gui
        et = tabs["eval"]
        et.chart_version.value = "v1"
        et.chart_col.value = "driver_age"
        et._on_chart(None)
        assert isinstance(et.chart_pane.object, plt.Figure), et.status.object

    def test_ave_table(self, fitted_gui):
        _app, tabs = fitted_gui
        et = tabs["eval"]
        et.ave_vars.value = ["driver_age"]
        et.ave_version.value = "v1"
        et._on_ave_table(None)
        assert et.ave_table_pane.value is not None, et.status.object

    def test_cv_stability(self, fitted_gui):
        _app, tabs = fitted_gui
        et = tabs["eval"]
        et.cvs_vars.value = ["driver_age"]
        et.cvs_fold_col.value = "cv_fold"
        et.cvs_version.value = "v1"
        et._on_cv_stability(None)
        assert et.cvs_table.value is not None, et.status.object
        assert isinstance(et.cvs_plot.object, plt.Figure)

    def test_save_and_load_round_trip(self, fitted_gui, tmp_path):
        app, tabs = fitted_gui
        et = tabs["eval"]
        path = str(tmp_path / "v1.pkl")
        et.save_version.value = "v1"
        et.save_path.value = path
        et._on_save(None)
        assert "saved" in str(et.status.object), et.status.object

        et.load_path.value = path
        et._on_load(None)
        assert app.tool is not None
        assert "v1" in app.tool.model_versions, et.status.object


@pytest.mark.requires_glum
class TestDiagnosticsTab:
    def test_vif_table(self, fitted_gui):
        _app, tabs = fitted_gui
        dt = tabs["diag"]
        dt.version.value = "v2"
        dt._on_vif(None)
        assert dt.vif_table.value is not None, dt.status.object

    def test_residual_heatmap(self, fitted_gui):
        _app, tabs = fitted_gui
        dt = tabs["diag"]
        dt.version.value = "v1"
        dt.rh_col1.value = "driver_age"
        dt.rh_col2.value = "vehicle_value"
        dt._on_residual_heatmap(None)
        assert isinstance(dt.rh_plot.object, plt.Figure), dt.status.object

    def test_bootstrap_metrics(self, fitted_gui):
        _app, tabs = fitted_gui
        dt = tabs["diag"]
        dt.version.value = "v1"
        dt.bm_nboot.value = 10
        dt._on_bootstrap_metrics(None)
        assert dt.bm_table.value is not None, dt.status.object
        assert isinstance(dt.bm_plot.object, plt.Figure)

    def test_overfitting_monitor(self, fitted_gui):
        _app, tabs = fitted_gui
        dt = tabs["diag"]
        dt.of_versions.value = ["v1", "v2"]
        dt._on_overfitting(None)
        assert dt.of_table.value is not None, dt.status.object
        assert isinstance(dt.of_plot.object, plt.Figure)


@pytest.mark.requires_glum
class TestDiscoveryTab:
    def test_buttons_gated_until_gbm_fitted(self, gui):
        pytest.importorskip("lightgbm")
        _app, tabs = gui
        dt = tabs["disc"]
        assert all(btn.disabled for btn in dt._gated_buttons)

        dt.gbm_n_estimators.value = 10
        dt._on_fit_gbm(None)
        assert "fitted" in dt.gbm_badge.object, dt.status.object
        assert all(not btn.disabled for btn in dt._gated_buttons)

    def test_permutation_importance(self, gui):
        pytest.importorskip("lightgbm")
        _app, tabs = gui
        dt = tabs["disc"]
        dt.gbm_n_estimators.value = 10
        dt._on_fit_gbm(None)
        dt._on_perm_importance(None)
        assert dt.imp_table.value is not None, dt.status.object
        assert isinstance(dt.imp_plot.object, plt.Figure)

    def test_tree_cooccurrence_heatmap(self, gui):
        pytest.importorskip("lightgbm")
        _app, tabs = gui
        dt = tabs["disc"]
        dt.gbm_n_estimators.value = 10
        dt._on_fit_gbm(None)
        dt.int_method.value = "Tree co-occurrence"
        dt._on_interactions(None)
        assert dt.int_table.value is not None, dt.status.object

    def test_category_groups(self, gui):
        pytest.importorskip("lightgbm")
        _app, tabs = gui
        dt = tabs["disc"]
        dt.grp_col.value = "state"
        dt._on_category_groups(None)
        assert dt.grp_table.value is not None, dt.status.object
        assert isinstance(dt.grp_mapping.object, dict)
