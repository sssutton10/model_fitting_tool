"""
Panel-based GUI for the elastic net ModelingTool.

Launch with::

    panel serve run_gui.py          # or
    python run_gui.py               # auto-opens browser
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import polars as pl

from .bin_suggestor import _plot_suggestions
from .plots import (
    bootstrap_ci_plot,
    cv_stability_plot,
    importance_plot,
    interaction_heatmap,
    overfitting_plot,
    pd_plot_2d,
    regularization_path_plot,
    relativities_ci_plot,
)
from .tool import ModelingTool
from .variable import VariableConfig

# ── Constants ────────────────────────────────────────────────────────────────

LINK_OPTIONS = ["log", "identity"]
DROP_REF_OPTIONS = ["max_weight", "first"]
IMPUTE_OPTIONS = ["None", "median", "mean", "most_frequent", "constant"]
ENCODING_OPTIONS = ["auto", "onehot", "None"]

_NONE_OPTION = "— None —"


def _to_none(val: str) -> Optional[str]:
    """Convert widget string values to None where appropriate."""
    if val in ("None", _NONE_OPTION, "", None):
        return None
    return val


def _parse_float_text(text: Optional[str]) -> Optional[float]:
    """Parse a text input into a float; blank means None, '0' means 0.0.

    Raises ValueError on non-numeric input so callers surface it in the
    status alert.
    """
    text = (text or "").strip()
    if not text:
        return None
    return float(text)


def _parse_list_float(text: str) -> Optional[List[float]]:
    """Parse comma-separated text into a list of floats, or None if empty."""
    text = text.strip()
    if not text:
        return None
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError:
        return None


def _compile_transform(code_str: str) -> Optional[Callable]:
    """Compile a user-provided expression string into a transform callable.

    The expression should reference ``df`` (a polars DataFrame) and return an
    array-like.  Example: ``df["age"] * df["vehicle_age"]``
    """
    code_str = code_str.strip()
    if not code_str:
        return None
    namespace: Dict[str, Any] = {"pl": pl, "np": np}
    exec(
        f"def _user_transform(df, **kwargs):\n    return {code_str}",
        namespace,
    )
    return namespace["_user_transform"]


def _safe_figure(fig: Optional[plt.Figure]) -> Optional[plt.Figure]:
    """Return *fig* or None; ensures we never pass garbage to Matplotlib pane."""
    if fig is None:
        return None
    if not isinstance(fig, plt.Figure):
        return None
    return fig


def _close_fig(fig: Optional[plt.Figure]) -> None:
    """Close a matplotlib figure to free memory."""
    if fig is not None and isinstance(fig, plt.Figure):
        plt.close(fig)


# ── Shared application state ────────────────────────────────────────────────

class ModelingApp(param.Parameterized):
    """Root shared state object.  Tabs mutate ``tool`` then call ``bump()``."""

    tool_version = param.Integer(default=0, doc="Incremented on every tool mutation")
    data_columns = param.List(default=[], doc="All column names")
    numeric_columns = param.List(default=[], doc="Numeric column names")
    variable_names = param.List(default=[], doc="Registered variable names")
    model_version_names = param.List(default=[], doc="Fitted model version names")

    def __init__(self, **params):
        super().__init__(**params)
        self.tool: Optional[ModelingTool] = None
        self.data: Optional[pl.DataFrame] = None

    def bump(self):
        """Signal that tool state has changed."""
        if self.tool is not None:
            self.variable_names = list(self.tool.variable_configs.keys())
            self.model_version_names = list(self.tool.model_versions.keys())
        self.tool_version += 1


# ── Tab 1: Data ─────────────────────────────────────────────────────────────

class DataTab(param.Parameterized):
    """Load data, configure target/weight, create ModelingTool."""

    app = param.ClassSelector(class_=ModelingApp)

    def __init__(self, **params):
        super().__init__(**params)

        # --- widgets ---
        self.file_path = pn.widgets.TextInput(
            name="File path (CSV or Parquet)",
            placeholder="C:/data/training.csv",
            width=500,
        )
        self.load_btn = pn.widgets.Button(name="Load File", button_type="primary")
        self.load_btn.on_click(self._on_load)

        self.target_col = pn.widgets.Select(name="Target column", options=[], width=250)
        self.weight_col = pn.widgets.Select(
            name="Weight column", options=[_NONE_OPTION], width=250,
        )
        self.link = pn.widgets.Select(name="Link", options=LINK_OPTIONS, value="log", width=120)
        self.tweedie_power = pn.widgets.FloatInput(
            name="Tweedie power", value=1.5, step=0.1, start=0.0, end=3.0, width=120,
        )
        self.cv_column = pn.widgets.Select(
            name="CV column", options=[_NONE_OPTION], width=200,
        )
        self.drop_ref = pn.widgets.Select(
            name="Drop reference", options=DROP_REF_OPTIONS, value="max_weight", width=160,
        )
        self.create_btn = pn.widgets.Button(
            name="Create ModelingTool", button_type="success", disabled=True,
        )
        self.create_btn.on_click(self._on_create)

        self.status = pn.pane.Alert("Load a CSV or Parquet file to begin.", alert_type="info")
        self.preview = pn.widgets.Tabulator(
            disabled=True, page_size=50, sizing_mode="stretch_both",
        )

        # --- explore widgets ---
        self.explore_col = pn.widgets.Select(name="Column", options=[], width=250)
        self.explore_col.param.watch(self._on_explore, "value")
        self.explore_status = pn.pane.Markdown("Load a file to explore columns.")
        self.value_counts_table = pn.widgets.Tabulator(
            disabled=True, page_size=50, sizing_mode="stretch_both",
        )
        self.stats_table = pn.widgets.Tabulator(
            disabled=True, page_size=50, sizing_mode="stretch_both",
        )

    # --- callbacks ---

    def _on_load(self, event):
        path = self.file_path.value.strip()
        if not path:
            self.status.object = "Please enter a file path."
            self.status.alert_type = "warning"
            return
        try:
            if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
                df = pl.read_parquet(path)
            else:
                df = pl.read_csv(path, infer_schema_length=10_000)
            self.app.data = df

            cols = df.columns
            num_cols = [
                c for c in cols
                if df[c].dtype in (
                    pl.Float32, pl.Float64,
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                )
            ]
            self.app.data_columns = cols
            self.app.numeric_columns = num_cols

            self.target_col.options = num_cols
            self.weight_col.options = [_NONE_OPTION] + num_cols
            self.cv_column.options = [_NONE_OPTION] + cols
            self.explore_col.options = cols

            self.preview.value = df.head(100).to_pandas()
            self.create_btn.disabled = False
            self.status.object = f"Loaded {len(df):,} rows x {len(cols)} columns from {path}"
            self.status.alert_type = "success"
        except Exception as exc:
            self.status.object = f"Error loading file: {exc}"
            self.status.alert_type = "danger"

    def _on_create(self, event):
        if self.app.data is None:
            self.status.object = "Load a file first."
            self.status.alert_type = "warning"
            return
        try:
            target = self.target_col.value
            weight = _to_none(self.weight_col.value)
            cv_col = _to_none(self.cv_column.value)

            self.app.tool = ModelingTool(
                data=self.app.data,
                target_col=target,
                weight_col=weight,
                link=self.link.value,
                tweedie_power=self.tweedie_power.value,
                drop_reference=self.drop_ref.value,
                cv_column=cv_col,
            )
            self.app.bump()
            self.status.object = (
                f"ModelingTool created.  Target: {target}, "
                f"Weight: {weight or 'None'}, Link: {self.link.value}"
            )
            self.status.alert_type = "success"
        except Exception as exc:
            self.status.object = f"Error creating tool: {exc}"
            self.status.alert_type = "danger"

    def _on_explore(self, event):
        df = self.app.data
        col = self.explore_col.value
        if df is None or not col or col not in df.columns:
            return
        try:
            n_rows = len(df)
            s = df[col]
            n_unique = s.n_unique()
            n_null = s.null_count()

            vc = s.value_counts(sort=True).with_columns(
                (pl.col("count") / n_rows * 100).round(2).alias("percent")
            )
            truncated = len(vc) > 50
            self.value_counts_table.value = vc.head(50).to_pandas()

            stats = s.describe().to_pandas()
            stats.loc[len(stats)] = ["n_unique", n_unique]
            self.stats_table.value = stats

            msg = f"**{col}** — {n_rows:,} rows, {n_unique:,} unique, {n_null:,} null"
            if truncated:
                msg += f" — showing top 50 of {n_unique:,} values"
            self.explore_status.object = msg
        except Exception as exc:
            self.explore_status.object = f"Error exploring column: {exc}"

    # --- layout ---

    def panel(self):
        config_col = pn.Column(
            "## 1. Load Data",
            self.file_path,
            self.load_btn,
            pn.layout.Divider(),
            "## 2. Configure",
            pn.Row(self.target_col, self.weight_col),
            pn.Row(self.link, self.tweedie_power),
            pn.Row(self.cv_column, self.drop_ref),
            self.create_btn,
            self.status,
            width=560,
        )
        explore_panel = pn.Column(
            self.explore_col,
            self.explore_status,
            pn.Row(
                pn.Column("#### Value Counts", self.value_counts_table, sizing_mode="stretch_both"),
                pn.Column("#### Summary Stats", self.stats_table, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )
        return pn.Row(
            config_col,
            pn.Tabs(
                ("Preview", pn.Column("### Data Preview", self.preview, sizing_mode="stretch_both")),
                ("Explore", explore_panel),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )


# ── Tab 2: Variables ────────────────────────────────────────────────────────

class VariablesTab(param.Parameterized):
    """Add/configure variables, view univariate plots, suggest bins."""

    app = param.ClassSelector(class_=ModelingApp)

    def __init__(self, **params):
        super().__init__(**params)

        # --- config form widgets ---
        self.col_select = pn.widgets.Select(name="Column", options=[], width=220)
        self.is_categorical = pn.widgets.Checkbox(name="Is categorical", value=False)
        self.cap_lower = pn.widgets.TextInput(name="Cap lower (blank=off)", value="", width=120)
        self.cap_upper = pn.widgets.TextInput(name="Cap upper (blank=off)", value="", width=120)
        self.log_transform = pn.widgets.Checkbox(name="Log transform", value=False)
        self.impute_strategy = pn.widgets.Select(
            name="Impute strategy", options=IMPUTE_OPTIONS, value="median", width=150,
        )
        self.impute_value = pn.widgets.TextInput(
            name="Impute value (if constant)", value="", width=140,
        )
        self.n_bins = pn.widgets.IntInput(name="N bins", value=0, start=0, step=1, width=100)
        self.bin_edges_text = pn.widgets.TextInput(
            name="Bin edges (comma-separated)", placeholder="25, 45, 65", width=250,
        )
        self.encoding = pn.widgets.Select(
            name="Encoding", options=ENCODING_OPTIONS, value="auto", width=120,
        )
        self.standardize = pn.widgets.Checkbox(name="Standardize", value=False)
        self.degree = pn.widgets.IntInput(
            name="Degree (poly)", value=1, start=1, step=1, width=100,
        )
        self.transform_kwargs_text = pn.widgets.TextInput(
            name="Transform kwargs (JSON)", placeholder='{"factor": 1000}', width=250,
        )
        self.input_cols_text = pn.widgets.TextInput(
            name="Input columns (comma-separated, for derived vars)",
            placeholder="col_a, col_b",
            width=300,
        )
        self.custom_transform_text = pn.widgets.TextAreaInput(
            name="Custom transform (Python expression)",
            placeholder='df["col_a"] * df["col_b"]',
            height=70,
            width=400,
        )
        self.add_btn = pn.widgets.Button(name="Add / Update Variable", button_type="primary")
        self.add_btn.on_click(self._on_add_variable)

        # --- univariate plot ---
        self.plot_btn = pn.widgets.Button(name="Univariate Plot", button_type="default")
        self.plot_btn.on_click(self._on_univariate_plot)
        self.uni_version = pn.widgets.Select(
            name="Use version's binning", options=[_NONE_OPTION], width=180,
        )
        self.plot_pane = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")

        # --- bin suggestion ---
        self.bin_methods = pn.widgets.CheckBoxGroup(
            name="Methods", options=["quantile", "equal_width", "gbm", "optbin"],
            value=["quantile", "equal_width"],
            inline=True,
        )
        self.bin_n = pn.widgets.IntInput(name="N bins (suggestion)", value=10, start=2, width=100)
        self.suggest_btn = pn.widgets.Button(name="Suggest Bins", button_type="default")
        self.suggest_btn.on_click(self._on_suggest_bins)
        self.suggest_result = pn.pane.Str("", sizing_mode="stretch_width")
        self.suggest_plot_pane = pn.pane.Matplotlib(
            None, tight=True, dpi=96, sizing_mode="scale_width",
        )

        # --- variable cards area ---
        # FlexBox wraps the fixed-width cards into a crisp responsive grid
        # instead of a single tall stacked column.
        self.cards_area = pn.FlexBox(
            sizing_mode="stretch_width", gap="12px", margin=(5, 0),
        )

        # --- status ---
        self.status = pn.pane.Alert("", alert_type="info", visible=False)

        # --- react to tool changes ---
        self.app.param.watch(self._on_tool_changed, ["tool_version"])

    # --- callbacks ---

    def _on_tool_changed(self, event):
        if self.app.tool is None:
            return
        self.col_select.options = self.app.data_columns
        self.uni_version.options = [_NONE_OPTION] + self.app.model_version_names
        self._rebuild_cards()

    def _build_config_kwargs(self) -> dict:
        """Gather widget values into kwargs for add_variable."""
        kwargs: dict = {}

        cap_lo = _parse_float_text(self.cap_lower.value)
        if cap_lo is not None:
            kwargs["cap_lower"] = cap_lo
        cap_hi = _parse_float_text(self.cap_upper.value)
        if cap_hi is not None:
            kwargs["cap_upper"] = cap_hi

        if self.log_transform.value:
            kwargs["log_transform"] = True

        imp = _to_none(self.impute_strategy.value)
        kwargs["impute_strategy"] = imp
        if imp == "constant" and self.impute_value.value.strip():
            kwargs["impute_value"] = self.impute_value.value.strip()

        n = self.n_bins.value
        if n and n > 0:
            kwargs["n_bins"] = int(n)

        edges = _parse_list_float(self.bin_edges_text.value)
        if edges:
            kwargs["bin_edges"] = edges

        enc = _to_none(self.encoding.value)
        if enc:
            kwargs["encoding"] = enc

        if self.standardize.value:
            kwargs["standardize"] = True

        if self.degree.value and self.degree.value > 1:
            kwargs["degree"] = int(self.degree.value)

        tk_text = self.transform_kwargs_text.value.strip()
        if tk_text:
            kwargs["transform_kwargs"] = json.loads(tk_text)

        if self.is_categorical.value:
            kwargs["is_categorical"] = True

        return kwargs

    def _on_add_variable(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first (Data tab).", "warning")
            return

        col = self.col_select.value
        if not col:
            self._show_status("Select a column.", "warning")
            return

        try:
            kwargs = self._build_config_kwargs()

            # input_cols
            input_cols_text = self.input_cols_text.value.strip()
            input_cols = None
            if input_cols_text:
                input_cols = [c.strip() for c in input_cols_text.split(",") if c.strip()]

            # custom transform
            ct = _compile_transform(self.custom_transform_text.value)

            self.app.tool.add_variable(
                col, input_cols=input_cols, custom_transform=ct, **kwargs,
            )
            self.app.bump()
            self._show_status(f"Variable '{col}' added/updated.", "success")
        except Exception as exc:
            self._show_status(f"Error: {exc}", "danger")

    def _on_univariate_plot(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        col = self.col_select.value
        if not col:
            return
        try:
            fig = self.app.tool.univariate_plot(
                col, show=False, version=_to_none(self.uni_version.value),
            )
            self.plot_pane.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"Plot error: {exc}", "danger")

    def _on_suggest_bins(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        col = self.col_select.value
        if not col:
            return
        try:
            methods = tuple(self.bin_methods.value) if self.bin_methods.value else ("quantile",)
            # Capture stdout from suggest_bins and suppress plt.show
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = self.app.tool.suggest_bins(
                    col, methods=methods, n_bins=self.bin_n.value, show_plot=False,
                )
            lines = []
            for method, edges in result.items():
                edges_str = ", ".join(f"{e:.4g}" for e in edges)
                lines.append(f"**{method}**: [{edges_str}]")
            self.suggest_result.object = "\n".join(lines)

            fig = _plot_suggestions(
                col, self.app.tool.data, result, weights=self.app.tool._weights,
            )
            self.suggest_plot_pane.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"Bin suggestion error: {exc}", "danger")

    def _rebuild_cards(self):
        """Rebuild the variable card list from current tool state."""
        if self.app.tool is None:
            self.cards_area.clear()
            return

        cards = []
        for col, cfg in self.app.tool.variable_configs.items():
            info_parts = []
            if cfg.is_categorical:
                info_parts.append("categorical")
            if cfg.n_bins:
                info_parts.append(f"bins={cfg.n_bins}")
            if cfg.bin_edges:
                info_parts.append(f"edges={len(cfg.bin_edges)} breaks")
            if cfg.cap_lower is not None:
                info_parts.append(f"cap_lo={cfg.cap_lower}")
            if cfg.cap_upper is not None:
                info_parts.append(f"cap_hi={cfg.cap_upper}")
            if cfg.log_transform:
                info_parts.append("log")
            if cfg.encoding and cfg.encoding != "auto":
                info_parts.append(f"enc={cfg.encoding}")
            if cfg.input_cols:
                info_parts.append(f"inputs={cfg.input_cols}")
            if cfg.custom_transform is not None:
                info_parts.append("custom_fn")

            info_str = " | ".join(info_parts) if info_parts else "default config"

            remove_btn = pn.widgets.Button(
                name="Remove", button_type="danger", width=80,
            )
            remove_btn.on_click(lambda evt, c=col: self._remove_variable(c))

            edit_btn = pn.widgets.Button(
                name="Edit", button_type="default", width=60,
            )
            edit_btn.on_click(lambda evt, c=col: self._edit_variable(c))

            card = pn.Card(
                pn.pane.Str(info_str),
                pn.Row(edit_btn, remove_btn),
                title=col,
                collapsed=True,
                width=350,
            )
            cards.append(card)

        self.cards_area.objects = cards

    def _remove_variable(self, col: str):
        if self.app.tool and col in self.app.tool.variable_configs:
            del self.app.tool.variable_configs[col]
            self.app.bump()
            self._show_status(f"Variable '{col}' removed.", "info")

    def _edit_variable(self, col: str):
        """Pre-fill the config form with an existing variable's settings."""
        if self.app.tool is None or col not in self.app.tool.variable_configs:
            return
        cfg = self.app.tool.variable_configs[col]
        self.col_select.value = cfg.col
        self.is_categorical.value = bool(cfg.is_categorical)
        self.cap_lower.value = str(cfg.cap_lower) if cfg.cap_lower is not None else ""
        self.cap_upper.value = str(cfg.cap_upper) if cfg.cap_upper is not None else ""
        self.log_transform.value = cfg.log_transform
        self.impute_strategy.value = str(cfg.impute_strategy) if cfg.impute_strategy else "None"
        self.impute_value.value = str(cfg.impute_value) if cfg.impute_value is not None else ""
        self.n_bins.value = cfg.n_bins if cfg.n_bins else 0
        self.bin_edges_text.value = (
            ", ".join(str(e) for e in cfg.bin_edges) if cfg.bin_edges else ""
        )
        self.encoding.value = str(cfg.encoding) if cfg.encoding else "None"
        self.standardize.value = cfg.standardize
        self.degree.value = cfg.degree if cfg.degree else 1
        self.transform_kwargs_text.value = (
            json.dumps(cfg.transform_kwargs) if cfg.transform_kwargs else ""
        )
        self.input_cols_text.value = (
            ", ".join(cfg.input_cols) if cfg.input_cols else ""
        )
        # Can't reverse-engineer custom_transform source code; clear it
        self.custom_transform_text.value = ""

    def _show_status(self, msg: str, alert_type: str = "info"):
        self.status.object = msg
        self.status.alert_type = alert_type
        self.status.visible = True

    # --- layout ---

    def panel(self):
        config_form = pn.Card(
            self.col_select,
            self.is_categorical,
            pn.Row(self.cap_lower, self.cap_upper),
            self.log_transform,
            pn.Row(self.impute_strategy, self.impute_value),
            pn.Row(self.n_bins, self.encoding),
            self.bin_edges_text,
            pn.Row(self.standardize, self.degree),
            self.transform_kwargs_text,
            self.input_cols_text,
            self.custom_transform_text,
            self.add_btn,
            title="Variable Configuration",
            width=440,
            margin=(0, 0, 12, 0),
        )

        bin_section = pn.Card(
            self.bin_methods,
            self.bin_n,
            self.suggest_btn,
            self.suggest_result,
            self.suggest_plot_pane,
            title="Bin Suggestion",
            width=440,
            margin=(0, 0, 12, 0),
        )

        left = pn.Column(config_form, bin_section, width=460, margin=(0, 16, 0, 0))

        right = pn.Column(
            pn.Row(self.plot_btn, self.uni_version),
            self.plot_pane,
            sizing_mode="stretch_width",
        )

        # stretch_width (not stretch_both): the top region takes its natural
        # height so the Registered Variables section below always stacks
        # cleanly beneath it instead of overlapping.
        top = pn.Row(left, right, sizing_mode="stretch_width")

        return pn.Column(
            self.status,
            top,
            pn.layout.Divider(margin=(20, 0, 10, 0)),
            "### Registered Variables",
            self.cards_area,
            sizing_mode="stretch_width",
        )


# ── Tab 3: Model ────────────────────────────────────────────────────────────

class ModelTab(param.Parameterized):
    """Fit models, view summaries, coefficients, relativities."""

    app = param.ClassSelector(class_=ModelingApp)

    def __init__(self, **params):
        super().__init__(**params)

        # --- fitting controls ---
        self.var_select = pn.widgets.MultiSelect(
            name="Variables", options=[], size=8, width=250,
        )
        self.version_name = pn.widgets.TextInput(
            name="Version name", value="v1", width=150,
        )
        self.alpha = pn.widgets.TextInput(
            name="Alpha (blank=CV)", value="", width=120,
        )
        self.l1_ratio = pn.widgets.FloatInput(
            name="L1 ratio", value=0.5, step=0.1, start=0.0, end=1.0, width=120,
        )
        self.use_cv = pn.widgets.Checkbox(name="Use CV", value=True)
        self.cv_folds = pn.widgets.IntInput(
            name="CV folds", value=5, start=2, step=1, width=80,
        )
        self.max_iter = pn.widgets.IntInput(
            name="Max iterations", value=1000, start=100, step=100, width=120,
        )
        self.fit_btn = pn.widgets.Button(name="Fit Model", button_type="primary")
        self.fit_btn.on_click(self._on_fit)

        # --- inspection controls ---
        self.version_select = pn.widgets.Select(
            name="Inspect version", options=[], width=180,
        )
        self.version_select.param.watch(self._on_version_selected, ["value"])
        self.top_n = pn.widgets.IntInput(
            name="Top N coefficients", value=30, start=5, step=5, width=120,
        )
        self.refresh_btn = pn.widgets.Button(name="Refresh", button_type="default")
        self.refresh_btn.on_click(self._on_refresh)

        # --- display panes ---
        self.versions_table = pn.widgets.Tabulator(
            disabled=True, page_size=20, sizing_mode="stretch_width",
        )
        self.summary_table = pn.widgets.Tabulator(
            disabled=True, page_size=50, sizing_mode="stretch_width",
        )
        self.coef_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")
        self.relat_table = pn.widgets.Tabulator(
            disabled=True, page_size=50, sizing_mode="stretch_width",
        )

        self.fit_container = pn.Column(sizing_mode="stretch_both")
        self.status = pn.pane.Alert("", alert_type="info", visible=False)

        # --- react to tool changes ---
        self.app.param.watch(self._on_tool_changed, ["tool_version"])

    def _on_tool_changed(self, event):
        if self.app.tool is None:
            return
        self.var_select.options = self.app.variable_names
        self.version_select.options = self.app.model_version_names
        self._refresh_versions_table()

    def _on_fit(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first (Data tab).", "warning")
            return
        variables = list(self.var_select.value)
        if not variables:
            self._show_status("Select at least one variable.", "warning")
            return

        version = self.version_name.value.strip()
        if not version:
            self._show_status("Enter a version name.", "warning")
            return

        self.fit_container.loading = True
        try:
            alpha = _parse_float_text(self.alpha.value)
            cv_val = int(self.cv_folds.value) if self.use_cv.value else None

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.app.tool.fit_model(
                    variables=variables,
                    version=version,
                    alpha=alpha,
                    l1_ratio=float(self.l1_ratio.value),
                    use_cv=self.use_cv.value,
                    cv=cv_val,
                    max_iter=int(self.max_iter.value),
                    print_summary=False,
                )
            self.app.bump()
            self.version_select.value = version
            self._show_status(f"Model '{version}' fitted successfully.", "success")
        except Exception as exc:
            self._show_status(f"Fit error: {exc}", "danger")
        finally:
            self.fit_container.loading = False

    def _on_version_selected(self, event):
        self._refresh_inspection()

    def _on_refresh(self, event):
        self._refresh_inspection()

    def _refresh_versions_table(self):
        if self.app.tool is None or not self.app.tool.model_versions:
            return
        try:
            df = self.app.tool.list_versions()
            self.versions_table.value = df.to_pandas()
        except Exception:
            pass

    def _refresh_inspection(self):
        """Update summary, coefficient plot, and relativities for selected version."""
        version = self.version_select.value
        if not version or self.app.tool is None:
            return
        if version not in self.app.tool.model_versions:
            return

        # Summary
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                summary_df = self.app.tool.model_summary(version)
            self.summary_table.value = summary_df.to_pandas()
        except Exception:
            pass

        # Coefficient plot
        try:
            fig = self.app.tool.coefficient_plot(
                version, top_n=int(self.top_n.value), show=False,
            )
            self.coef_plot.object = fig
            _close_fig(fig)
        except Exception:
            self.coef_plot.object = None

        # Relativities
        try:
            rel_df = self.app.tool.relativities_table(version)
            self.relat_table.value = rel_df.to_pandas()
        except Exception:
            pass

        self._refresh_versions_table()

    def _show_status(self, msg: str, alert_type: str = "info"):
        self.status.object = msg
        self.status.alert_type = alert_type
        self.status.visible = True

    def panel(self):
        fit_controls = pn.Card(
            self.var_select,
            pn.Row(self.version_name, self.alpha),
            pn.Row(self.l1_ratio, self.max_iter),
            pn.Row(self.use_cv, self.cv_folds),
            self.fit_btn,
            title="Fit Model",
            width=440,
        )

        inspect_controls = pn.Column(
            pn.Row(self.version_select, self.top_n, self.refresh_btn),
        )

        left = pn.Column(
            fit_controls,
            "### Versions",
            self.versions_table,
            width=460,
        )

        right = pn.Column(
            inspect_controls,
            pn.Tabs(
                ("Summary", self.summary_table),
                ("Coefficients", self.coef_plot),
                ("Relativities", self.relat_table),
                dynamic=True,
            ),
            sizing_mode="stretch_both",
        )

        self.fit_container.objects = [
            self.status,
            pn.Row(left, right, sizing_mode="stretch_both"),
        ]
        return self.fit_container


# ── Tab 4: Evaluation ───────────────────────────────────────────────────────

class EvaluationTab(param.Parameterized):
    """A/E and residual charts, model comparison, save/load."""

    app = param.ClassSelector(class_=ModelingApp)

    def __init__(self, **params):
        super().__init__(**params)

        # --- per-variable charts ---
        self.chart_version = pn.widgets.Select(name="Version", options=[], width=150)
        self.chart_col = pn.widgets.Select(name="Column", options=[], width=200)
        self.chart_type = pn.widgets.RadioButtonGroup(
            name="Chart type", options=["Residual", "A/E"], value="Residual",
        )
        self.chart_nbins = pn.widgets.IntInput(
            name="N bins", value=10, start=2, step=1, width=80,
        )
        self.chart_btn = pn.widgets.Button(name="Update Chart", button_type="primary")
        self.chart_btn.on_click(self._on_chart)
        self.chart_pane = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")

        self.plotall_btn = pn.widgets.Button(name="Plot All Variables", button_type="default")
        self.plotall_btn.on_click(self._on_plot_all)
        self.all_charts_area = pn.Column(sizing_mode="stretch_width")

        # --- comparison ---
        self.cmp_v1 = pn.widgets.Select(name="Version 1", options=[], width=150)
        self.cmp_v2 = pn.widgets.Select(name="Version 2", options=[], width=150)
        self.cmp_nbuckets = pn.widgets.IntInput(
            name="N buckets", value=10, start=2, step=1, width=80,
        )
        self.cmp_btn = pn.widgets.Button(name="Compare", button_type="primary")
        self.cmp_btn.on_click(self._on_compare)
        self.cmp_metrics = pn.widgets.Tabulator(
            disabled=True, page_size=20, sizing_mode="stretch_width",
        )
        self.cmp_chart = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")

        # --- A/v/E table ---
        self.ave_vars = pn.widgets.MultiSelect(
            name="Analysis variables", options=[], size=6, width=250,
        )
        self.ave_version = pn.widgets.Select(name="Version", options=[], width=150)
        self.ave_nbins = pn.widgets.IntInput(name="N bins", value=10, start=2, width=80)
        self.ave_btn = pn.widgets.Button(name="A/v/E Table", button_type="primary")
        self.ave_btn.on_click(self._on_ave_table)
        self.ave_table_pane = pn.widgets.Tabulator(
            disabled=True, page_size=50, sizing_mode="stretch_width",
        )

        # --- CV stability ---
        self.cvs_vars = pn.widgets.MultiSelect(
            name="Variables", options=[], size=6, width=250,
        )
        self.cvs_fold_col = pn.widgets.Select(name="Fold column", options=[], width=180)
        self.cvs_version = pn.widgets.Select(
            name="Borrow hyperparams from", options=[_NONE_OPTION], width=200,
        )
        self.cvs_btn = pn.widgets.Button(name="Run CV Stability", button_type="primary")
        self.cvs_btn.on_click(self._on_cv_stability)
        self.cvs_table = pn.widgets.Tabulator(
            disabled=True, page_size=30, sizing_mode="stretch_width",
        )
        self.cvs_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")
        self.cvs_container = pn.Column(sizing_mode="stretch_width")

        # --- Excel workflows ---
        self.xl_add_path = pn.widgets.TextInput(
            name="Excel path (.xlsx)", placeholder="factors.xlsx", width=300,
        )
        self.xl_add_sheet = pn.widgets.TextInput(name="Sheet name", value="Sheet1", width=150)
        self.xl_add_version = pn.widgets.TextInput(name="Version name", value="excel", width=150)
        self.xl_add_missing = pn.widgets.FloatInput(
            name="Missing factor", value=1.0, step=0.05, width=120,
        )
        self.xl_add_base = pn.widgets.Select(
            name="Base version", options=[_NONE_OPTION], width=160,
        )
        self.xl_add_btn = pn.widgets.Button(name="Add Excel Version", button_type="primary")
        self.xl_add_btn.on_click(self._on_add_excel_version)

        self.xl_load_path = pn.widgets.TextInput(
            name="Excel path (.xlsx)", placeholder="factors.xlsx", width=300,
        )
        self.xl_load_sheet = pn.widgets.TextInput(name="Sheet name", value="Sheet1", width=150)
        self.xl_load_target = pn.widgets.Select(name="Target column", options=[], width=180)
        self.xl_load_weight = pn.widgets.Select(
            name="Weight column", options=[_NONE_OPTION], width=180,
        )
        self.xl_load_pkl = pn.widgets.TextInput(
            name="Pkl path (optional)", placeholder="models/v1.pkl", width=250,
        )
        self.xl_load_btn = pn.widgets.Button(
            name="Load Tool from Excel", button_type="warning",
        )
        self.xl_load_btn.on_click(self._on_load_from_excel)

        self.frozen_path = pn.widgets.TextInput(
            name="Frozen pkl path", placeholder="models/v1.pkl", width=300,
        )
        self.frozen_btn = pn.widgets.Button(name="Load Frozen", button_type="warning")
        self.frozen_btn.on_click(self._on_load_frozen)

        # --- save/load ---
        self.save_version = pn.widgets.Select(name="Version to save", options=[], width=150)
        self.save_path = pn.widgets.TextInput(
            name="Save path", placeholder="models/v1.pkl", width=300,
        )
        self.save_btn = pn.widgets.Button(name="Save", button_type="success")
        self.save_btn.on_click(self._on_save)

        self.load_path = pn.widgets.TextInput(
            name="Load path (.pkl)", placeholder="models/v1.pkl", width=300,
        )
        self.load_btn = pn.widgets.Button(name="Load", button_type="warning")
        self.load_btn.on_click(self._on_load)

        self.status = pn.pane.Alert("", alert_type="info", visible=False)

        # --- react ---
        self.app.param.watch(self._on_tool_changed, ["tool_version"])

    def _on_tool_changed(self, event):
        if self.app.tool is None:
            return
        versions = self.app.model_version_names
        self.chart_version.options = versions
        self.cmp_v1.options = versions
        self.cmp_v2.options = versions
        self.save_version.options = versions
        self.chart_col.options = self.app.data_columns
        self.ave_vars.options = self.app.data_columns
        self.ave_version.options = versions
        self.cvs_vars.options = self.app.variable_names
        self.cvs_fold_col.options = self.app.data_columns
        self.cvs_version.options = [_NONE_OPTION] + versions
        self.xl_add_base.options = [_NONE_OPTION] + versions
        self.xl_load_target.options = self.app.numeric_columns
        self.xl_load_weight.options = [_NONE_OPTION] + self.app.numeric_columns

    def _on_chart(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        version = self.chart_version.value
        col = self.chart_col.value
        if not version or not col:
            self._show_status("Select a version and column.", "warning")
            return
        try:
            if self.chart_type.value == "A/E":
                fig = self.app.tool.ae_chart(
                    col, version, n_bins=int(self.chart_nbins.value), show=False,
                )
            else:
                fig = self.app.tool.residual_chart(
                    col, version, n_bins=int(self.chart_nbins.value), show=False,
                )
            self.chart_pane.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"Chart error: {exc}", "danger")

    def _on_plot_all(self, event):
        if self.app.tool is None:
            return
        version = self.chart_version.value
        if not version:
            return
        try:
            chart_key = "ae" if self.chart_type.value == "A/E" else "residual"
            figs = self.app.tool.plot_all_variables(
                version, chart=chart_key, show=False,
            )
            panes = []
            for fig in figs:
                panes.append(pn.pane.Matplotlib(fig, tight=True, dpi=96, sizing_mode="scale_width"))
                _close_fig(fig)
            self.all_charts_area.objects = panes
        except Exception as exc:
            self._show_status(f"Plot all error: {exc}", "danger")

    def _on_compare(self, event):
        if self.app.tool is None:
            return
        v1, v2 = self.cmp_v1.value, self.cmp_v2.value
        if not v1 or not v2:
            self._show_status("Select two versions to compare.", "warning")
            return
        try:
            # compare_models creates the double-lift figure internally but never
            # returns it; capture it by diffing matplotlib's figure registry.
            fignums_before = set(plt.get_fignums())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = self.app.tool.compare_models(
                    v1, v2, n_buckets=int(self.cmp_nbuckets.value), show=False,
                )
            new_nums = [n for n in plt.get_fignums() if n not in fignums_before]

            self.cmp_metrics.value = result["metrics"].to_pandas()
            self.cmp_chart.object = plt.figure(new_nums[-1]) if new_nums else None
            for n in new_nums:
                plt.close(plt.figure(n))

            interp = [
                ln.strip() for ln in buf.getvalue().splitlines()
                if "double_lift_score interpretation" in ln
            ]
            if interp:
                self._show_status(interp[0], "info")
        except Exception as exc:
            self._show_status(f"Compare error: {exc}", "danger")

    def _on_ave_table(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        variables = list(self.ave_vars.value)
        version = self.ave_version.value
        if not variables or not version:
            self._show_status("Select analysis variables and a version.", "warning")
            return
        try:
            df = self.app.tool.ave_table(
                variables, version, n_bins=int(self.ave_nbins.value),
            )
            self.ave_table_pane.value = df.to_pandas()
        except Exception as exc:
            self._show_status(f"A/v/E error: {exc}", "danger")

    def _on_cv_stability(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        variables = list(self.cvs_vars.value)
        fold_col = self.cvs_fold_col.value
        if not variables or not fold_col:
            self._show_status("Select variables and a fold column.", "warning")
            return
        self.cvs_container.loading = True
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                df = self.app.tool.fit_cv_stability(
                    variables,
                    fold_col,
                    version=_to_none(self.cvs_version.value),
                    plot=False,
                )
            self.cvs_table.value = df.to_pandas()
            fig = cv_stability_plot(df)
            self.cvs_plot.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"CV stability error: {exc}", "danger")
        finally:
            self.cvs_container.loading = False

    def _on_add_excel_version(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        path = self.xl_add_path.value.strip()
        if not path:
            self._show_status("Enter an Excel file path.", "warning")
            return
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.app.tool.add_excel_version(
                    path,
                    self.xl_add_sheet.value.strip(),
                    version=self.xl_add_version.value.strip() or "excel",
                    missing_factor=float(self.xl_add_missing.value),
                    base_version=_to_none(self.xl_add_base.value),
                )
            self.app.bump()
            self._show_status(
                f"Excel version '{self.xl_add_version.value}' added.", "success",
            )
        except Exception as exc:
            self._show_status(f"Add Excel version error: {exc}", "danger")

    def _on_load_from_excel(self, event):
        path = self.xl_load_path.value.strip()
        if not path:
            self._show_status("Enter an Excel file path.", "warning")
            return
        if self.app.data is None:
            self._show_status("Load data first (Data tab).", "warning")
            return
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tool = ModelingTool.load_from_excel(
                    path,
                    self.xl_load_sheet.value.strip(),
                    data=self.app.data,
                    target_col=self.xl_load_target.value,
                    weight_col=_to_none(self.xl_load_weight.value),
                    pkl_path=_to_none(self.xl_load_pkl.value.strip()),
                )
            self.app.tool = tool
            self.app.bump()
            self._show_status(f"Tool loaded from Excel: {path}", "success")
        except Exception as exc:
            self._show_status(f"Load from Excel error: {exc}", "danger")

    def _on_load_frozen(self, event):
        path = self.frozen_path.value.strip()
        if not path:
            self._show_status("Enter a frozen pkl path.", "warning")
            return
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tool = ModelingTool.load_frozen(path)
            self.app.tool = tool
            self.app.data = tool.data
            self.app.bump()
            self._show_status(
                "Frozen tool loaded — prediction-only: it has no training data, "
                "so charts, metrics, and refitting will error. To work with data, "
                "use Load (.pkl) instead.",
                "warning",
            )
        except Exception as exc:
            self._show_status(f"Load frozen error: {exc}", "danger")

    def _on_save(self, event):
        if self.app.tool is None:
            return
        version = self.save_version.value
        path = self.save_path.value.strip()
        if not version or not path:
            self._show_status("Select a version and enter a save path.", "warning")
            return
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.app.tool.save(version, path)
            self._show_status(f"Version '{version}' saved to {path}", "success")
        except Exception as exc:
            self._show_status(f"Save error: {exc}", "danger")

    def _on_load(self, event):
        path = self.load_path.value.strip()
        if not path:
            self._show_status("Enter a .pkl file path.", "warning")
            return
        if self.app.data is None:
            self._show_status("Load data first (Data tab).", "warning")
            return
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tool = ModelingTool.load(path, data=self.app.data)
            self.app.tool = tool
            self.app.bump()
            self._show_status(f"Model loaded from {path} as version 'v1'.", "success")
        except Exception as exc:
            self._show_status(f"Load error: {exc}", "danger")

    def _show_status(self, msg: str, alert_type: str = "info"):
        self.status.object = msg
        self.status.alert_type = alert_type
        self.status.visible = True

    def panel(self):
        charts_section = pn.Card(
            pn.Row(self.chart_version, self.chart_col, self.chart_type, self.chart_nbins),
            pn.Row(self.chart_btn, self.plotall_btn),
            self.chart_pane,
            self.all_charts_area,
            title="Variable Charts",
        )

        compare_section = pn.Card(
            pn.Row(self.cmp_v1, self.cmp_v2, self.cmp_nbuckets, self.cmp_btn),
            self.cmp_metrics,
            self.cmp_chart,
            title="Model Comparison",
        )

        ave_section = pn.Card(
            pn.Row(self.ave_vars, pn.Column(self.ave_version, self.ave_nbins, self.ave_btn)),
            self.ave_table_pane,
            title="A/v/E Table",
            collapsed=True,
        )

        self.cvs_container.objects = [
            pn.Row(self.cvs_vars, pn.Column(self.cvs_fold_col, self.cvs_version, self.cvs_btn)),
            self.cvs_table,
            self.cvs_plot,
        ]
        cvs_section = pn.Card(
            self.cvs_container,
            title="CV Stability (refits once per fold)",
            collapsed=True,
        )

        excel_section = pn.Card(
            "#### Add Excel factor version to current tool",
            pn.Row(self.xl_add_path, self.xl_add_sheet),
            pn.Row(self.xl_add_version, self.xl_add_missing, self.xl_add_base),
            self.xl_add_btn,
            pn.layout.Divider(),
            "#### Load a new tool from an Excel factor table",
            pn.Row(self.xl_load_path, self.xl_load_sheet),
            pn.Row(self.xl_load_target, self.xl_load_weight),
            pn.Row(self.xl_load_pkl, self.xl_load_btn),
            pn.layout.Divider(),
            "#### Load frozen model (prediction-only, no data)",
            pn.Row(self.frozen_path, self.frozen_btn),
            title="Excel Workflows",
            collapsed=True,
        )

        save_load_section = pn.Card(
            pn.Row(self.save_version, self.save_path, self.save_btn),
            pn.layout.Divider(),
            pn.Row(self.load_path, self.load_btn),
            title="Save / Load",
        )

        return pn.Column(
            self.status,
            charts_section,
            compare_section,
            ave_section,
            cvs_section,
            excel_section,
            save_load_section,
            sizing_mode="stretch_both",
        )


# ── Tab 5: Diagnostics ──────────────────────────────────────────────────────

class DiagnosticsTab(param.Parameterized):
    """VIF, residual heatmap, regularization path, overfitting, bootstrap CIs."""

    app = param.ClassSelector(class_=ModelingApp)

    def __init__(self, **params):
        super().__init__(**params)

        self.version = pn.widgets.Select(name="Version", options=[], width=180)

        # --- VIF ---
        self.vif_btn = pn.widgets.Button(name="VIF Table", button_type="primary")
        self.vif_btn.on_click(self._on_vif)
        self.vif_table = pn.widgets.Tabulator(
            disabled=True, page_size=50, sizing_mode="stretch_width",
        )

        # --- residual heatmap ---
        self.rh_col1 = pn.widgets.Select(name="Variable 1", options=[], width=180)
        self.rh_col2 = pn.widgets.Select(name="Variable 2", options=[], width=180)
        self.rh_nbins = pn.widgets.IntInput(name="N bins", value=8, start=2, width=80)
        self.rh_btn = pn.widgets.Button(name="Residual Heatmap", button_type="primary")
        self.rh_btn.on_click(self._on_residual_heatmap)
        self.rh_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")

        # --- regularization path ---
        self.rp_l1_ratio = pn.widgets.FloatInput(
            name="L1 ratio", value=0.5, step=0.1, start=0.0, end=1.0, width=100,
        )
        self.rp_n_alphas = pn.widgets.IntInput(
            name="N alphas (each = one refit)", value=20, start=5, width=180,
        )
        self.rp_btn = pn.widgets.Button(name="Regularization Path", button_type="primary")
        self.rp_btn.on_click(self._on_reg_path)
        self.rp_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")
        self.rp_container = pn.Column(sizing_mode="stretch_width")

        # --- overfitting monitor ---
        self.of_versions = pn.widgets.MultiSelect(
            name="Versions", options=[], size=5, width=250,
        )
        self.of_btn = pn.widgets.Button(name="Overfitting Monitor", button_type="primary")
        self.of_btn.on_click(self._on_overfitting)
        self.of_table = pn.widgets.Tabulator(
            disabled=True, page_size=20, sizing_mode="stretch_width",
        )
        self.of_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")
        self.of_container = pn.Column(sizing_mode="stretch_width")

        # --- bootstrap metrics ---
        self.bm_nboot = pn.widgets.IntInput(name="N bootstrap", value=200, start=10, width=120)
        self.bm_ci = pn.widgets.FloatInput(
            name="CI", value=0.95, step=0.01, start=0.5, end=0.999, width=100,
        )
        self.bm_btn = pn.widgets.Button(name="Bootstrap Metrics", button_type="primary")
        self.bm_btn.on_click(self._on_bootstrap_metrics)
        self.bm_table = pn.widgets.Tabulator(
            disabled=True, page_size=20, sizing_mode="stretch_width",
        )
        self.bm_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")
        self.bm_container = pn.Column(sizing_mode="stretch_width")

        # --- bootstrap relativities ---
        self.br_nboot = pn.widgets.IntInput(name="N bootstrap", value=50, start=10, width=120)
        self.br_ci = pn.widgets.FloatInput(
            name="CI", value=0.95, step=0.01, start=0.5, end=0.999, width=100,
        )
        self.br_btn = pn.widgets.Button(name="Bootstrap Relativities", button_type="primary")
        self.br_btn.on_click(self._on_bootstrap_relativities)
        self.br_table = pn.widgets.Tabulator(
            disabled=True, page_size=50, sizing_mode="stretch_width",
        )
        self.br_var = pn.widgets.Select(name="CI plot variable", options=[], width=180)
        self.br_var.param.watch(self._on_br_var_selected, ["value"])
        self.br_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")
        self.br_container = pn.Column(sizing_mode="stretch_width")
        self._br_result: Optional[pl.DataFrame] = None

        self.status = pn.pane.Alert("", alert_type="info", visible=False)

        self.app.param.watch(self._on_tool_changed, ["tool_version"])

    def _on_tool_changed(self, event):
        if self.app.tool is None:
            return
        versions = self.app.model_version_names
        self.version.options = versions
        self.of_versions.options = versions
        self.rh_col1.options = self.app.data_columns
        self.rh_col2.options = self.app.data_columns

    def _require_version(self) -> Optional[str]:
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first (Data tab).", "warning")
            return None
        version = self.version.value
        if not version:
            self._show_status("Select a model version.", "warning")
            return None
        return version

    def _on_vif(self, event):
        version = self._require_version()
        if version is None:
            return
        try:
            self.vif_table.value = self.app.tool.vif_table(version).to_pandas()
        except Exception as exc:
            self._show_status(f"VIF error: {exc}", "danger")

    def _on_residual_heatmap(self, event):
        version = self._require_version()
        if version is None:
            return
        col1, col2 = self.rh_col1.value, self.rh_col2.value
        if not col1 or not col2:
            self._show_status("Select two variables.", "warning")
            return
        try:
            fig, _df = self.app.tool.residual_heatmap(
                version, col1, col2, n_bins=int(self.rh_nbins.value), show=False,
            )
            self.rh_plot.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"Residual heatmap error: {exc}", "danger")

    def _on_reg_path(self, event):
        version = self._require_version()
        if version is None:
            return
        self.rp_container.loading = True
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                df = self.app.tool.regularization_path(
                    version=version,
                    l1_ratio=float(self.rp_l1_ratio.value),
                    n_alphas=int(self.rp_n_alphas.value),
                    show=False,
                )
            fig = regularization_path_plot(df)
            self.rp_plot.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"Regularization path error: {exc}", "danger")
        finally:
            self.rp_container.loading = False

    def _on_overfitting(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        versions = list(self.of_versions.value)
        if not versions:
            self._show_status("Select at least one version.", "warning")
            return
        self.of_container.loading = True
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                df = self.app.tool.overfitting_monitor(versions, show=False)
            self.of_table.value = df.to_pandas()
            fig = overfitting_plot(df)
            self.of_plot.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"Overfitting monitor error: {exc}", "danger")
        finally:
            self.of_container.loading = False

    def _on_bootstrap_metrics(self, event):
        version = self._require_version()
        if version is None:
            return
        self.bm_container.loading = True
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                df = self.app.tool.bootstrap_metrics(
                    version,
                    n_bootstrap=int(self.bm_nboot.value),
                    ci=float(self.bm_ci.value),
                    show=False,
                )
            self.bm_table.value = df.to_pandas()
            fig = bootstrap_ci_plot(df, title=f"Bootstrap CIs — {version}")
            self.bm_plot.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"Bootstrap metrics error: {exc}", "danger")
        finally:
            self.bm_container.loading = False

    def _on_bootstrap_relativities(self, event):
        version = self._require_version()
        if version is None:
            return
        self.br_container.loading = True
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                df = self.app.tool.bootstrap_relativities(
                    version,
                    n_bootstrap=int(self.br_nboot.value),
                    ci=float(self.br_ci.value),
                    show=False,
                )
            self._br_result = df
            self.br_table.value = df.to_pandas()
            variables = df["variable"].unique(maintain_order=True).to_list()
            self.br_var.options = variables
            if variables:
                self.br_var.value = variables[0]
                self._render_br_plot(variables[0])
        except Exception as exc:
            self._show_status(f"Bootstrap relativities error: {exc}", "danger")
        finally:
            self.br_container.loading = False

    def _on_br_var_selected(self, event):
        if self._br_result is not None and event.new:
            self._render_br_plot(event.new)

    def _render_br_plot(self, variable: str):
        try:
            fig = relativities_ci_plot(self._br_result, variable)
            self.br_plot.object = fig
            _close_fig(fig)
        except Exception as exc:
            self._show_status(f"CI plot error: {exc}", "danger")

    def _show_status(self, msg: str, alert_type: str = "info"):
        self.status.object = msg
        self.status.alert_type = alert_type
        self.status.visible = True

    def panel(self):
        vif_card = pn.Card(self.vif_btn, self.vif_table, title="VIF (Multicollinearity)")

        rh_card = pn.Card(
            pn.Row(self.rh_col1, self.rh_col2, self.rh_nbins, self.rh_btn),
            self.rh_plot,
            title="Residual Heatmap",
            collapsed=True,
        )

        self.rp_container.objects = [
            pn.Row(self.rp_l1_ratio, self.rp_n_alphas, self.rp_btn),
            self.rp_plot,
        ]
        rp_card = pn.Card(
            self.rp_container,
            title="Regularization Path (slow: refits per alpha)",
            collapsed=True,
        )

        self.of_container.objects = [
            pn.Row(self.of_versions, self.of_btn),
            self.of_table,
            self.of_plot,
        ]
        of_card = pn.Card(self.of_container, title="Overfitting Monitor", collapsed=True)

        self.bm_container.objects = [
            pn.Row(self.bm_nboot, self.bm_ci, self.bm_btn),
            self.bm_table,
            self.bm_plot,
        ]
        bm_card = pn.Card(self.bm_container, title="Bootstrap Metric CIs", collapsed=True)

        self.br_container.objects = [
            pn.Row(self.br_nboot, self.br_ci, self.br_btn),
            self.br_table,
            pn.Row(self.br_var),
            self.br_plot,
        ]
        br_card = pn.Card(
            self.br_container,
            title="Bootstrap Relativity CIs (slow: refits per resample)",
            collapsed=True,
        )

        return pn.Column(
            self.status,
            pn.Row(self.version),
            vif_card,
            rh_card,
            rp_card,
            of_card,
            bm_card,
            br_card,
            sizing_mode="stretch_both",
        )


# ── Tab 6: Discovery ────────────────────────────────────────────────────────

class DiscoveryTab(param.Parameterized):
    """Shadow GBM diagnostics: importance, interactions, SHAP, Boruta, etc."""

    app = param.ClassSelector(class_=ModelingApp)

    def __init__(self, **params):
        super().__init__(**params)

        # --- step 1: shadow GBM ---
        self.gbm_features = pn.widgets.MultiSelect(
            name="Feature columns (empty = auto)", options=[], size=6, width=250,
        )
        self.gbm_n_estimators = pn.widgets.IntInput(
            name="N estimators", value=200, start=10, width=120,
        )
        self.gbm_max_depth = pn.widgets.IntInput(name="Max depth", value=5, start=1, width=100)
        self.gbm_lr = pn.widgets.FloatInput(
            name="Learning rate", value=0.05, step=0.01, start=0.001, width=120,
        )
        self.gbm_fit_btn = pn.widgets.Button(name="Fit Shadow GBM", button_type="primary")
        self.gbm_fit_btn.on_click(self._on_fit_gbm)
        self.gbm_badge = pn.pane.Markdown("**Shadow GBM:** not fitted")
        self.gbm_container = pn.Column(sizing_mode="stretch_width")

        # --- step 2: shadow-dependent diagnostics ---
        self.perm_btn = pn.widgets.Button(
            name="Permutation Importance", button_type="primary", disabled=True,
        )
        self.perm_btn.on_click(self._on_perm_importance)
        self.shap_imp_btn = pn.widgets.Button(
            name="SHAP Importance", button_type="primary", disabled=True,
        )
        self.shap_imp_btn.on_click(self._on_shap_importance)
        self.imp_table = pn.widgets.Tabulator(
            disabled=True, page_size=30, sizing_mode="stretch_width",
        )
        self.imp_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")
        self.imp_container = pn.Column(sizing_mode="stretch_width")

        self.int_method = pn.widgets.Select(
            name="Method",
            options=["H-statistic", "SHAP interactions", "Tree co-occurrence"],
            width=180,
        )
        self.int_top_n = pn.widgets.IntInput(name="Top N", value=15, start=2, width=80)
        self.int_btn = pn.widgets.Button(
            name="Rank Interactions", button_type="primary", disabled=True,
        )
        self.int_btn.on_click(self._on_interactions)
        self.int_table = pn.widgets.Tabulator(
            disabled=True, page_size=30, sizing_mode="stretch_width",
        )
        self.int_plot = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")
        self.int_container = pn.Column(sizing_mode="stretch_width")

        self.pd_var1 = pn.widgets.Select(name="Variable 1 (numeric)", options=[], width=180)
        self.pd_var2 = pn.widgets.Select(name="Variable 2 (numeric)", options=[], width=180)
        self.pd_btn = pn.widgets.Button(
            name="2-D Partial Dependence", button_type="primary", disabled=True,
        )
        self.pd_btn.on_click(self._on_pd2d)
        self.pd_plot_pane = pn.pane.Matplotlib(
            None, tight=True, dpi=96, sizing_mode="scale_width",
        )

        self.shap_dep_var = pn.widgets.Select(name="Variable", options=[], width=180)
        self.shap_dep_color = pn.widgets.Select(
            name="Color by", options=[_NONE_OPTION], width=180,
        )
        self.shap_dep_btn = pn.widgets.Button(
            name="SHAP Dependence", button_type="primary", disabled=True,
        )
        self.shap_dep_btn.on_click(self._on_shap_dependence)
        self.shap_dep_plot = pn.pane.Matplotlib(
            None, tight=True, dpi=96, sizing_mode="scale_width",
        )

        self._gated_buttons = [
            self.perm_btn, self.shap_imp_btn, self.int_btn,
            self.pd_btn, self.shap_dep_btn,
        ]

        # --- step 3: standalone diagnostics ---
        self.grp_col = pn.widgets.Select(name="Categorical column", options=[], width=200)
        self.grp_max = pn.widgets.IntInput(name="Max groups", value=10, start=2, width=100)
        self.grp_btn = pn.widgets.Button(name="Suggest Category Groups", button_type="primary")
        self.grp_btn.on_click(self._on_category_groups)
        self.grp_table = pn.widgets.Tabulator(
            disabled=True, page_size=30, sizing_mode="stretch_width",
        )
        self.grp_mapping = pn.pane.JSON(None, depth=2, sizing_mode="stretch_width")

        self.mono_var = pn.widgets.Select(name="Variable (numeric)", options=[], width=180)
        self.mono_btn = pn.widgets.Button(name="Monotonicity Test", button_type="primary")
        self.mono_btn.on_click(self._on_monotonicity)
        self.mono_result = pn.pane.Markdown("")

        self.boruta_iters = pn.widgets.IntInput(name="Iterations", value=20, start=5, width=100)
        self.boruta_btn = pn.widgets.Button(name="Boruta Select", button_type="primary")
        self.boruta_btn.on_click(self._on_boruta)
        self.boruta_table = pn.widgets.Tabulator(
            disabled=True, page_size=30, sizing_mode="stretch_width",
        )
        self.boruta_container = pn.Column(sizing_mode="stretch_width")

        self.rgbm_version = pn.widgets.Select(name="GLM version", options=[], width=180)
        self.rgbm_btn = pn.widgets.Button(name="Residual GBM", button_type="primary")
        self.rgbm_btn.on_click(self._on_residual_gbm)
        self.rgbm_table = pn.widgets.Tabulator(
            disabled=True, page_size=30, sizing_mode="stretch_width",
        )

        self.status = pn.pane.Alert("", alert_type="info", visible=False)

        self.app.param.watch(self._on_tool_changed, ["tool_version"])

    def _on_tool_changed(self, event):
        if self.app.tool is None:
            return
        cols = self.app.data_columns
        num_cols = self.app.numeric_columns
        self.gbm_features.options = cols
        self.pd_var1.options = num_cols
        self.pd_var2.options = num_cols
        self.shap_dep_var.options = cols
        self.shap_dep_color.options = [_NONE_OPTION] + cols
        self.grp_col.options = cols
        self.mono_var.options = num_cols
        self.rgbm_version.options = self.app.model_version_names
        self._sync_gated_buttons()

    def _sync_gated_buttons(self):
        fitted = (
            self.app.tool is not None
            and getattr(self.app.tool, "_shadow_model", None) is not None
        )
        for btn in self._gated_buttons:
            btn.disabled = not fitted
        self.gbm_badge.object = (
            "**Shadow GBM:** fitted" if fitted else "**Shadow GBM:** not fitted"
        )

    def _run(self, label: str, fn, *args, **kwargs):
        """Run a discovery call with stdout redirected; show errors in the alert."""
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                return fn(*args, **kwargs)
        except ImportError as exc:
            self._show_status(f"{label}: missing optional dependency — {exc}", "warning")
        except RuntimeError as exc:
            self._show_status(f"{label}: {exc}", "warning")
        except Exception as exc:
            self._show_status(f"{label} error: {exc}", "danger")
        return None

    def _on_fit_gbm(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first (Data tab).", "warning")
            return
        features = list(self.gbm_features.value) or None
        self.gbm_container.loading = True
        try:
            result = self._run(
                "Shadow GBM",
                self.app.tool.fit_shadow_gbm,
                feature_cols=features,
                n_estimators=int(self.gbm_n_estimators.value),
                max_depth=int(self.gbm_max_depth.value),
                learning_rate=float(self.gbm_lr.value),
            )
            if result is not None:
                self._show_status("Shadow GBM fitted.", "success")
        finally:
            self.gbm_container.loading = False
            self._sync_gated_buttons()

    def _on_perm_importance(self, event):
        df = self._run(
            "Permutation importance", self.app.tool.permutation_importance, version=None,
        )
        if df is None:
            return
        self.imp_table.value = df.to_pandas()
        fig = importance_plot(df, title="Permutation Importance")
        self.imp_plot.object = fig
        _close_fig(fig)

    def _on_shap_importance(self, event):
        self.imp_container.loading = True
        try:
            df = self._run("SHAP importance", self.app.tool.shap_importance)
            if df is None:
                return
            self.imp_table.value = df.to_pandas()
            fig = importance_plot(df, title="SHAP Importance")
            self.imp_plot.object = fig
            _close_fig(fig)
        finally:
            self.imp_container.loading = False

    def _on_interactions(self, event):
        method = self.int_method.value
        top_n = int(self.int_top_n.value)
        self.int_container.loading = True
        try:
            if method == "H-statistic":
                df = self._run(
                    "Interaction ranking", self.app.tool.interaction_ranking, top_n=top_n,
                )
                score_col = "h_statistic"
            elif method == "SHAP interactions":
                df = self._run(
                    "SHAP interactions",
                    self.app.tool.shap_interaction_ranking,
                    top_n=top_n,
                )
                score_col = "interaction_strength"
            else:
                df = self._run(
                    "Tree co-occurrence",
                    self.app.tool.tree_interaction_cooccurrence,
                    top_n=top_n,
                )
                score_col = "cooccurrence_score"
            if df is None:
                return
            self.int_table.value = df.to_pandas()
            # interaction_heatmap expects the score in an 'h_statistic' column
            heat_df = df if score_col == "h_statistic" else df.rename(
                {score_col: "h_statistic"}
            )
            fig = interaction_heatmap(heat_df, top_n=top_n)
            self.int_plot.object = fig
            _close_fig(fig)
        finally:
            self.int_container.loading = False

    def _on_pd2d(self, event):
        var1, var2 = self.pd_var1.value, self.pd_var2.value
        if not var1 or not var2:
            self._show_status("Select two numeric variables.", "warning")
            return
        df = self._run(
            "Partial dependence", self.app.tool.partial_dependence_2d, var1, var2,
        )
        if df is None:
            return
        fig = pd_plot_2d(df, var1, var2)
        self.pd_plot_pane.object = fig
        _close_fig(fig)

    def _on_shap_dependence(self, event):
        var = self.shap_dep_var.value
        if not var:
            self._show_status("Select a variable.", "warning")
            return
        color_var = _to_none(self.shap_dep_color.value)
        df = self._run(
            "SHAP dependence", self.app.tool.shap_dependence, var, color_var=color_var,
        )
        if df is None:
            return
        # No plots.py helper for this one — build a simple scatter inline.
        fig, ax = plt.subplots(figsize=(8, 5))
        x = df[var].to_numpy()
        y = df["shap_value"].to_numpy()
        if color_var is not None and color_var in df.columns:
            sc = ax.scatter(x, y, c=df[color_var].to_numpy(), s=12, alpha=0.6, cmap="viridis")
            fig.colorbar(sc, ax=ax, label=color_var)
        else:
            ax.scatter(x, y, s=12, alpha=0.6)
        ax.set_xlabel(var)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence — {var}")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        self.shap_dep_plot.object = fig
        _close_fig(fig)

    def _on_category_groups(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        col = self.grp_col.value
        if not col:
            self._show_status("Select a column.", "warning")
            return
        result = self._run(
            "Category grouping",
            self.app.tool.suggest_category_groups,
            col,
            max_groups=int(self.grp_max.value),
        )
        if result is None:
            return
        mapping, summary_df = result
        self.grp_table.value = summary_df.to_pandas()
        self.grp_mapping.object = mapping

    def _on_monotonicity(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        var = self.mono_var.value
        if not var:
            self._show_status("Select a variable.", "warning")
            return
        result = self._run("Monotonicity test", self.app.tool.monotonicity_test, var)
        if result is None:
            return
        lines = [f"#### Monotonicity test — {var}", ""]
        for key, val in result.items():
            lines.append(f"- **{key}**: {val}")
        self.mono_result.object = "\n".join(lines)

    def _on_boruta(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        self.boruta_container.loading = True
        try:
            df = self._run(
                "Boruta",
                self.app.tool.boruta_select,
                n_iterations=int(self.boruta_iters.value),
            )
            if df is not None:
                self.boruta_table.value = df.to_pandas()
        finally:
            self.boruta_container.loading = False

    def _on_residual_gbm(self, event):
        if self.app.tool is None:
            self._show_status("Create a ModelingTool first.", "warning")
            return
        version = self.rgbm_version.value
        if not version:
            self._show_status("Select a fitted GLM version.", "warning")
            return
        df = self._run("Residual GBM", self.app.tool.residual_gbm, version)
        if df is not None:
            self.rgbm_table.value = df.to_pandas()

    def _show_status(self, msg: str, alert_type: str = "info"):
        self.status.object = msg
        self.status.alert_type = alert_type
        self.status.visible = True

    def panel(self):
        self.gbm_container.objects = [
            pn.Row(
                self.gbm_features,
                pn.Column(self.gbm_n_estimators, self.gbm_max_depth, self.gbm_lr),
            ),
            pn.Row(self.gbm_fit_btn, self.gbm_badge),
        ]
        gbm_card = pn.Card(
            self.gbm_container,
            title="Step 1 — Fit Shadow GBM (required for the cards below)",
        )

        self.imp_container.objects = [
            pn.Row(self.perm_btn, self.shap_imp_btn),
            self.imp_table,
            self.imp_plot,
        ]
        imp_card = pn.Card(self.imp_container, title="Variable Importance", collapsed=True)

        self.int_container.objects = [
            pn.Row(self.int_method, self.int_top_n, self.int_btn),
            self.int_table,
            self.int_plot,
        ]
        int_card = pn.Card(self.int_container, title="Interaction Ranking", collapsed=True)

        pd_card = pn.Card(
            pn.Row(self.pd_var1, self.pd_var2, self.pd_btn),
            self.pd_plot_pane,
            title="2-D Partial Dependence",
            collapsed=True,
        )

        shap_dep_card = pn.Card(
            pn.Row(self.shap_dep_var, self.shap_dep_color, self.shap_dep_btn),
            self.shap_dep_plot,
            title="SHAP Dependence",
            collapsed=True,
        )

        grp_card = pn.Card(
            pn.Row(self.grp_col, self.grp_max, self.grp_btn),
            self.grp_table,
            self.grp_mapping,
            title="Category Grouping (standalone)",
            collapsed=True,
        )

        mono_card = pn.Card(
            pn.Row(self.mono_var, self.mono_btn),
            self.mono_result,
            title="Monotonicity Test (standalone)",
            collapsed=True,
        )

        self.boruta_container.objects = [
            pn.Row(self.boruta_iters, self.boruta_btn),
            self.boruta_table,
        ]
        boruta_card = pn.Card(
            self.boruta_container,
            title="Boruta Feature Selection (standalone, slow)",
            collapsed=True,
        )

        rgbm_card = pn.Card(
            pn.Row(self.rgbm_version, self.rgbm_btn),
            self.rgbm_table,
            title="Residual GBM (needs a fitted GLM version)",
            collapsed=True,
        )

        return pn.Column(
            self.status,
            gbm_card,
            imp_card,
            int_card,
            pd_card,
            shap_dep_card,
            grp_card,
            mono_card,
            boruta_card,
            rgbm_card,
            sizing_mode="stretch_both",
        )


# ── App assembly ─────────────────────────────────────────────────────────────

def create_app() -> pn.Tabs:
    """Build and return the complete GUI application."""
    pn.extension("tabulator", notifications=True)

    app = ModelingApp()
    data_tab = DataTab(app=app)
    vars_tab = VariablesTab(app=app)
    model_tab = ModelTab(app=app)
    eval_tab = EvaluationTab(app=app)
    diag_tab = DiagnosticsTab(app=app)
    disc_tab = DiscoveryTab(app=app)

    tabs = pn.Tabs(
        ("1. Data", data_tab.panel()),
        ("2. Variables", vars_tab.panel()),
        ("3. Model", model_tab.panel()),
        ("4. Evaluation", eval_tab.panel()),
        ("5. Diagnostics", diag_tab.panel()),
        ("6. Discovery", disc_tab.panel()),
        dynamic=True,
    )

    template = pn.template.FastListTemplate(
        title="Elastic Net Modeling Tool",
        main=[tabs],
        accent_base_color="#2196F3",
        header_background="#1565C0",
    )
    return template


def serve(**kwargs):
    """Launch the GUI server."""
    app = create_app()
    pn.serve({"Modeling Tool": lambda: create_app()}, **kwargs)
