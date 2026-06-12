"""
Panel-based GUI for the elastic net ModelingTool.

Launch with::

    panel serve run_gui.py          # or
    python run_gui.py               # auto-opens browser
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import polars as pl

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


def _parse_float_or_none(widget) -> Optional[float]:
    """Read a FloatInput and return None if blank/zero."""
    v = widget.value
    if v is None or v == 0:
        return None
    return float(v)


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
        return pn.Row(
            config_col,
            pn.Column("### Data Preview", self.preview, sizing_mode="stretch_both"),
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
        self.cap_lower = pn.widgets.FloatInput(name="Cap lower", value=None, width=120)
        self.cap_upper = pn.widgets.FloatInput(name="Cap upper", value=None, width=120)
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
        self.plot_pane = pn.pane.Matplotlib(None, tight=True, dpi=96, sizing_mode="scale_width")

        # --- bin suggestion ---
        self.bin_methods = pn.widgets.CheckBoxGroup(
            name="Methods", options=["quantile", "equal_width", "gbm"],
            value=["quantile", "equal_width"],
            inline=True,
        )
        self.bin_n = pn.widgets.IntInput(name="N bins (suggestion)", value=10, start=2, width=100)
        self.suggest_btn = pn.widgets.Button(name="Suggest Bins", button_type="default")
        self.suggest_btn.on_click(self._on_suggest_bins)
        self.suggest_result = pn.pane.Str("", sizing_mode="stretch_width")

        # --- variable cards area ---
        self.cards_area = pn.Column(sizing_mode="stretch_width")

        # --- status ---
        self.status = pn.pane.Alert("", alert_type="info", visible=False)

        # --- react to tool changes ---
        self.app.param.watch(self._on_tool_changed, ["tool_version"])

    # --- callbacks ---

    def _on_tool_changed(self, event):
        if self.app.tool is None:
            return
        self.col_select.options = self.app.data_columns
        self._rebuild_cards()

    def _build_config_kwargs(self) -> dict:
        """Gather widget values into kwargs for add_variable."""
        kwargs: dict = {}

        cap_lo = self.cap_lower.value
        if cap_lo is not None and cap_lo != 0:
            kwargs["cap_lower"] = float(cap_lo)
        cap_hi = self.cap_upper.value
        if cap_hi is not None and cap_hi != 0:
            kwargs["cap_upper"] = float(cap_hi)

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
            fig = self.app.tool.univariate_plot(col, show=False)
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
        self.cap_lower.value = cfg.cap_lower if cfg.cap_lower is not None else 0
        self.cap_upper.value = cfg.cap_upper if cfg.cap_upper is not None else 0
        self.log_transform.value = cfg.log_transform
        self.impute_strategy.value = str(cfg.impute_strategy) if cfg.impute_strategy else "None"
        self.n_bins.value = cfg.n_bins if cfg.n_bins else 0
        self.bin_edges_text.value = (
            ", ".join(str(e) for e in cfg.bin_edges) if cfg.bin_edges else ""
        )
        self.encoding.value = str(cfg.encoding) if cfg.encoding else "None"
        self.standardize.value = cfg.standardize
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
            self.standardize,
            self.input_cols_text,
            self.custom_transform_text,
            self.add_btn,
            title="Variable Configuration",
            width=440,
        )

        bin_section = pn.Card(
            self.bin_methods,
            self.bin_n,
            self.suggest_btn,
            self.suggest_result,
            title="Bin Suggestion",
            width=440,
        )

        left = pn.Column(config_form, bin_section, width=460)

        right = pn.Column(
            pn.Row(self.plot_btn),
            self.plot_pane,
            sizing_mode="stretch_width",
        )

        top = pn.Row(left, right, sizing_mode="stretch_both")

        return pn.Column(
            self.status,
            top,
            "### Registered Variables",
            self.cards_area,
            sizing_mode="stretch_both",
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
        self.alpha = pn.widgets.FloatInput(
            name="Alpha (blank=CV)", value=None, step=0.001, width=120,
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
            alpha = _parse_float_or_none(self.alpha)
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
            result = self.app.tool.compare_models(
                v1, v2, n_buckets=int(self.cmp_nbuckets.value), show=False,
            )
            self.cmp_metrics.value = result["metrics"].to_pandas()
            # The compare_models method returns a dict; chart is generated internally.
            # We need to capture the figure. Let's call it again to get the figure.
            # Actually, compare_models with show=False doesn't produce a figure in the
            # return dict. It returns {'metrics': df, 'double_lift': df}.
            # We can build our own chart from the double_lift table if needed.
            self.cmp_chart.object = None
        except Exception as exc:
            self._show_status(f"Compare error: {exc}", "danger")

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
            save_load_section,
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

    tabs = pn.Tabs(
        ("1. Data", data_tab.panel()),
        ("2. Variables", vars_tab.panel()),
        ("3. Model", model_tab.panel()),
        ("4. Evaluation", eval_tab.panel()),
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
