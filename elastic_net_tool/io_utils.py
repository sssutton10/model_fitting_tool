"""Save and load model versions using pickle."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import pickle
import re
import textwrap
import types
from pathlib import Path
from typing import Any, Callable

import polars as pl

_SNAPSHOT_FORMAT_VERSION = 2


def _extract_import_statements(fn: Callable, source: str = "") -> list[str]:
    """
    Inspect the global names referenced by *fn* and return a list of import
    statements needed to re-create those bindings in a fresh namespace.

    Handles both ``import module as alias`` and ``from module import name``
    patterns.  Names that cannot be resolved to an importable object are
    silently skipped.
    """
    stmts: list[str] = []
    fn_globals = getattr(fn, "__globals__", {})
    referenced = set(getattr(fn.__code__, "co_names", []))
    # Names used only in annotations/default expressions are absent from
    # co_names but still must exist when the saved function source is exec'd.
    referenced.update(re.findall(r"\b[A-Za-z_]\w*\b", source))
    # Also capture free variables' globals for nested/closure functions
    for const in fn.__code__.co_consts:
        if isinstance(const, types.CodeType):
            referenced.update(const.co_names)

    for name in referenced:
        if name not in fn_globals:
            continue
        obj = fn_globals[name]
        if isinstance(obj, types.ModuleType):
            mod_name = obj.__name__
            if mod_name != name:
                stmts.append(f"import {mod_name} as {name}")
            else:
                stmts.append(f"import {mod_name}")
        elif callable(obj) and hasattr(obj, "__module__") and hasattr(obj, "__name__"):
            stmts.append(f"from {obj.__module__} import {obj.__name__} as {name}")
    return stmts


def _serialize_custom_transform(fn: Callable) -> dict[str, Any] | None:
    """
    Serialize a callable to a dict containing its source code so it can be
    reconstructed in a fresh Python session without the original definition.

    Returns None if the source cannot be retrieved.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None

    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    freevars = getattr(code, "co_freevars", ())
    if freevars:
        return None

    name = getattr(fn, "__name__", None)
    imports = _extract_import_statements(fn, source)

    if name == "<lambda>":
        # Extract just the lambda expression from the (possibly indented) source line
        m = re.search(r"(lambda\b[^#\n]*)", source)
        if not m:
            return None
        lambda_expr = m.group(1).rstrip().rstrip(",").rstrip()
        return {"type": "lambda", "source": lambda_expr, "imports": imports}
    else:
        return {
            "type": "def",
            "source": textwrap.dedent(source),
            "name": name,
            "imports": imports,
        }


def _deserialize_custom_transform(data: dict[str, Any] | None) -> Callable | None:
    """Reconstruct a callable from its serialized source dict."""
    if data is None:
        return None
    ns: dict[str, Any] = {}
    # Re-establish any third-party / stdlib imports the function relied on
    for stmt in data.get("imports", []):
        try:
            exec(stmt, ns)  # noqa: S102
        except ImportError:
            pass
    if data["type"] == "lambda":
        exec(f"_fn = {data['source']}", ns)  # noqa: S102
        return ns.get("_fn")
    else:
        exec(compile(data["source"], "<saved_transform>", "exec"), ns)  # noqa: S102
        return ns.get(data["name"])


def _clean_preprocessor(preprocessor: Any) -> Any:
    """
    Return a shallow copy of *preprocessor* whose ``configs`` dict has all
    ``custom_transform`` callables replaced with ``None``.

    This makes the object safe to pickle even when the callables are lambdas
    or locally-defined functions that cannot be serialised by reference.
    """
    cleaned = copy.copy(preprocessor)
    cleaned.configs = {
        col: dataclasses.replace(cfg, custom_transform=None)
        if cfg.custom_transform is not None
        else cfg
        for col, cfg in preprocessor.configs.items()
    }
    return cleaned


def _restore_custom_transforms(snapshot: dict[str, Any]) -> None:
    """Mutate *snapshot* in-place, injecting reconstructed custom_transform callables."""
    ts = snapshot.get("tool_settings", {})
    vs = snapshot.get("version", {})
    configs = ts.get("variable_configs", {})
    sources = ts.get("custom_transform_sources", {})
    preprocessor = vs.get("preprocessor")

    for col, src_data in sources.items():
        fn = _deserialize_custom_transform(src_data)
        if fn is None:
            continue
        if col in configs:
            configs[col] = dataclasses.replace(configs[col], custom_transform=fn)
        if preprocessor is not None and col in preprocessor.configs:
            preprocessor.configs[col] = dataclasses.replace(
                preprocessor.configs[col], custom_transform=fn
            )


def _summarize_variable_transformations(
    variable_configs: dict[str, Any],
    transform_sources: dict[str, Any],
) -> dict[str, Any]:
    """Return a JSON-serialisable summary of every variable's transformation pipeline."""
    return {
        column: _summarize_variable_config(column, config, transform_sources)
        for column, config in variable_configs.items()
    }


def _summarize_variable_config(
    column: str,
    config: Any,
    transform_sources: dict[str, Any],
) -> dict[str, Any]:
    """Return the JSON-safe transformation summary for one variable."""
    entry = _base_transform_summary(config)
    entry.update(_encoding_transform_summary(config))
    custom_transform = _custom_transform_summary(
        column,
        config,
        transform_sources,
    )
    if custom_transform is not None:
        entry["custom_transform"] = custom_transform
    if config.transform_kwargs:
        entry["transform_kwargs"] = config.transform_kwargs
    return entry


def _base_transform_summary(config: Any) -> dict[str, Any]:
    """Summarize dependencies, caps, logging, and imputation."""
    entry: dict[str, Any] = {}
    if config.input_cols:
        entry["input_cols"] = config.input_cols
    if config.cap_lower is not None:
        entry["cap_lower"] = config.cap_lower
    if config.cap_upper is not None:
        entry["cap_upper"] = config.cap_upper
    if config.log_transform:
        entry["log_transform"] = True
    if config.impute_strategy:
        entry["impute_strategy"] = config.impute_strategy
        if config.impute_value is not None:
            entry["impute_value"] = config.impute_value
    return entry


def _encoding_transform_summary(config: Any) -> dict[str, Any]:
    """Summarize binning, scaling, and polynomial options."""
    entry: dict[str, Any] = {}
    if config.n_bins:
        entry["n_bins"] = config.n_bins
    if config.bin_edges:
        entry["bin_edges"] = list(config.bin_edges)
    if config.standardize:
        entry["standardize"] = True
    if config.degree != 1:
        entry["degree"] = config.degree
    return entry


def _custom_transform_summary(
    column: str,
    config: Any,
    transform_sources: dict[str, Any],
) -> str | None:
    """Return the saved source or display name for a custom transform."""
    if column in transform_sources:
        return transform_sources[column]["source"]
    if config.custom_transform is not None:
        return getattr(
            config.custom_transform,
            "__name__",
            "<function>",
        )
    return None


def _make_snapshot(version, tool) -> dict[str, Any]:
    """Package a ModelVersion (or FactorModelVersion) and tool settings into a serialisable dict."""
    safe_configs: dict[str, Any] = {}
    transform_sources: dict[str, Any] = {}
    preprocessor = getattr(version, "preprocessor", None)

    # The fitted preprocessor contains the exact dependency closure used by
    # the model. Include every one of those configs, not just emitted model
    # variables, so chained derived variables can be rebuilt after loading.
    relevant_configs: dict[str, Any] = {}
    if preprocessor is not None:
        relevant_configs.update(preprocessor.configs)
    for col in version.variables:
        if col not in relevant_configs and col in tool.variable_configs:
            relevant_configs[col] = tool.variable_configs[col]

    for col, cfg in relevant_configs.items():
        if cfg.custom_transform is not None:
            src = _serialize_custom_transform(cfg.custom_transform)
            if src is None:
                raise ValueError(
                    f"Cannot save model because custom_transform for variable '{col}' "
                    "has no retrievable source code. Use a named function defined in "
                    "a Python module, then refit and save again."
                )
            transform_sources[col] = src
            safe_configs[col] = dataclasses.replace(cfg, custom_transform=None)
        else:
            safe_configs[col] = cfg

    is_factor = hasattr(version, "factor_table")
    if is_factor:
        cleaned_prep = (
            _clean_preprocessor(version.preprocessor)
            if version.preprocessor is not None
            else None
        )
        version_dict: dict[str, Any] = {
            "name": version.name,
            "variables": version.variables,
            "factor_table": version.factor_table,
            "preprocessor": cleaned_prep,
            "preprocessor_vars": version.preprocessor_vars,
            "offset_col": version.offset_col,
            "fit_info": version.fit_info,
        }
    else:
        version_dict = {
            "name": version.name,
            "variables": version.variables,
            "preprocessor": _clean_preprocessor(version.preprocessor),
            "glm": version.glm,
            "feature_names": version.feature_names,
            "coefficients": version.coefficients,
            "alpha": version.alpha,
            "l1_ratio": version.l1_ratio,
            "family": version.family,
            "link": version.link,
            "fit_info": version.fit_info,
            "tweedie_power": version.tweedie_power,
            "gradient_tol": getattr(version, "gradient_tol", None),
            "cv_stability": getattr(version, "cv_stability", None),
        }

    return {
        "format_version": _SNAPSHOT_FORMAT_VERSION,
        "version_type": "factor" if is_factor else "glm",
        "version": version_dict,
        "tool_settings": {
            "target_col": tool.target_col,
            "weight_col": tool.weight_col,
            "offset_col": getattr(tool, "offset_col", None),
            "link": tool.link,
            "drop_reference": getattr(tool, "drop_reference", "max_weight"),
            "variable_configs": safe_configs,
            "custom_transform_sources": transform_sources,
        },
    }


def save_version(version, tool, filepath: str) -> None:
    """
    Serialise a model version (and its tool settings) to disk.

    Parameters
    ----------
    version : ModelVersion
    tool : ModelingTool
    filepath : str
        Destination ``.pkl`` file path.
    """
    snapshot = _make_snapshot(version, tool)
    is_factor = snapshot["version_type"] == "factor"

    # Human-readable sidecar for quick reference when browsing saved files
    if is_factor:
        json_metadata: dict[str, Any] = {
            "name": version.name,
            "version_type": "factor",
            "variables": version.variables,
            "fit_info": version.fit_info,
            "target_col": tool.target_col,
            "weight_col": tool.weight_col,
            "offset_col": getattr(tool, "offset_col", None),
        }
    else:
        json_metadata = {
            "name": version.name,
            "version_type": "glm",
            "variables": version.variables,
            "family": version.family.__class__.__name__,
            "link": version.link.__class__.__name__,
            "fit_info": version.fit_info,
            "target_col": tool.target_col,
            "weight_col": tool.weight_col,
            "offset_col": getattr(tool, "offset_col", None),
            "l1_ratio": version.l1_ratio,
            "alpha": version.alpha,
            "variable_transformations": _summarize_variable_transformations(
                snapshot["tool_settings"]["variable_configs"],
                snapshot["tool_settings"]["custom_transform_sources"],
            ),
        }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    json_path = path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_metadata, f, indent=2)

    print(f"Saved version '{version.name}' → {path}")


def load_version(
    filepath: str,
    data: pl.DataFrame | None = None,
    refit: bool = True,
) -> dict[str, Any]:
    """
    Load a saved snapshot from disk.

    Parameters
    ----------
    refit : bool
        When True, the caller must supply *data* and will refit the model.
        When False, the fitted model objects are restored directly.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        snapshot = pickle.load(f)

    if snapshot.get("format_version") != _SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            "This model artifact predates the typed preprocessing-state format and "
            "cannot be loaded. Refit and resave the model with the current version."
        )

    # Reconstruct any custom_transform callables from their saved source code
    _restore_custom_transforms(snapshot)

    if refit and data is None:
        raise ValueError(
            "data must be provided when refit=True. "
            "Pass refit=False to restore from fitted state."
        )

    return {"snapshot": snapshot, "refit": refit, "data": data}
