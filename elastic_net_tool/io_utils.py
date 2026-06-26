"""Save and load model versions using pickle."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle
import json
import re
import textwrap
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import polars as pl


def _extract_import_statements(fn: Callable) -> List[str]:
    """
    Inspect the global names referenced by *fn* and return a list of import
    statements needed to re-create those bindings in a fresh namespace.

    Handles both ``import module as alias`` and ``from module import name``
    patterns.  Names that cannot be resolved to an importable object are
    silently skipped.
    """
    stmts: List[str] = []
    fn_globals = getattr(fn, "__globals__", {})
    referenced = set(getattr(fn.__code__, "co_names", []))
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


def _serialize_custom_transform(fn: Callable) -> Optional[Dict[str, Any]]:
    """
    Serialize a callable to a dict containing its source code so it can be
    reconstructed in a fresh Python session without the original definition.

    Returns None if the source cannot be retrieved.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None

    name = getattr(fn, "__name__", None)
    imports = _extract_import_statements(fn)

    if name == "<lambda>":
        # Extract just the lambda expression from the (possibly indented) source line
        m = re.search(r"(lambda\b[^#\n]*)", source)
        if not m:
            return None
        lambda_expr = m.group(1).rstrip().rstrip(",").rstrip()
        return {"type": "lambda", "source": lambda_expr, "imports": imports}
    else:
        return {"type": "def", "source": textwrap.dedent(source), "name": name, "imports": imports}


def _deserialize_custom_transform(data: Optional[Dict[str, Any]]) -> Optional[Callable]:
    """Reconstruct a callable from its serialized source dict."""
    if data is None:
        return None
    ns: Dict[str, Any] = {}
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
              if cfg.custom_transform is not None else cfg
        for col, cfg in preprocessor.configs.items()
    }
    return cleaned


def _restore_custom_transforms(snapshot: Dict[str, Any]) -> None:
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
    variable_configs: Dict[str, Any],
    transform_sources: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a JSON-serialisable summary of every variable's transformation pipeline."""
    out: Dict[str, Any] = {}
    for col, cfg in variable_configs.items():
        entry: Dict[str, Any] = {}
        if cfg.input_cols:
            entry["input_cols"] = cfg.input_cols
        if cfg.cap_lower is not None:
            entry["cap_lower"] = cfg.cap_lower
        if cfg.cap_upper is not None:
            entry["cap_upper"] = cfg.cap_upper
        if cfg.log_transform:
            entry["log_transform"] = True
        if cfg.impute_strategy:
            entry["impute_strategy"] = cfg.impute_strategy
            if cfg.impute_value is not None:
                entry["impute_value"] = cfg.impute_value
        if cfg.n_bins:
            entry["n_bins"] = cfg.n_bins
        if cfg.bin_edges:
            entry["bin_edges"] = list(cfg.bin_edges)
        if cfg.standardize:
            entry["standardize"] = True
        if cfg.degree != 1:
            entry["degree"] = cfg.degree
        if col in transform_sources:
            entry["custom_transform"] = transform_sources[col]["source"]
        elif cfg.custom_transform is not None:
            entry["custom_transform"] = getattr(cfg.custom_transform, "__name__", "<function>")
        if cfg.transform_kwargs:
            entry["transform_kwargs"] = cfg.transform_kwargs
        out[col] = entry
    return out


def _make_snapshot(version, tool) -> Dict[str, Any]:
    """Package a ModelVersion and tool settings into a serialisable dict."""
    # Separate callable sources from configs so lambdas / locally-defined
    # functions are captured even when standard pickle cannot serialise them.
    safe_configs: Dict[str, Any] = {}
    transform_sources: Dict[str, Any] = {}
    for col, cfg in tool.variable_configs.items():
        if col not in version.variables:
            continue
        if cfg.custom_transform is not None:
            src = _serialize_custom_transform(cfg.custom_transform)
            if src is not None:
                transform_sources[col] = src
            safe_configs[col] = dataclasses.replace(cfg, custom_transform=None)
        else:
            safe_configs[col] = cfg

    return {
        "version": {
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
            'tweedie_power': version.tweedie_power,
        },
        "tool_settings": {
            "target_col": tool.target_col,
            "weight_col": tool.weight_col,
            'offset_col': getattr(tool, "offset_col", None),
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

    # Human-readable version of important metadata for quick reference when browsing saved files
    json_metadata = {
        "name": version.name,
        "variables": version.variables,
        "family": version.family.__class__.__name__,
        "link": version.link.__class__.__name__,
        "fit_info": version.fit_info,
        'target_col': tool.target_col,
        'weight_col': tool.weight_col,
        'offset_col': getattr(tool, "offset_col", None),
        'l1_ratio': version.l1_ratio,
        'alpha': version.alpha,
        'variable_transformations': _summarize_variable_transformations(
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
    data: Optional[pl.DataFrame] = None,
    refit: bool = True,
) -> Dict[str, Any]:
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

    # Reconstruct any custom_transform callables from their saved source code
    _restore_custom_transforms(snapshot)

    if refit and data is None:
        raise ValueError(
            "data must be provided when refit=True. "
            "Pass refit=False to restore from fitted state."
        )

    return {"snapshot": snapshot, "refit": refit, "data": data}
