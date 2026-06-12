"""Save and load model versions using pickle."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl


def _make_snapshot(version, tool) -> Dict[str, Any]:
    """Package a ModelVersion and tool settings into a serialisable dict."""
    return {
        "version": {
            "name": version.name,
            "variables": version.variables,
            "preprocessor": version.preprocessor,
            "glm": version.glm,
            "feature_names": version.feature_names,
            "coefficients": version.coefficients,
            "alpha": version.alpha,
            "l1_ratio": version.l1_ratio,
            "family": version.family,
            "link": version.link,
            "fit_info": version.fit_info,
        },
        "tool_settings": {
            "target_col": tool.target_col,
            "weight_col": tool.weight_col,
            "tweedie_power": tool.tweedie_power,
            "link": tool.link,
            "drop_reference": getattr(tool, "drop_reference", "max_weight"),
            "variable_configs": {
                col: cfg
                for col, cfg in tool.variable_configs.items()
                if col in version.variables
            },
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
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
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

    if refit and data is None:
        raise ValueError(
            "data must be provided when refit=True. "
            "Pass refit=False to restore from fitted state."
        )

    return {"snapshot": snapshot, "refit": refit, "data": data}
