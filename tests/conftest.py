"""
Shared fixtures and configuration for the elastic_net_tool test suite.

Glum availability
-----------------
``glum`` (and several other heavy dependencies) may not be installed in every
environment.  This conftest:

1. Detects whether glum is available **before** any elastic_net_tool imports.
2. If glum is absent, inserts a lightweight ``MagicMock`` into ``sys.modules``
   so that ``elastic_net_tool/__init__.py`` (which imports ``model.py``) can
   still be loaded — allowing the independent submodule tests
   (``test_variable``, ``test_metrics``) to run without the full dependency
   stack.
3. Registers a custom ``requires_glum`` marker + an autouse fixture so that
   any test class/module decorated with that marker is automatically skipped
   when glum is absent.
"""

from __future__ import annotations

import importlib.util
import sys

import matplotlib
import numpy as np
import polars as pl
import pytest

# ── Step 1: detect glum BEFORE any elastic_net_tool import ───────────────────
# importlib.util.find_spec queries the file-system finders, not sys.modules,
# so it returns None even if we later add a mock to sys.modules.
_HAS_GLUM: bool = importlib.util.find_spec("glum") is not None

# ── Step 2: if glum is absent, mock it so __init__.py can be imported ────────
if not _HAS_GLUM:
    from unittest.mock import MagicMock  # noqa: E402
    _glum_mock = MagicMock()
    # Give it realistic attribute names so model.py's top-level try/except
    # doesn't raise ImportError (the raise is guarded by except ImportError).
    _glum_mock.__version__ = "0.0.0-mock"
    sys.modules.setdefault("glum", _glum_mock)

# ── Step 3: now it is safe to import elastic_net_tool ────────────────────────
matplotlib.use("Agg")   # headless – no display needed during CI / testing


# ── Pytest hooks ──────────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_glum: skip this test when glum is not installed "
        "(pip install glum)",
    )
    # Store the flag so fixtures can access it via request.config
    config._has_glum = _HAS_GLUM  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _skip_without_glum(request: pytest.FixtureRequest) -> None:
    """Auto-skip any test marked ``requires_glum`` when glum is absent."""
    if request.node.get_closest_marker("requires_glum"):
        if not request.config._has_glum:  # type: ignore[attr-defined]
            pytest.skip("glum not installed – pip install glum")


# ── Primary dataset fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def sample_df(rng: np.random.Generator) -> pl.DataFrame:
    """
    300-row synthetic insurance dataset.

    Columns
    -------
    driver_age        continuous, no missings
    vehicle_value     continuous, right-skewed (exponential)
    annual_mileage    continuous, ~10 % sentinel-missing (-999_999_999)
    state             5-level categorical; CA gets 40 % → dropped by max_weight
    region            3-level categorical
    loss_ratio        positive target (lognormal)
    earned_premium    exposure weight
    cv_fold           integers 0-4, cycling through rows (60 rows per fold)
    """
    n = 300
    driver_age = rng.integers(18, 80, n).astype(float)
    vehicle_value = rng.exponential(15_000, n)
    raw_mileage = rng.integers(5_000, 30_000, n).astype(float)
    missing_mask = rng.random(n) < 0.10
    annual_mileage = np.where(missing_mask, -999_999_999.0, raw_mileage)

    # CA gets 40 % → highest total weight → dropped as reference by max_weight
    state_choices = rng.choice(
        ["CA", "TX", "FL", "NY", "OH"],
        size=n,
        p=[0.40, 0.25, 0.15, 0.12, 0.08],
    ).tolist()
    region_choices = rng.choice(["East", "West", "South"], n).tolist()

    loss_ratio = np.maximum(rng.lognormal(mean=-0.3, sigma=0.5, size=n), 0.01)
    earned_premium = rng.uniform(500, 5_000, n)
    cv_fold = (np.arange(n) % 5).tolist()

    return pl.DataFrame({
        "driver_age":     driver_age,
        "vehicle_value":  vehicle_value,
        "annual_mileage": annual_mileage,
        "state":          state_choices,
        "region":         region_choices,
        "loss_ratio":     loss_ratio,
        "earned_premium": earned_premium,
        "cv_fold":        cv_fold,
    })


# ── Pre-built tool fixtures (glum required) ───────────────────────────────────

@pytest.fixture(scope="session")
def base_tool(sample_df: pl.DataFrame):
    """Plain tool with no variables pre-configured."""
    if not _HAS_GLUM:
        pytest.skip("glum not installed")
    from elastic_net_tool import ModelingTool
    return ModelingTool(
        sample_df,
        target_col="loss_ratio",
        weight_col="earned_premium",
    )


@pytest.fixture(scope="session")
def fitted_tool(sample_df: pl.DataFrame):
    """
    Tool with two fitted versions (v1 and v2) ready for comparison tests.
    Uses fixed alpha (no CV) to keep fixture creation fast.
    """
    if not _HAS_GLUM:
        pytest.skip("glum not installed")
    from elastic_net_tool import ModelingTool
    tool = ModelingTool(
        sample_df,
        target_col="loss_ratio",
        weight_col="earned_premium",
    )
    tool.add_variable("driver_age", cap_upper=0.99)
    tool.add_variable("state", encoding="onehot")

    tool.fit_model(["driver_age"], version="v1",
                   use_cv=False, alpha=0.01, l1_ratio=0.5, print_summary=False)
    tool.fit_model(["driver_age", "state"], version="v2",
                   use_cv=False, alpha=0.01, l1_ratio=0.5, print_summary=False)
    return tool
