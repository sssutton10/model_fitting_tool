"""Tests for VIF and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


class TestVIFTable:
    def test_uncorrelated_features(self):
        """Uncorrelated features should have VIF close to 1."""
        from elastic_net_tool.metrics import vif_table

        rng = np.random.default_rng(42)
        n = 200
        dm = pl.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        result = vif_table(dm)
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"variable", "vif"}
        assert len(result) == 3
        # All VIFs should be close to 1.0 for independent features
        for vif_val in result["vif"].to_list():
            assert vif_val < 2.0, f"VIF {vif_val} too high for uncorrelated features"

    def test_correlated_features(self):
        """Highly correlated features should have high VIF."""
        from elastic_net_tool.metrics import vif_table

        rng = np.random.default_rng(42)
        n = 200
        a = rng.normal(0, 1, n)
        b = a + rng.normal(0, 0.01, n)  # Nearly identical to a
        c = rng.normal(0, 1, n)
        dm = pl.DataFrame({"a": a, "b": b, "c": c})
        result = vif_table(dm)
        # a and b should have very high VIF
        max_vif = result["vif"].max()
        assert max_vif > 50, f"Expected high VIF for correlated features, got {max_vif}"

    def test_sorted_descending(self):
        from elastic_net_tool.metrics import vif_table

        rng = np.random.default_rng(42)
        n = 100
        dm = pl.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        })
        result = vif_table(dm)
        vifs = result["vif"].to_list()
        assert vifs == sorted(vifs, reverse=True)


class TestBootstrapMetrics:
    def test_default_metrics(self):
        from elastic_net_tool.metrics import bootstrap_metrics

        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.lognormal(0, 0.5, n)
        y_pred = y_true + rng.normal(0, 0.1, n)
        w = np.ones(n)

        result = bootstrap_metrics(y_true, y_pred, w, n_bootstrap=50)
        assert isinstance(result, pl.DataFrame)
        assert "metric" in result.columns
        assert "point_estimate" in result.columns
        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns
        assert "std_error" in result.columns
        # Default metrics: gini_norm and mse
        metrics = result["metric"].to_list()
        assert "gini_norm" in metrics
        assert "mse" in metrics

    def test_ci_contains_point_estimate(self):
        from elastic_net_tool.metrics import bootstrap_metrics

        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.lognormal(0, 0.5, n)
        y_pred = y_true + rng.normal(0, 0.1, n)

        result = bootstrap_metrics(y_true, y_pred, n_bootstrap=100)
        for row in result.iter_rows(named=True):
            assert row["ci_lower"] <= row["point_estimate"] <= row["ci_upper"], (
                f"Point estimate {row['point_estimate']} not in "
                f"[{row['ci_lower']}, {row['ci_upper']}] for {row['metric']}"
            )

    def test_custom_metric_fn(self):
        from elastic_net_tool.metrics import bootstrap_metrics

        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.lognormal(0, 0.5, n)
        y_pred = y_true + rng.normal(0, 0.1, n)

        def mae(yt, yp, w):
            return -float(np.average(np.abs(yt - yp), weights=w))

        result = bootstrap_metrics(
            y_true, y_pred,
            metric_fns={"mae": mae},
            n_bootstrap=50,
        )
        assert len(result) == 1
        assert result["metric"][0] == "mae"

    def test_std_error_positive(self):
        from elastic_net_tool.metrics import bootstrap_metrics

        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.lognormal(0, 0.5, n)
        y_pred = y_true + rng.normal(0, 0.1, n)

        result = bootstrap_metrics(y_true, y_pred, n_bootstrap=50)
        for se in result["std_error"].to_list():
            assert se >= 0, f"std_error should be non-negative, got {se}"
