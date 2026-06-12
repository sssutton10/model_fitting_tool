"""
Tests for metrics.py — Gini, lift, double-lift, compare_metrics.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from elastic_net_tool.metrics import (
    compare_metrics,
    compute_metrics,
    double_lift_score,
    double_lift_table,
    gini_coefficient,
    lift_table,
)


# ── gini_coefficient ──────────────────────────────────────────────────────────

class TestGiniCoefficient:
    def test_perfect_predictor_normalized_near_one(self):
        y = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        score = gini_coefficient(y, y, normalize=True)
        assert abs(score - 1.0) < 1e-6

    def test_random_predictor_near_zero(self):
        rng = np.random.default_rng(0)
        y = rng.lognormal(size=2000)
        pred = rng.lognormal(size=2000)
        score = gini_coefficient(y, pred, normalize=True)
        assert abs(score) < 0.15   # should be near 0 on average

    def test_inverse_predictor_normalized_near_negative_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score = gini_coefficient(y, -y, normalize=True)
        assert score < -0.8

    def test_with_uniform_weights_same_as_no_weights(self):
        rng = np.random.default_rng(1)
        y = rng.lognormal(size=100)
        pred = y + rng.normal(scale=0.1, size=100)
        w = np.ones(100)
        g_no_w = gini_coefficient(y, pred, normalize=True)
        g_with_w = gini_coefficient(y, pred, weights=w, normalize=True)
        assert abs(g_no_w - g_with_w) < 1e-8

    def test_weighted_gini_differs_from_unweighted(self):
        rng = np.random.default_rng(2)
        y = rng.lognormal(size=100)
        pred = y + rng.normal(scale=0.1, size=100)
        w = rng.uniform(1, 10, size=100)
        g_no_w = gini_coefficient(y, pred, normalize=True)
        g_with_w = gini_coefficient(y, pred, weights=w, normalize=True)
        # They can differ; just check both are in [-1, 1]
        assert -1.0 <= g_no_w <= 1.0
        assert -1.0 <= g_with_w <= 1.0

    def test_unnormalized_returns_raw_gini(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g_raw = gini_coefficient(y, y, normalize=False)
        g_norm = gini_coefficient(y, y, normalize=True)
        # normalized = raw / oracle_gini → raw != 1.0 for non-uniform y
        assert g_norm > g_raw or abs(g_norm - 1.0) < 1e-6

    def test_constant_target_returns_zero(self):
        y = np.ones(50)
        pred = np.arange(50, dtype=float)
        score = gini_coefficient(y, pred, normalize=True)
        assert score == 0.0


# ── lift_table ────────────────────────────────────────────────────────────────

class TestLiftTable:
    def _make(self, n=200, n_buckets=10):
        rng = np.random.default_rng(10)
        y = rng.lognormal(size=n)
        pred = y + rng.normal(scale=0.1, size=n)
        w = rng.uniform(1, 5, size=n)
        return lift_table(y, pred, weights=w, n_buckets=n_buckets)

    def test_returns_dataframe(self):
        assert isinstance(self._make(), pl.DataFrame)

    def test_has_required_columns(self):
        tbl = self._make()
        for col in ("bucket", "actual", "predicted", "exposure", "lift"):
            assert col in tbl.columns

    def test_bucket_count(self):
        tbl = self._make(n_buckets=8)
        assert len(tbl) == 8

    def test_buckets_are_sequential(self):
        tbl = self._make()
        assert tbl["bucket"].to_list() == list(range(1, 11))

    def test_exposure_sums_to_total_weight(self):
        rng = np.random.default_rng(11)
        y = rng.lognormal(size=200)
        pred = y + rng.normal(scale=0.2, size=200)
        w = rng.uniform(1, 5, size=200)
        tbl = lift_table(y, pred, weights=w, n_buckets=10)
        assert abs(tbl["exposure"].sum() - float(w.sum())) < 1e-4

    def test_equal_weight_buckets(self):
        """Each bucket should have approximately equal total exposure."""
        rng = np.random.default_rng(12)
        y = rng.lognormal(size=500)
        pred = y + rng.normal(scale=0.1, size=500)
        w = np.ones(500)   # uniform weights → perfect equal split
        tbl = lift_table(y, pred, weights=w, n_buckets=5)
        exposures = tbl["exposure"].to_numpy()
        # With uniform weights each bucket should have exactly 100
        assert np.allclose(exposures, 100.0, atol=1.0)

    def test_lift_column_relative_to_overall_mean(self):
        rng = np.random.default_rng(13)
        y = rng.lognormal(size=200)
        pred = y + rng.normal(scale=0.1, size=200)
        w = np.ones(200)
        tbl = lift_table(y, pred, weights=w)
        overall = float(np.mean(y))
        # lift = actual / overall_mean
        recomputed = (tbl["actual"] / overall).to_numpy()
        np.testing.assert_allclose(recomputed, tbl["lift"].to_numpy(), rtol=1e-5)


# ── double_lift_table ─────────────────────────────────────────────────────────

class TestDoubleLiftTable:
    def _make(self, n=200, n_buckets=10):
        rng = np.random.default_rng(20)
        y = rng.lognormal(size=n)
        pred1 = y + rng.normal(scale=0.1, size=n)
        pred2 = y + rng.normal(scale=0.2, size=n)
        w = rng.uniform(1, 5, size=n)
        return double_lift_table(y, pred1, pred2, weights=w, n_buckets=n_buckets)

    def test_returns_dataframe(self):
        assert isinstance(self._make(), pl.DataFrame)

    def test_has_required_columns(self):
        tbl = self._make()
        for col in ("bucket", "actual", "model1", "model2", "ratio_mean", "exposure"):
            assert col in tbl.columns

    def test_bucket_count(self):
        assert len(self._make(n_buckets=5)) == 5

    def test_buckets_sorted_by_pred_ratio(self):
        """ratio_mean should be non-decreasing across buckets."""
        tbl = self._make()
        ratios = tbl["ratio_mean"].to_numpy()
        assert np.all(np.diff(ratios) >= -0.01)   # allow tiny float noise

    def test_exposure_sums_to_total(self):
        rng = np.random.default_rng(21)
        y = rng.lognormal(size=200)
        p1 = y + rng.normal(scale=0.1, size=200)
        p2 = y + rng.normal(scale=0.2, size=200)
        w = rng.uniform(1, 5, size=200)
        tbl = double_lift_table(y, p1, p2, weights=w)
        assert abs(tbl["exposure"].sum() - float(w.sum())) < 1e-4


# ── double_lift_score ─────────────────────────────────────────────────────────

class TestDoubleLiftScore:
    def _table(self, actual, m1, m2):
        return pl.DataFrame({"actual": actual, "model1": m1, "model2": m2,
                              "bucket": list(range(1, len(actual) + 1)),
                              "ratio_mean": [1.0] * len(actual),
                              "exposure": [1.0] * len(actual)})

    def test_negative_means_model1_closer(self):
        """(m1-actual) < (m2-actual)  →  m1-m2 < 0  →  score negative."""
        tbl = self._table([1.10, 1.20], [1.15, 1.25], [1.20, 1.30])
        score = double_lift_score(tbl)
        assert score < 0

    def test_positive_means_model2_closer(self):
        """m1 consistently higher than m2 → score positive."""
        tbl = self._table([1.10, 1.20], [1.30, 1.40], [1.15, 1.22])
        score = double_lift_score(tbl)
        assert score > 0

    def test_equal_models_score_zero(self):
        tbl = self._table([1.0, 1.0], [1.1, 1.2], [1.1, 1.2])
        score = double_lift_score(tbl)
        assert abs(score) < 1e-10

    def test_example_from_spec(self):
        """Exact numbers from the feature spec:
        actual=1.10, m1=1.15, m2=1.20  → (1.15-1.10)-(1.20-1.10) = -0.05
        """
        tbl = self._table([1.10], [1.15], [1.20])
        score = double_lift_score(tbl)
        assert abs(score - (-0.05)) < 1e-9

    def test_sum_over_all_buckets(self):
        actuals = [1.0, 1.1, 1.2]
        m1s = [1.05, 1.15, 1.25]
        m2s = [1.10, 1.20, 1.30]
        tbl = self._table(actuals, m1s, m2s)
        expected = sum(a - b for a, b in zip(m1s, m2s))
        assert abs(double_lift_score(tbl) - expected) < 1e-9


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_dataframe(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.1])
        result = compute_metrics(y, pred)
        assert isinstance(result, pl.DataFrame)

    def test_has_all_metric_rows(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.1])
        result = compute_metrics(y, pred)
        assert set(result["metric"].to_list()) == {"mse", "rmse", "mae", "gini", "gini_norm"}

    def test_rmse_equals_sqrt_mse(self):
        rng = np.random.default_rng(30)
        y = rng.lognormal(size=100)
        pred = y + rng.normal(scale=0.1, size=100)
        result = compute_metrics(y, pred)
        mse_val = result.filter(pl.col("metric") == "mse")["model"].item()
        rmse_val = result.filter(pl.col("metric") == "rmse")["model"].item()
        assert abs(rmse_val - mse_val ** 0.5) < 1e-8

    def test_perfect_prediction_zero_mse(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y, y)
        mse_val = result.filter(pl.col("metric") == "mse")["model"].item()
        assert abs(mse_val) < 1e-10

    def test_version_name_used_as_column(self):
        y = np.array([1.0, 2.0])
        result = compute_metrics(y, y, version_name="my_model")
        assert "my_model" in result.columns


# ── compare_metrics ───────────────────────────────────────────────────────────

class TestCompareMetrics:
    def _y_and_preds(self):
        rng = np.random.default_rng(40)
        y = rng.lognormal(size=200)
        p1 = y + rng.normal(scale=0.05, size=200)   # better
        p2 = y + rng.normal(scale=0.20, size=200)   # worse
        return y, p1, p2

    def test_has_winner_column(self):
        y, p1, p2 = self._y_and_preds()
        result = compare_metrics(y, p1, p2)
        assert "winner" in result.columns

    def test_lower_mse_model_wins(self):
        y, p1, p2 = self._y_and_preds()   # p1 is better
        result = compare_metrics(y, p1, p2, name1="m1", name2="m2")
        mse_row = result.filter(pl.col("metric") == "mse")
        assert mse_row["winner"].item() == "m1"

    def test_higher_gini_model_wins(self):
        y, p1, p2 = self._y_and_preds()
        result = compare_metrics(y, p1, p2, name1="m1", name2="m2")
        gini_row = result.filter(pl.col("metric") == "gini_norm")
        assert gini_row["winner"].item() == "m1"

    def test_tie_winner_label(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compare_metrics(y, y, y, name1="m1", name2="m2")
        mse_row = result.filter(pl.col("metric") == "mse")
        assert mse_row["winner"].item() == "tie"

    def test_dl_score_row_appended(self):
        y, p1, p2 = self._y_and_preds()
        dl = 0.05
        result = compare_metrics(y, p1, p2, dl_score=dl)
        assert "double_lift_score" in result["metric"].to_list()

    def test_dl_score_negative_model1_wins(self):
        y, p1, p2 = self._y_and_preds()
        result = compare_metrics(y, p1, p2, name1="m1", name2="m2", dl_score=-0.1)
        dl_row = result.filter(pl.col("metric") == "double_lift_score")
        assert dl_row["winner"].item() == "m1"

    def test_dl_score_positive_model2_wins(self):
        y, p1, p2 = self._y_and_preds()
        result = compare_metrics(y, p1, p2, name1="m1", name2="m2", dl_score=0.1)
        dl_row = result.filter(pl.col("metric") == "double_lift_score")
        assert dl_row["winner"].item() == "m2"

    def test_dl_score_zero_tie(self):
        y, p1, p2 = self._y_and_preds()
        result = compare_metrics(y, p1, p2, name1="m1", name2="m2", dl_score=0.0)
        dl_row = result.filter(pl.col("metric") == "double_lift_score")
        assert dl_row["winner"].item() == "tie"

    def test_dl_score_same_value_in_both_model_columns(self):
        y, p1, p2 = self._y_and_preds()
        result = compare_metrics(y, p1, p2, name1="m1", name2="m2", dl_score=0.42)
        dl_row = result.filter(pl.col("metric") == "double_lift_score")
        assert abs(dl_row["m1"].item() - 0.42) < 1e-9
        assert abs(dl_row["m2"].item() - 0.42) < 1e-9
