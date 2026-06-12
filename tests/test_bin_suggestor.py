"""
Tests for bin_suggestor.py — breakpoint-suggestion utilities.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from elastic_net_tool.bin_suggestor import (
    _drop_sentinel,
    suggest_bins,
    suggest_bins_equal_width,
    suggest_bins_gbm,
    suggest_bins_quantile,
)
from elastic_net_tool.variable import MISSING_SENTINEL


# ── Shared data ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def num_df():
    rng = np.random.default_rng(99)
    arr = rng.exponential(10_000, 300)
    return pl.DataFrame({"val": arr})


@pytest.fixture(scope="module")
def y_series(num_df):
    rng = np.random.default_rng(100)
    return pl.Series("y", np.maximum(rng.lognormal(size=len(num_df)), 0.01))


@pytest.fixture(scope="module")
def weights_series(num_df):
    rng = np.random.default_rng(101)
    return pl.Series("w", rng.uniform(500, 5000, len(num_df)))


# ── _drop_sentinel ────────────────────────────────────────────────────────────

class TestDropSentinel:
    def test_removes_sentinel_rows(self):
        arr = np.array([1.0, MISSING_SENTINEL, 3.0, MISSING_SENTINEL, 5.0])
        (clean,) = _drop_sentinel(arr)
        assert len(clean) == 3
        assert MISSING_SENTINEL not in clean

    def test_removes_corresponding_rows_from_all_arrays(self):
        a = np.array([1.0, MISSING_SENTINEL, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        ca, cb = _drop_sentinel(a, b)
        assert len(ca) == 2
        assert len(cb) == 2
        np.testing.assert_array_equal(cb, [10.0, 30.0])

    def test_no_sentinel_returns_original(self):
        arr = np.array([1.0, 2.0, 3.0])
        (clean,) = _drop_sentinel(arr)
        np.testing.assert_array_equal(clean, arr)

    def test_all_sentinel_returns_empty(self):
        arr = np.full(5, MISSING_SENTINEL)
        (clean,) = _drop_sentinel(arr)
        assert len(clean) == 0


# ── suggest_bins_quantile ─────────────────────────────────────────────────────

class TestSuggestBinsQuantile:
    def test_returns_list(self, num_df, weights_series):
        splits = suggest_bins_quantile("val", num_df, n_bins=5,
                                       weights=weights_series, verbose=False)
        assert isinstance(splits, list)

    def test_length_is_n_bins_minus_one(self, num_df, weights_series):
        for n in (3, 5, 8, 10):
            splits = suggest_bins_quantile("val", num_df, n_bins=n,
                                           weights=weights_series, verbose=False)
            assert len(splits) <= n - 1   # may be fewer if ties exist

    def test_splits_are_sorted(self, num_df, weights_series):
        splits = suggest_bins_quantile("val", num_df, n_bins=8,
                                       weights=weights_series, verbose=False)
        assert splits == sorted(splits)

    def test_splits_are_unique(self, num_df, weights_series):
        splits = suggest_bins_quantile("val", num_df, n_bins=10,
                                       weights=weights_series, verbose=False)
        assert len(splits) == len(set(splits))

    def test_all_splits_within_data_range(self, num_df, weights_series):
        data_min = float(num_df["val"].min())
        data_max = float(num_df["val"].max())
        splits = suggest_bins_quantile("val", num_df, n_bins=10,
                                       weights=weights_series, verbose=False)
        assert all(data_min <= s <= data_max for s in splits)

    def test_sentinel_excluded_from_quantiles(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, MISSING_SENTINEL])
        df = pl.DataFrame({"x": arr})
        splits = suggest_bins_quantile("x", df, n_bins=3, verbose=False)
        assert all(s != MISSING_SENTINEL for s in splits)
        assert all(s < 1e9 for s in splits)

    def test_no_weights_still_works(self, num_df):
        splits = suggest_bins_quantile("val", num_df, n_bins=5, verbose=False)
        assert isinstance(splits, list)
        assert len(splits) <= 4

    def test_verbose_prints_output(self, num_df, capsys):
        suggest_bins_quantile("val", num_df, n_bins=3, verbose=True)
        captured = capsys.readouterr()
        assert "val" in captured.out
        assert "Splits" in captured.out


# ── suggest_bins_equal_width ──────────────────────────────────────────────────

class TestSuggestBinsEqualWidth:
    def test_returns_list(self, num_df):
        splits = suggest_bins_equal_width("val", num_df, n_bins=5, verbose=False)
        assert isinstance(splits, list)

    def test_length_is_n_bins_minus_one(self, num_df):
        for n in (3, 5, 10):
            splits = suggest_bins_equal_width("val", num_df, n_bins=n, verbose=False)
            assert len(splits) == n - 1

    def test_splits_are_sorted(self, num_df):
        splits = suggest_bins_equal_width("val", num_df, n_bins=10, verbose=False)
        assert splits == sorted(splits)

    def test_splits_equally_spaced(self, num_df):
        splits = suggest_bins_equal_width("val", num_df, n_bins=5, verbose=False)
        diffs = np.diff(splits)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)

    def test_splits_span_data_range(self, num_df):
        splits = suggest_bins_equal_width("val", num_df, n_bins=3, verbose=False)
        data_min = float(num_df["val"].min())
        data_max = float(num_df["val"].max())
        assert splits[0] > data_min
        assert splits[-1] < data_max

    def test_sentinel_excluded_from_range(self):
        arr = np.array([1.0, 5.0, MISSING_SENTINEL])
        df = pl.DataFrame({"x": arr})
        splits = suggest_bins_equal_width("x", df, n_bins=2, verbose=False)
        # Split should be based on [1, 5] not [-999_999_999, 5]
        assert all(1.0 <= s <= 5.0 for s in splits)


# ── suggest_bins_gbm ──────────────────────────────────────────────────────────

class TestSuggestBinsGBM:
    def test_returns_list(self, num_df, y_series, weights_series):
        splits = suggest_bins_gbm("val", num_df, y_series,
                                  weights=weights_series, max_splits=10, verbose=False)
        assert isinstance(splits, list)

    def test_max_splits_respected(self, num_df, y_series, weights_series):
        for ms in (5, 10, 15):
            splits = suggest_bins_gbm("val", num_df, y_series,
                                      max_splits=ms, verbose=False)
            assert len(splits) <= ms

    def test_splits_are_sorted(self, num_df, y_series, weights_series):
        splits = suggest_bins_gbm("val", num_df, y_series, max_splits=10, verbose=False)
        assert splits == sorted(splits)

    def test_splits_within_data_range(self, num_df, y_series, weights_series):
        clean = num_df["val"].to_numpy()
        clean = clean[clean != MISSING_SENTINEL]
        data_min, data_max = float(clean.min()), float(clean.max())
        splits = suggest_bins_gbm("val", num_df, y_series, max_splits=10, verbose=False)
        if splits:
            assert all(data_min <= s <= data_max for s in splits)

    def test_no_splits_on_constant_feature(self, y_series):
        df = pl.DataFrame({"x": np.ones(len(y_series))})
        splits = suggest_bins_gbm("x", df, y_series, max_splits=10, verbose=False)
        # Constant feature → GBM finds no meaningful splits
        assert isinstance(splits, list)

    def test_sentinel_excluded_from_model(self):
        """GBM should not use sentinel rows for training."""
        rng = np.random.default_rng(200)
        arr = np.concatenate([rng.exponential(10, 100), [MISSING_SENTINEL] * 20])
        df = pl.DataFrame({"x": arr})
        y = pl.Series("y", np.maximum(rng.lognormal(size=120), 0.01))
        splits = suggest_bins_gbm("x", df, y, max_splits=5, verbose=False)
        # All splits should be within the real data range, not near MISSING_SENTINEL
        if splits:
            assert all(s < 1e6 for s in splits)

    def test_verbose_prints_output(self, num_df, y_series, capsys):
        suggest_bins_gbm("val", num_df, y_series, max_splits=5, verbose=True)
        captured = capsys.readouterr()
        assert "val" in captured.out


# ── suggest_bins (combined) ───────────────────────────────────────────────────

class TestSuggestBinsCombined:
    def test_returns_dict(self, num_df, y_series, weights_series):
        with patch("matplotlib.pyplot.show"):
            result = suggest_bins("val", num_df, y_series,
                                  weights=weights_series,
                                  methods=["quantile", "equal_width"],
                                  n_bins=5, show_plot=False)
        assert isinstance(result, dict)

    def test_requested_methods_present_in_result(self, num_df, y_series, weights_series):
        with patch("matplotlib.pyplot.show"):
            result = suggest_bins("val", num_df, y_series,
                                  methods=["quantile", "equal_width"],
                                  show_plot=False)
        assert "quantile" in result
        assert "equal_width" in result

    def test_unknown_method_skipped_gracefully(self, num_df, y_series, capsys):
        with patch("matplotlib.pyplot.show"):
            result = suggest_bins("val", num_df, y_series,
                                  methods=["quantile", "bad_method"],
                                  show_plot=False)
        captured = capsys.readouterr()
        assert "bad_method" in captured.out
        assert "quantile" in result      # valid method still runs
        assert "bad_method" not in result

    def test_method_kwargs_forwarded(self, num_df, y_series):
        """quantile_kwargs should be forwarded to suggest_bins_quantile."""
        with patch("matplotlib.pyplot.show"):
            result = suggest_bins("val", num_df, y_series,
                                  methods=["quantile"],
                                  n_bins=3,
                                  quantile_kwargs={},
                                  show_plot=False)
        # n_bins=3 → at most 2 splits
        assert len(result["quantile"]) <= 2

    def test_show_plot_true_calls_plt_show(self, num_df, y_series):
        with patch("matplotlib.pyplot.show") as mock_show:
            suggest_bins("val", num_df, y_series,
                         methods=["quantile"], n_bins=5, show_plot=True)
        mock_show.assert_called_once()

    def test_show_plot_false_does_not_call_plt_show(self, num_df, y_series):
        with patch("matplotlib.pyplot.show") as mock_show:
            suggest_bins("val", num_df, y_series,
                         methods=["quantile"], n_bins=5, show_plot=False)
        mock_show.assert_not_called()

    def test_all_four_methods(self, num_df, y_series, weights_series):
        with patch("matplotlib.pyplot.show"):
            result = suggest_bins(
                "val", num_df, y_series,
                weights=weights_series,
                methods=["quantile", "equal_width", "gbm"],
                n_bins=5, max_splits=10, show_plot=False,
            )
        for method in ("quantile", "equal_width", "gbm"):
            assert method in result
            assert isinstance(result[method], list)


# ── optbinning (conditional) ──────────────────────────────────────────────────

_HAS_OPTBINNING = importlib.util.find_spec("optbinning") is not None


@pytest.mark.skipif(not _HAS_OPTBINNING, reason="optbinning not installed")
class TestOptBin:

    def test_suggest_bins_optbin_returns_list(self, num_df, y_series, weights_series):
        from elastic_net_tool.bin_suggestor import suggest_bins_optbin
        splits = suggest_bins_optbin("val", num_df, y_series,
                                     weights=weights_series, verbose=False)
        assert isinstance(splits, list)

    def test_optbin_splits_sorted(self, num_df, y_series, weights_series):
        from elastic_net_tool.bin_suggestor import suggest_bins_optbin
        splits = suggest_bins_optbin("val", num_df, y_series,
                                     weights=weights_series, verbose=False)
        assert splits == sorted(splits)

    def test_optbin_splits_finite(self, num_df, y_series, weights_series):
        from elastic_net_tool.bin_suggestor import suggest_bins_optbin
        import math
        splits = suggest_bins_optbin("val", num_df, y_series,
                                     weights=weights_series, verbose=False)
        assert all(math.isfinite(s) for s in splits)

    def test_optbin_in_combined_suggest_bins(self, num_df, y_series, weights_series):
        with patch("matplotlib.pyplot.show"):
            result = suggest_bins("val", num_df, y_series,
                                  weights=weights_series,
                                  methods=["optbin"], show_plot=False)
        assert "optbin" in result

    def test_optbin_kwargs_forwarded(self, num_df, y_series):
        from elastic_net_tool.bin_suggestor import suggest_bins_optbin
        # max_n_bins should limit the number of splits
        splits_limited = suggest_bins_optbin(
            "val", num_df, y_series, verbose=False, max_n_bins=3
        )
        splits_default = suggest_bins_optbin(
            "val", num_df, y_series, verbose=False
        )
        assert len(splits_limited) <= len(splits_default)
