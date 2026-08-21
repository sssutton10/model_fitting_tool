"""
Tests for variable.py — VariableConfig, Preprocessor, helpers.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from elastic_net_tool.variable import (
    MISSING_SENTINEL,
    Preprocessor,
    VariableConfig,
    compute_quantile_bin_edges,
    default_config,
)
from elastic_net_tool.model import _build_preprocessor
from elastic_net_tool.plots import _resolve_level


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_num_df(values, col="x"):
    return pl.DataFrame({col: pl.Series(col, values, dtype=pl.Float64)})


def _make_cat_df(values, col="cat"):
    return pl.DataFrame({col: values})


def _prep(cfg: VariableConfig, df: pl.DataFrame, weights=None) -> pl.DataFrame:
    p = Preprocessor([cfg])
    p.fit(df, weights=weights)
    return p.transform(df)


# ── default_config ────────────────────────────────────────────────────────────

class TestDefaultConfig:
    def test_numeric_column_is_not_categorical(self, sample_df):
        cfg = default_config("driver_age", sample_df["driver_age"])
        assert cfg.is_categorical is False

    def test_string_column_is_categorical(self, sample_df):
        cfg = default_config("state", sample_df["state"])
        assert cfg.is_categorical is True

    def test_numeric_has_cap_upper(self, sample_df):
        cfg = default_config("driver_age", sample_df["driver_age"])
        assert cfg.cap_upper == 0.99

    def test_categorical_impute_strategy_most_frequent(self, sample_df):
        cfg = default_config("state", sample_df["state"])
        assert cfg.impute_strategy == "most_frequent"


# ── compute_quantile_bin_edges ────────────────────────────────────────────────

class TestComputeQuantileBinEdges:
    def test_returns_array(self):
        arr = np.arange(100.0)
        edges = compute_quantile_bin_edges(arr, 5)
        assert isinstance(edges, np.ndarray)

    def test_has_n_plus_one_edges(self):
        arr = np.arange(100.0)
        edges = compute_quantile_bin_edges(arr, 5)
        assert len(edges) == 6

    def test_edges_are_monotone(self):
        arr = np.arange(100.0)
        edges = compute_quantile_bin_edges(arr, 10)
        assert np.all(np.diff(edges) >= 0)

    def test_excludes_sentinel(self):
        arr = np.array([MISSING_SENTINEL] * 10 + list(range(1, 91)), dtype=float)
        edges = compute_quantile_bin_edges(arr, 5)
        assert edges[0] >= 1.0  # sentinel should not drag the min edge down

    def test_all_sentinel_returns_fallback(self):
        arr = np.full(10, MISSING_SENTINEL)
        edges = compute_quantile_bin_edges(arr, 5)
        assert len(edges) == 2


# ── Preprocessor – numeric transforms ────────────────────────────────────────

class TestNumericTransforms:
    def test_passthrough_no_transforms(self):
        df = _make_num_df([1.0, 2.0, 3.0])
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, standardize=False)
        out = _prep(cfg, df)
        np.testing.assert_allclose(out["x"].to_numpy(), [1.0, 2.0, 3.0])

    def test_cap_upper_clips_high_values(self):
        # 99th pctile of [0..99] = 98.01; values above should be clipped
        arr = np.arange(100.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=0.99, impute_strategy=None, standardize=False)
        out = _prep(cfg, df)
        assert float(out["x"].max()) <= float(np.percentile(arr, 99)) + 1e-6

    def test_cap_lower_clips_low_values(self):
        arr = np.arange(100.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_lower=0.01, cap_upper=None,
                             impute_strategy=None, standardize=False)
        out = _prep(cfg, df)
        assert float(out["x"].min()) >= float(np.percentile(arr, 1)) - 1e-6

    def test_log_transform_applies_log1p(self):
        df = _make_num_df([0.0, 1.0, 9.0])
        cfg = VariableConfig("x", cap_upper=None, log_transform=True,
                             impute_strategy=None, standardize=False)
        out = _prep(cfg, df)
        np.testing.assert_allclose(
            out["x"].to_numpy(), np.log1p([0.0, 1.0, 9.0]), rtol=1e-6
        )

    def test_standardize_produces_zero_mean(self):
        arr = np.arange(100.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, standardize=True)
        out = _prep(cfg, df)
        assert abs(float(out["x"].mean())) < 1e-6

    def test_standardize_produces_unit_std(self):
        arr = np.arange(100.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, standardize=True)
        out = _prep(cfg, df)
        assert abs(float(out["x"].std()) - 1.0) < 0.05

    def test_impute_median_fills_nulls(self):
        df = pl.DataFrame({"x": pl.Series("x", [1.0, None, 3.0, None, 5.0])})
        cfg = VariableConfig("x", cap_upper=None, impute_strategy="median", standardize=False)
        out = _prep(cfg, df)
        assert out["x"].is_null().sum() == 0

    def test_impute_median_uses_correct_value(self):
        df = pl.DataFrame({"x": pl.Series("x", [1.0, None, 3.0, None, 5.0])})
        cfg = VariableConfig("x", cap_upper=None, impute_strategy="median", standardize=False)
        out = _prep(cfg, df)
        # median of [1,3,5] = 3; nulls should become 3
        null_positions = [1, 3]
        for i in null_positions:
            assert abs(out["x"].to_numpy()[i] - 3.0) < 1e-6

    def test_impute_mean_fills_nulls(self):
        df = pl.DataFrame({"x": pl.Series("x", [2.0, None, 4.0])})
        cfg = VariableConfig("x", cap_upper=None, impute_strategy="mean", standardize=False)
        out = _prep(cfg, df)
        assert abs(out["x"].to_numpy()[1] - 3.0) < 1e-6

    def test_impute_constant_fills_nulls(self):
        df = pl.DataFrame({"x": pl.Series("x", [1.0, None, 3.0])})
        cfg = VariableConfig("x", cap_upper=None, impute_strategy="constant",
                             impute_value=99.0, standardize=False)
        out = _prep(cfg, df)
        assert abs(out["x"].to_numpy()[1] - 99.0) < 1e-6


# ── Preprocessor – binning ────────────────────────────────────────────────────

class TestBinning:
    def test_n_bins_creates_dummy_columns(self):
        arr = np.arange(100.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=5)
        out = _prep(cfg, df)
        # missing col + (n_bins - 1) label-based bin dummies
        assert "x_missing" in out.columns
        # New names are like x_A_[lo, hi) — not x_bin<i>
        assert any(c != "x_missing" and c.startswith("x_") for c in out.columns)

    def test_n_bins_missing_column_all_zeros_when_no_sentinel(self):
        arr = np.arange(100.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=5)
        out = _prep(cfg, df)
        assert out["x_missing"].sum() == 0

    def test_sentinel_value_gets_missing_bin(self):
        arr = np.array([1.0, 2.0, MISSING_SENTINEL, 4.0, 5.0])
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=4)
        out = _prep(cfg, df)
        assert out["x_missing"].to_numpy()[2] == 1.0   # third row is sentinel

    def test_sentinel_row_all_bin_dummies_zero(self):
        arr = np.array([1.0, MISSING_SENTINEL, 3.0])
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=2)
        out = _prep(cfg, df)
        bin_cols = [c for c in out.columns if c.startswith("x_") and c != "x_missing"]
        for col in bin_cols:
            assert out[col].to_numpy()[1] == 0.0   # sentinel row has no bin active

    def test_explicit_bin_edges_respected(self):
        arr = np.array([5.0, 15.0, 25.0, 35.0])
        df = _make_num_df(arr)
        # 2 break points → 3 bins: <10, [10, 20), 20+
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None,
                             bin_edges=[10.0, 20.0])
        out = _prep(cfg, df)
        assert "x_missing" in out.columns

    def test_bin_dummies_are_mutually_exclusive_per_row(self):
        arr = np.arange(50.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=5)
        out = _prep(cfg, df)
        bin_cols = [c for c in out.columns if c.startswith("x_") and c != "x_missing"]
        # Each non-sentinel row activates exactly one bin dummy (dropped bin shows 0)
        row_sums = sum(out[c].to_numpy() for c in bin_cols)
        assert all(s in (0.0, 1.0) for s in row_sums)

    def test_feature_names_binned(self):
        arr = np.arange(100.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=4)
        p = Preprocessor([cfg])
        p.fit(df)
        names = p.get_feature_names()
        assert "x_missing" in names
        assert all(n.startswith("x_") for n in names)


# ── Preprocessor – categorical ────────────────────────────────────────────────

class TestCategorical:
    def test_onehot_columns_created(self):
        df = _make_cat_df(["A", "B", "C", "A"])
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True)
        out = _prep(cfg, df)
        assert any(c.startswith("cat_") for c in out.columns)

    def test_onehot_drop_first_removes_one_level(self):
        df = _make_cat_df(["A", "B", "C", "A"])
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True)
        p = Preprocessor([cfg])
        p.fit(df)
        # 3 unique levels → 2 dummies after dropping max-weight level
        assert len([c for c in p.get_feature_names() if c.startswith("cat_")]) == 2

    def test_onehot_drop_max_weight_drops_heaviest(self):
        """The level with the largest weight sum should be dropped."""
        df = pl.DataFrame({"cat": ["A", "A", "A", "B", "C"]})
        weights = pl.Series("w", [10.0, 10.0, 10.0, 1.0, 1.0])
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True)
        p = Preprocessor([cfg])
        p.fit(df, weights=weights)
        # A has total weight 30, should be dropped
        assert "cat_A" not in p.get_feature_names()
        assert "cat_B" in p.get_feature_names()
        assert "cat_C" in p.get_feature_names()

    def test_onehot_drop_first_alpha_when_no_weights(self):
        """Without weights, the first alphabetical level should be dropped."""
        df = pl.DataFrame({"cat": ["A", "B", "C", "C", "C"]})
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True)
        p = Preprocessor([cfg])
        p.fit(df, weights=None)
        assert "cat_A" not in p.get_feature_names()

    def test_onehot_values_are_zero_one(self):
        df = _make_cat_df(["A", "B", "A", "B"])
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True)
        out = _prep(cfg, df)
        for col in out.columns:
            assert set(out[col].to_numpy().tolist()).issubset({0.0, 1.0})

    def test_onehot_rows_are_mutually_exclusive(self):
        # A (appears twice) is heaviest → dropped; B and C each get a dummy.
        # Base-level rows (A) show all zeros, non-base rows show exactly one 1.
        df = _make_cat_df(["A", "B", "C", "A"])
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True)
        out = _prep(cfg, df)
        row_sums = out.select(pl.sum_horizontal(pl.all())).to_numpy().flatten()
        assert all(s in (0.0, 1.0) for s in row_sums)

    def test_null_imputed_to_mode(self):
        # Mode is A (appears twice); B appears once → B is dropped (lowest weight / first alpha fallback).
        # Actually: A has more weight → A is dropped; B is kept.
        # None → imputed to mode (A) → base level → cat_B = 0 for that row.
        # Use data where mode is NOT dropped to verify imputation drives the right dummy.
        # Here B=mode, A appears once; no weights → first alpha (A) dropped; B kept.
        # None → imputes to B → cat_B = 1.
        df = pl.DataFrame({"cat": ["A", None, "B", "B"]})
        cfg = VariableConfig("cat", encoding="onehot",
                             impute_strategy="most_frequent", is_categorical=True)
        out = _prep(cfg, df)
        # B is the mode; A (first alphabetically, no weights) is dropped.
        # Row 1 (was None) → imputed to B → cat_B = 1
        assert out["cat_B"].to_numpy()[1] == 1.0

    def test_feature_names_onehot(self):
        # X, Y, Z each appear once; no weights → first alphabetically (X) is dropped.
        df = _make_cat_df(["X", "Y", "Z"])
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True)
        p = Preprocessor([cfg])
        p.fit(df)
        names = set(p.get_feature_names())
        assert "cat_X" not in names       # X dropped (first alphabetically)
        assert names == {"cat_Y", "cat_Z"}


# ── Preprocessor – custom transforms ─────────────────────────────────────────

class TestCustomTransforms:
    def test_lambda_numeric_transform(self):
        df = _make_num_df([1.0, 2.0, 3.0])
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None,
                             custom_transform=lambda df: df["x"].to_numpy() * 2)
        out = _prep(cfg, df)
        np.testing.assert_allclose(out["x"].to_numpy(), [2.0, 4.0, 6.0])

    def test_named_function_numeric_transform(self):
        def double(df):
            return df["x"].to_numpy() * 2.0

        df = _make_num_df([1.0, 2.0, 3.0])
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None,
                             custom_transform=double)
        out = _prep(cfg, df)
        np.testing.assert_allclose(out["x"].to_numpy(), [2.0, 4.0, 6.0])

    def test_transform_kwargs_forwarded_to_named_function(self):
        def scale(df, factor=1.0):
            return df["x"].to_numpy() / factor

        df = _make_num_df([100.0, 200.0, 300.0])
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None,
                             custom_transform=scale,
                             transform_kwargs={"factor": 100.0})
        out = _prep(cfg, df)
        np.testing.assert_allclose(out["x"].to_numpy(), [1.0, 2.0, 3.0])

    def test_transform_kwargs_forwarded_to_lambda(self):
        df = _make_num_df([10.0, 20.0, 30.0])
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None,
                             custom_transform=lambda df, offset=0: df["x"].to_numpy() + offset,
                             transform_kwargs={"offset": 5.0})
        out = _prep(cfg, df)
        np.testing.assert_allclose(out["x"].to_numpy(), [15.0, 25.0, 35.0])

    def test_categorical_element_wise_remap(self):
        # After transform: ["N", "other", "other", "other"].
        # No weights → first alphabetically dropped: "N" (ASCII 78) < "other" (ASCII 111).
        df = pl.DataFrame({"cat": ["north", "south", "east", "west"]})
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True,
                             custom_transform=lambda df: [
                                 "N" if v == "north" else "other"
                                 for v in df["cat"].to_list()
                             ])
        out = _prep(cfg, df)
        assert "cat_other" in out.columns
        assert "cat_N" not in out.columns  # "N" is first alphabetically → dropped

    def test_categorical_transform_kwargs(self):
        # After transform: ["West", "South", "South", "East"].
        # No weights → first alphabetically dropped: "East" < "South" < "West".
        def group(df, mapping=None):
            if mapping is None:
                mapping = {}
            return [mapping.get(v, v) for v in df["cat"].to_list()]

        df = pl.DataFrame({"cat": ["CA", "TX", "FL", "NY"]})
        cfg = VariableConfig("cat", encoding="onehot", is_categorical=True,
                             custom_transform=group,
                             transform_kwargs={"mapping": {"CA": "West", "TX": "South",
                                                           "FL": "South", "NY": "East"}})
        out = _prep(cfg, df)
        assert "cat_South" in out.columns
        assert "cat_West" in out.columns
        assert "cat_East" not in out.columns  # first alphabetically → dropped


# ── Preprocessor – multi-input derived variables ──────────────────────────────

class TestMultiInputVariables:
    def test_multi_input_creates_new_column(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        cfg = VariableConfig(
            col="a_times_b",
            input_cols=["a", "b"],
            custom_transform=lambda df: df["a"].to_numpy() * df["b"].to_numpy(),
            cap_upper=None, impute_strategy=None,
        )
        out = _prep(cfg, df)
        np.testing.assert_allclose(out["a_times_b"].to_numpy(), [4.0, 10.0, 18.0])

    def test_multi_input_with_kwargs(self):
        def weighted_sum(df, w_a=1.0, w_b=1.0):
            return df["a"].to_numpy() * w_a + df["b"].to_numpy() * w_b

        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        cfg = VariableConfig(
            col="combo",
            input_cols=["a", "b"],
            custom_transform=weighted_sum,
            transform_kwargs={"w_a": 2.0, "w_b": 0.5},
            cap_upper=None, impute_strategy=None,
        )
        out = _prep(cfg, df)
        # 1*2 + 3*0.5 = 3.5;  2*2 + 4*0.5 = 6.0
        np.testing.assert_allclose(out["combo"].to_numpy(), [3.5, 6.0])

    def test_multi_input_missing_transform_raises(self):
        df = pl.DataFrame({"a": [1.0], "b": [2.0]})
        cfg = VariableConfig(col="derived", input_cols=["a", "b"],
                             custom_transform=None, cap_upper=None)
        with pytest.raises(ValueError, match="custom_transform"):
            _prep(cfg, df)


# ── Preprocessor – chained derived variables ──────────────────────────────

class TestChainedDerivedVariables:
    @staticmethod
    def _configs():
        return {
            "a": VariableConfig(
                "a", input_cols=["x"], custom_transform=lambda df: df["x"] * 2,
                cap_upper=None, impute_strategy=None,
            ),
            "b": VariableConfig(
                "b", input_cols=["a"], custom_transform=lambda df: df["a"] + 1,
                cap_upper=None, impute_strategy=None,
            ),
            "c": VariableConfig(
                "c", input_cols=["b"], custom_transform=lambda df: df["b"] * 10,
                cap_upper=None, impute_strategy=None,
            ),
        }

    def test_arbitrary_depth_chain_emits_only_requested_output(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        configs = self._configs()
        prep = Preprocessor(list(configs.values()), output_cols=["c"])

        out = prep.fit_transform(df)

        assert out.columns == ["c"]
        assert prep.get_feature_names() == ["c"]
        assert set(prep._params) == {"c"}
        np.testing.assert_allclose(out["c"].to_numpy(), [30.0, 50.0, 70.0])
        predicted = prep.transform(pl.DataFrame({"x": [4.0, 5.0]}))
        np.testing.assert_allclose(predicted["c"].to_numpy(), [90.0, 110.0])

    def test_requested_upstream_and_downstream_are_each_emitted_once(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        prep = _build_preprocessor(["a", "b", "c"], df, self._configs())

        out = prep.fit_transform(df)

        assert out.columns == ["a", "b", "c"]
        assert prep.get_feature_names() == ["a", "b", "c"]

    def test_requested_order_does_not_control_dependency_resolution(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        prep = _build_preprocessor(["c", "a"], df, self._configs())

        out = prep.fit_transform(df)

        assert out.columns == ["c", "a"]
        np.testing.assert_allclose(out["c"].to_numpy(), [30.0, 50.0, 70.0])
        np.testing.assert_allclose(out["a"].to_numpy(), [2.0, 4.0, 6.0])

    def test_downstream_receives_raw_upstream_value(self):
        df = pl.DataFrame({"x": [1.0, 10.0]})
        configs = {
            "a": VariableConfig(
                "a", input_cols=["x"], custom_transform=lambda df: df["x"] * 2,
                cap_upper=5.0, impute_strategy=None,
            ),
            "b": VariableConfig(
                "b", input_cols=["a"], custom_transform=lambda df: df["a"] + 1,
                cap_upper=None, impute_strategy=None,
            ),
        }
        prep = _build_preprocessor(["a", "b"], df, configs)

        out = prep.fit_transform(df)

        np.testing.assert_allclose(out["a"].to_numpy(), [2.0, 5.0])
        np.testing.assert_allclose(out["b"].to_numpy(), [3.0, 21.0])

    def test_dependency_only_config_does_not_fit_its_pipeline(self):
        df = pl.DataFrame({"x": [0.0, 1.0]})
        configs = {
            "a": VariableConfig(
                "a", input_cols=["x"], custom_transform=lambda d: d["x"],
                log_transform=True, cap_upper=None, impute_strategy=None,
            ),
            "b": VariableConfig(
                "b", input_cols=["a"], custom_transform=lambda d: d["a"] + 1,
                cap_upper=None, impute_strategy=None,
            ),
        }

        out = _build_preprocessor(["b"], df, configs).fit_transform(df)

        np.testing.assert_allclose(out["b"].to_numpy(), [1.0, 2.0])

    def test_shared_dependency_is_resolved_once_per_pass(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        calls = {"a": 0}

        def derive_a(frame):
            calls["a"] += 1
            return frame["x"] * 2

        configs = {
            "a": VariableConfig("a", input_cols=["x"], custom_transform=derive_a,
                                cap_upper=None, impute_strategy=None),
            "b": VariableConfig("b", input_cols=["a"], custom_transform=lambda d: d["a"] + 1,
                                cap_upper=None, impute_strategy=None),
            "c": VariableConfig("c", input_cols=["a"], custom_transform=lambda d: d["a"] - 1,
                                cap_upper=None, impute_strategy=None),
            "d": VariableConfig("d", input_cols=["b", "c"],
                                custom_transform=lambda d: d["b"] + d["c"],
                                cap_upper=None, impute_strategy=None),
        }

        Preprocessor(list(configs.values()), output_cols=["d"]).fit_transform(df)

        assert calls["a"] == 2  # once during fit and once during transform

    def test_registered_transform_overrides_same_named_raw_column(self):
        df = pl.DataFrame({"x": [1.0, 2.0], "a": [100.0, 100.0]})
        configs = {
            "a": VariableConfig("a", input_cols=["x"], custom_transform=lambda d: d["x"] * 2,
                                cap_upper=None, impute_strategy=None),
            "b": VariableConfig("b", input_cols=["a"], custom_transform=lambda d: d["a"] + 1,
                                cap_upper=None, impute_strategy=None),
        }

        out = _build_preprocessor(["b"], df, configs).fit_transform(df)

        np.testing.assert_allclose(out["b"].to_numpy(), [3.0, 5.0])

    def test_cycle_raises_clear_error(self):
        df = pl.DataFrame({"x": [1.0]})
        configs = {
            "a": VariableConfig("a", input_cols=["b"], custom_transform=lambda d: d["b"]),
            "b": VariableConfig("b", input_cols=["a"], custom_transform=lambda d: d["a"]),
        }

        with pytest.raises(ValueError, match=r"a -> b -> a"):
            _build_preprocessor(["a"], df, configs)

    def test_missing_dependency_raises_clear_error(self):
        df = pl.DataFrame({"x": [1.0]})
        cfg = VariableConfig("a", input_cols=["missing"], custom_transform=lambda d: d["missing"])

        with pytest.raises(KeyError, match="missing"):
            _build_preprocessor(["a"], df, {"a": cfg})

    def test_preprocessor_without_output_cols_attribute_remains_usable(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        prep = Preprocessor([
            VariableConfig("x", cap_upper=None, impute_strategy=None)
        ]).fit(df)
        del prep.output_cols  # simulate a Preprocessor restored from an older pickle

        out = prep.transform(df)

        assert out.columns == ["x"]

    def test_plot_level_resolution_materializes_chain(self):
        df = pl.DataFrame({"x": np.arange(1.0, 13.0)})
        prep = _build_preprocessor(["c"], df, self._configs()).fit(df)

        levels = _resolve_level("c", df, prep, n_bins=3)

        assert len(levels) == len(df)
        assert levels.n_unique() == 3

        explicit_levels = _resolve_level(
            "c", df, prep, n_bins=3, breaks=[50.0]
        )
        assert explicit_levels.n_unique() == 2

        binned_configs = self._configs()
        binned_configs["c"] = VariableConfig(
            "c", input_cols=["b"], custom_transform=lambda d: d["b"] * 10,
            bin_edges=[50.0], impute_strategy=None,
        )
        binned_prep = _build_preprocessor(["c"], df, binned_configs).fit(df)
        overridden_levels = _resolve_level(
            "c", df, binned_prep, n_bins=3, breaks=[40.0]
        )
        assert overridden_levels.n_unique() == 2

    def test_plot_level_resolution_with_explicit_breaks_handles_missing(self):
        df = pl.DataFrame({"x": [1.0, MISSING_SENTINEL, 3.0]})

        levels = _resolve_level(
            "x", df, preprocessor=None, n_bins=3, breaks=[2.0]
        )

        assert levels.to_list() == ["A_<2", "Missing", "B_2+"]


# ── Preprocessor – error handling ────────────────────────────────────────────

class TestPreprocessorErrors:
    def test_transform_before_fit_raises(self):
        df = _make_num_df([1.0, 2.0])
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None)
        p = Preprocessor([cfg])
        with pytest.raises(RuntimeError, match="fit"):
            p.transform(df)

    def test_get_bin_labels_on_non_binned_raises(self):
        df = _make_num_df([1.0, 2.0, 3.0])
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None)
        p = Preprocessor([cfg])
        p.fit(df)
        with pytest.raises(ValueError, match="bin edges"):
            p.get_bin_labels("x", df["x"])


# ── get_bin_labels ────────────────────────────────────────────────────────────

class TestGetBinLabels:
    def test_returns_series(self):
        arr = np.arange(20.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=4)
        p = Preprocessor([cfg])
        p.fit(df)
        labels = p.get_bin_labels("x", df["x"])
        assert isinstance(labels, pl.Series)
        assert len(labels) == len(arr)

    def test_sentinel_gets_missing_label(self):
        arr = np.array([1.0, MISSING_SENTINEL, 3.0])
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=2)
        p = Preprocessor([cfg])
        p.fit(df)
        labels = p.get_bin_labels("x", df["x"])
        assert labels.to_list()[1] == "Missing"

    def test_labels_have_letter_prefix(self):
        arr = np.arange(20.0)
        df = _make_num_df(arr)
        cfg = VariableConfig("x", cap_upper=None, impute_strategy=None, n_bins=4)
        p = Preprocessor([cfg])
        p.fit(df)
        labels = p.get_bin_labels("x", df["x"])
        non_missing = [l for l in labels.to_list() if l != "Missing"]
        # New format: "A_[lo, hi)" or "D_lo+" — letter then underscore
        assert all(len(l) >= 3 and l[1] == "_" for l in non_missing)
