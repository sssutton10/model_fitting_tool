"""
Tests for model.py — fit_model, fit_cv_stability, ModelVersion.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.requires_glum

from elastic_net_tool.model import (
    ModelVersion,
    fit_cv_stability,
    fit_model,
    _geometric_mean_signed,
)
from elastic_net_tool.variable import VariableConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fit(sample_df, variables, **kwargs):
    """Shortcut: fit a model with no CV (fixed alpha) for speed."""
    defaults = dict(
        use_cv=False,
        alpha=0.01,
        l1_ratio=0.5,
    )
    defaults.update(kwargs)
    return fit_model(
        X=sample_df,
        y=sample_df["loss_ratio"],
        variables=variables,
        version_name="test",
        configs={},
        weights=sample_df["earned_premium"],
        **defaults,
    )


# ── ModelVersion basics ───────────────────────────────────────────────────────

class TestModelVersion:
    def test_returns_model_version_instance(self, sample_df):
        mv = _fit(sample_df, ["driver_age"])
        assert isinstance(mv, ModelVersion)

    def test_name_is_set(self, sample_df):
        mv = fit_model(
            X=sample_df, y=sample_df["loss_ratio"],
            variables=["driver_age"], version_name="my_v",
            configs={}, use_cv=False, alpha=0.01,
        )
        assert mv.name == "my_v"

    def test_variables_recorded(self, sample_df):
        mv = _fit(sample_df, ["driver_age", "vehicle_value"])
        assert "driver_age" in mv.variables
        assert "vehicle_value" in mv.variables

    def test_train_predictions_shape(self, sample_df):
        mv = _fit(sample_df, ["driver_age"])
        assert mv.train_predictions.shape == (len(sample_df),)

    def test_train_predictions_positive_with_log_link(self, sample_df):
        mv = _fit(sample_df, ["driver_age"], link="log")
        assert np.all(mv.train_predictions > 0)

    def test_coefficients_has_intercept(self, sample_df):
        mv = _fit(sample_df, ["driver_age"])
        assert "intercept" in mv.coefficients["feature"].to_list()

    def test_coefficients_row_count(self, sample_df):
        mv = _fit(sample_df, ["driver_age", "vehicle_value"])
        # intercept + 2 numeric features = 3 rows
        assert len(mv.coefficients) == 3

    def test_alpha_and_l1_ratio_stored(self, sample_df):
        mv = _fit(sample_df, ["driver_age"], alpha=0.05, l1_ratio=0.7)
        assert abs(mv.alpha - 0.05) < 1e-9
        assert abs(mv.l1_ratio - 0.7) < 1e-9

    def test_coefficient_table_sorted_by_abs(self, sample_df):
        mv = _fit(sample_df, ["driver_age", "vehicle_value"])
        tbl = mv.coefficient_table()
        abs_vals = tbl["coefficient"].abs().to_numpy()
        assert np.all(np.diff(abs_vals) <= 0)   # non-increasing

    def test_predict_on_new_data(self, sample_df):
        mv = _fit(sample_df, ["driver_age"])
        preds = mv.predict(sample_df)
        assert preds.shape == (len(sample_df),)
        assert np.all(preds > 0)


# ── Default configs applied automatically ────────────────────────────────────

class TestDefaultConfigApplication:
    def test_categorical_column_encoded_automatically(self, sample_df):
        mv = _fit(sample_df, ["state"])
        # default config should one-hot encode → multiple coefficients
        state_feats = [f for f in mv.feature_names if f.startswith("state_")]
        assert len(state_feats) >= 1

    def test_numeric_column_has_single_feature(self, sample_df):
        mv = _fit(sample_df, ["driver_age"])
        assert "driver_age" in mv.feature_names
        assert len(mv.feature_names) == 1


# ── Pre-registered configs ────────────────────────────────────────────────────

class TestPreRegisteredConfigs:
    def test_binned_variable_creates_dummies(self, sample_df):
        cfg = VariableConfig("driver_age", n_bins=4, cap_upper=None, impute_strategy=None)
        mv = fit_model(
            X=sample_df, y=sample_df["loss_ratio"],
            variables=["driver_age"], version_name="v",
            configs={"driver_age": cfg},
            use_cv=False, alpha=0.01,
        )
        # New label format: driver_age_A_[lo, hi) / driver_age_B_lo+
        bin_feats = [f for f in mv.feature_names
                     if f.startswith("driver_age_") and f != "driver_age_missing"]
        assert len(bin_feats) >= 1

    def test_custom_transform_applied(self, sample_df):
        cfg = VariableConfig(
            "driver_age",
            cap_upper=None, impute_strategy=None,
            custom_transform=lambda a: a / 100.0,
        )
        mv = fit_model(
            X=sample_df, y=sample_df["loss_ratio"],
            variables=["driver_age"], version_name="v",
            configs={"driver_age": cfg},
            use_cv=False, alpha=0.0,
        )
        assert isinstance(mv, ModelVersion)


# ── CV model fitting ──────────────────────────────────────────────────────────

class TestCVFitting:
    def test_cv_selects_alpha(self, sample_df):
        mv = fit_model(
            X=sample_df, y=sample_df["loss_ratio"],
            variables=["driver_age"],
            version_name="cv",
            configs={},
            weights=sample_df["earned_premium"],
            use_cv=True,
            cv=3,
            alphas=np.array([0.001, 0.01, 0.1]),
        )
        assert mv.alpha in [0.001, 0.01, 0.1]

    def test_cv_fit_info_recorded(self, sample_df):
        mv = fit_model(
            X=sample_df, y=sample_df["loss_ratio"],
            variables=["driver_age"],
            version_name="cv",
            configs={},
            use_cv=True,
            cv=3,
            alphas=np.array([0.01, 0.1]),
        )
        assert "cv_folds" in mv.fit_info
        assert mv.fit_info["cv_folds"] == 3

    def test_predefined_split_accepted(self, sample_df):
        from sklearn.model_selection import PredefinedSplit
        fold_arr = (np.arange(len(sample_df)) % 3).tolist()
        test_fold = np.array([{"fold": i, "idx": i} for i in fold_arr], dtype=object)
        fold_map = {f: i for i, f in enumerate(sorted(set(fold_arr)))}
        test_fold_int = np.array([fold_map[f] for f in fold_arr])
        cv = PredefinedSplit(test_fold_int)
        mv = fit_model(
            X=sample_df, y=sample_df["loss_ratio"],
            variables=["driver_age"],
            version_name="ps",
            configs={},
            weights=sample_df["earned_premium"],
            use_cv=True,
            cv=cv,
            alphas=np.array([0.01, 0.1]),
        )
        assert mv.fit_info["cv_folds"] == "PredefinedSplit"

    def test_no_cv_zero_alpha_is_mle(self, sample_df):
        """alpha=0 with no CV should fit an unpenalised GLM."""
        mv = fit_model(
            X=sample_df, y=sample_df["loss_ratio"],
            variables=["driver_age"],
            version_name="mle",
            configs={},
            use_cv=False,
            alpha=0.0,
        )
        assert mv.alpha == 0.0
        assert len(mv.fit_info) == 0

    def test_explicit_alpha_bypasses_cv(self, sample_df):
        """An explicit alpha overrides use_cv=True — no CV should run."""
        mv = fit_model(
            X=sample_df, y=sample_df["loss_ratio"],
            variables=["driver_age"],
            version_name="mle",
            configs={},
            alpha=0.0,
            # use_cv defaults to True, but explicit alpha should override it
        )
        assert mv.alpha == 0.0
        assert mv.fit_info == {}   # empty means CV was skipped


# ── fit_cv_stability ──────────────────────────────────────────────────────────

class TestFitCVStability:
    def _run(self, sample_df):
        return fit_cv_stability(
            X=sample_df,
            y=sample_df["loss_ratio"],
            variables=["driver_age", "vehicle_value"],
            configs={},
            fold_col="cv_fold",
            weights=sample_df["earned_premium"],
            alpha=0.01,
            l1_ratio=0.5,
        )

    def test_returns_dataframe(self, sample_df):
        result = self._run(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_fold_column(self, sample_df):
        result = self._run(sample_df)
        assert "fold" in result.columns

    def test_row_count_is_folds_plus_three_summary(self, sample_df):
        n_folds = sample_df["cv_fold"].n_unique()
        result = self._run(sample_df)
        assert len(result) == n_folds + 3   # +geomean, +std, +cv_pct

    def test_summary_rows_present(self, sample_df):
        result = self._run(sample_df)
        fold_vals = result["fold"].to_list()
        assert "geomean" in fold_vals
        assert "std" in fold_vals
        assert "cv_pct" in fold_vals

    def test_has_intercept_column(self, sample_df):
        result = self._run(sample_df)
        assert "intercept" in result.columns

    def test_fold_rows_have_numeric_fold_labels(self, sample_df):
        result = self._run(sample_df)
        n_folds = sample_df["cv_fold"].n_unique()
        fold_rows = result.head(n_folds)["fold"].to_list()
        assert all(f.isdigit() or f.lstrip("-").isdigit() for f in fold_rows)

    def test_std_row_non_negative(self, sample_df):
        result = self._run(sample_df)
        std_row = result.filter(pl.col("fold") == "std")
        numeric_cols = [c for c in std_row.columns if c != "fold"]
        for col in numeric_cols:
            assert std_row[col].item() >= 0.0

    def test_geomean_has_same_sign_as_majority(self, sample_df):
        """Geometric mean should reflect majority sign of the coefficients."""
        result = self._run(sample_df)
        n_folds = sample_df["cv_fold"].n_unique()
        fold_data = result.head(n_folds)
        geomean_row = result.filter(pl.col("fold") == "geomean")
        for col in ["driver_age", "vehicle_value"]:
            if col in fold_data.columns:
                vals = fold_data[col].to_numpy()
                gm = geomean_row[col].item()
                majority_pos = np.sum(vals > 0) >= np.sum(vals < 0)
                if majority_pos:
                    assert gm >= 0
                else:
                    assert gm <= 0


# ── _geometric_mean_signed ────────────────────────────────────────────────────

class TestGeometricMeanSigned:
    def test_all_positive(self):
        vals = np.array([2.0, 4.0, 8.0])
        gm = _geometric_mean_signed(vals)
        expected = np.exp(np.mean(np.log([2.0, 4.0, 8.0])))
        assert abs(gm - expected) < 1e-8

    def test_all_negative_returns_negative(self):
        vals = np.array([-2.0, -4.0, -8.0])
        gm = _geometric_mean_signed(vals)
        assert gm < 0

    def test_all_zero_returns_zero(self):
        vals = np.zeros(5)
        assert _geometric_mean_signed(vals) == 0.0

    def test_empty_returns_zero(self):
        assert _geometric_mean_signed(np.array([])) == 0.0

    def test_mixed_sign_majority_wins(self):
        vals = np.array([1.0, 2.0, -0.5])   # majority positive
        gm = _geometric_mean_signed(vals)
        assert gm > 0

    def test_single_value_equals_itself(self):
        gm = _geometric_mean_signed(np.array([3.0]))
        assert abs(gm - 3.0) < 1e-8


# ── FactorModelVersion helpers ────────────────────────────────────────────────

from elastic_net_tool.model import _factor_dict, _apply_factors, _resolve_level_arr


class TestFactorModelHelpers:
    pytestmark = []   # these helpers don't require glum

    # _factor_dict ──────────────────────────────────────────────────────────────

    def test_factor_dict_basic(self):
        ft = pl.DataFrame({"Level": ["A", "B", "C"], "Factor": [1.1, 0.9, 1.2]})
        d = _factor_dict(ft)
        assert d == {"A": 1.1, "B": 0.9, "C": 1.2}

    def test_factor_dict_empty_returns_empty_dict(self):
        ft = pl.DataFrame({"Level": pl.Series([], dtype=pl.String),
                           "Factor": pl.Series([], dtype=pl.Float64)})
        assert _factor_dict(ft) == {}

    # _apply_factors ────────────────────────────────────────────────────────────

    def test_apply_factors_all_known_levels(self):
        level_arr = np.array(["A", "B", "A"], dtype=object)
        fdict = {"A": 1.5, "B": 0.8}
        result = _apply_factors(level_arr, fdict, "var1", 1.0)
        np.testing.assert_array_almost_equal(result, [1.5, 0.8, 1.5])

    def test_apply_factors_unknown_level_uses_missing_factor(self):
        level_arr = np.array(["A", "UNKNOWN"], dtype=object)
        fdict = {"A": 1.5}
        result = _apply_factors(level_arr, fdict, "var1", 2.0)
        np.testing.assert_array_almost_equal(result, [1.5, 2.0])

    def test_apply_factors_all_unknown_uses_missing_factor(self):
        level_arr = np.array(["X", "Y"], dtype=object)
        result = _apply_factors(level_arr, {}, "var1", 0.5)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5])

    # _resolve_level_arr ────────────────────────────────────────────────────────

    def test_resolve_level_arr_no_preprocessor_casts_to_string(self):
        X = pl.DataFrame({"age": [25, 30, 45]})
        result = _resolve_level_arr("age", X, None, set(), 3, None)
        assert list(result) == ["25", "30", "45"]

    def test_resolve_level_arr_onehot(self):
        X = pl.DataFrame({"color": ["red", "blue", "red"]})
        Xt = pl.DataFrame({
            "color_blue":  [0, 1, 0],
            "color_green": [0, 0, 0],
        })
        p = {
            "is_categorical": True,
            "encoding": "onehot",
            "categories": ["blue", "green"],
            "dropped_category": "red",
        }
        result = _resolve_level_arr("color", X, Xt, set(Xt.columns), 3, p)
        assert list(result) == ["red (base)", "blue", "red (base)"]

    def test_resolve_level_arr_binned(self):
        X = pl.DataFrame({"age": [25, 35, 45]})
        # Bins: A_<30, B_[30,40), C_40+  —  dropped_bin=1 (B_[30,40))
        Xt = pl.DataFrame({
            "age_A_<30":   [1, 0, 0],
            "age_C_40+":   [0, 0, 1],
            "age_missing": [0, 0, 0],
        })
        p = {
            "is_categorical": False,
            "bin_edges": [30.0, 40.0],
            "bin_labels": ["A_<30", "B_[30,40)", "C_40+"],
            "dropped_bin": 1,
        }
        result = _resolve_level_arr("age", X, Xt, set(Xt.columns), 3, p)
        assert list(result) == ["A_<30", "B_[30,40) (base)", "C_40+"]

    def test_resolve_level_arr_binned_missing_sentinel(self):
        X = pl.DataFrame({"age": [25, None, 45]})
        Xt = pl.DataFrame({
            "age_A_<30":   [1, 0, 0],
            "age_C_40+":   [0, 0, 1],
            "age_missing": [0, 1, 0],
        })
        p = {
            "is_categorical": False,
            "bin_edges": [30.0, 40.0],
            "bin_labels": ["A_<30", "B_[30,40)", "C_40+"],
            "dropped_bin": 1,
        }
        result = _resolve_level_arr("age", X, Xt, set(Xt.columns), 3, p)
        assert list(result) == ["A_<30", "Missing", "C_40+"]

    def test_resolve_level_arr_continuous_fallback(self):
        # p is set but has no bin_edges and is not categorical — falls back to string cast
        X = pl.DataFrame({"score": [1.5, 2.3, 0.7]})
        Xt = pl.DataFrame({"score": [1.5, 2.3, 0.7]})
        p = {"is_categorical": False}
        result = _resolve_level_arr("score", X, Xt, set(Xt.columns), 3, p)
        assert list(result) == ["1.5", "2.3", "0.7"]
