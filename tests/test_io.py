"""
Tests for io_utils.py and ModelingTool.save / .load / .load_frozen.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.requires_glum

from elastic_net_tool import FactorModelVersion, ModelingTool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _small_tool(sample_df: pl.DataFrame) -> ModelingTool:
    """Fit a minimal tool so tests are fast."""
    tool = ModelingTool(sample_df, target_col="loss_ratio",
                        weight_col="earned_premium")
    tool.add_variable("driver_age", cap_upper=0.99)
    tool.add_variable("state", encoding="onehot")
    tool.fit_model(["driver_age", "state"], version="v1",
                   use_cv=False, alpha=0.01, l1_ratio=0.5, print_summary=False)
    return tool


def _derive_age_twice(df: pl.DataFrame) -> pl.Series:
    return df["driver_age"] * 2


def _derive_age_chain(df: pl.DataFrame) -> pl.Series:
    return df["age_twice"] + 1


def _chained_tool(sample_df: pl.DataFrame) -> ModelingTool:
    tool = ModelingTool(sample_df, target_col="loss_ratio",
                        weight_col="earned_premium")
    tool.add_variable(
        "age_twice", input_cols=["driver_age"],
        custom_transform=_derive_age_twice,
        cap_upper=None, impute_strategy=None,
    )
    tool.add_variable(
        "age_chain", input_cols=["age_twice"],
        custom_transform=_derive_age_chain,
        cap_upper=None, impute_strategy=None,
    )
    tool.fit_model(["age_chain"], version="chain", use_cv=False,
                   alpha=0.01, print_summary=False)
    return tool


def _factor_tool(
    sample_df: pl.DataFrame,
    *,
    tool_offset_col: str | None = None,
    factor_offset_col: str | None = None,
) -> ModelingTool:
    tool = ModelingTool(
        sample_df,
        target_col="loss_ratio",
        weight_col="earned_premium",
        offset_col=tool_offset_col,
    )
    factors = pl.DataFrame(
        {
            "Variable": ["state", "state", "state"],
            "Level": ["CA", "NY", "TX"],
            "Factor": [1.0, 1.1, 0.9],
        }
    )
    model = FactorModelVersion(
        name="factor",
        variables=["state"],
        factor_table=factors,
        preprocessor=None,
        preprocessor_vars=[],
        train_predictions=np.ones(len(sample_df)),
        offset_col=factor_offset_col,
    )
    tool.model_versions["factor"] = model
    tool.current_version = "factor"
    return tool


# ── save ──────────────────────────────────────────────────────────────────────

class TestSave:
    def test_file_created(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        assert (tmp_path / "model.pkl").exists()

    def test_save_prints_confirmation(self, sample_df, tmp_path, capsys):
        tool = _small_tool(sample_df)
        tool.save("v1", str(tmp_path / "m.pkl"))
        captured = capsys.readouterr()
        assert "v1" in captured.out

    def test_save_missing_version_raises(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        with pytest.raises(KeyError):
            tool.save("v_missing", str(tmp_path / "m.pkl"))

    def test_save_creates_parent_directory(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        nested = tmp_path / "subdir" / "nested" / "model.pkl"
        tool.save("v1", str(nested))
        assert nested.exists()

    def test_factor_version_saves_its_own_offset_col(self, sample_df, tmp_path):
        data = sample_df.with_columns(
            pl.lit(0.25).alias("tool_offset"),
            pl.lit(2.0).alias("factor_offset"),
        )
        source = _factor_tool(
            data,
            tool_offset_col="tool_offset",
            factor_offset_col="factor_offset",
        )
        expected = source.predict(data, "factor")
        path = tmp_path / "factor.pkl"

        source.save("factor", path)

        from elastic_net_tool.io_utils import load_version

        snapshot = load_version(path, refit=False)["snapshot"]
        assert snapshot["tool_settings"]["offset_col"] == "factor_offset"
        assert json.loads(path.with_suffix(".json").read_text())["offset_col"] == (
            "factor_offset"
        )

        frozen = ModelingTool.load_version_frozen(path, data=data)
        assert frozen.offset_col == "factor_offset"
        assert frozen.model_versions["factor"].offset_col == "factor_offset"
        np.testing.assert_allclose(
            frozen.model_versions["factor"].train_predictions,
            expected,
        )

        destination = ModelingTool(
            data,
            target_col="loss_ratio",
            weight_col="earned_premium",
            offset_col="tool_offset",
        )
        ModelingTool.load_version_frozen(path, into=destination)
        np.testing.assert_allclose(destination.predict(data, "factor"), expected)


# ── load ──────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_returns_modeling_tool(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df)
        assert isinstance(loaded, ModelingTool)

    def test_loaded_version_is_v1(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df)
        assert "v1" in loaded.model_versions

    def test_loaded_predictions_shape(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df)
        preds = loaded.model_versions["v1"].train_predictions
        assert preds.shape == (len(sample_df),)

    def test_loaded_predictions_positive(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df)
        assert np.all(loaded.model_versions["v1"].train_predictions > 0)

    def test_loaded_target_col_preserved(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df)
        assert loaded.target_col == "loss_ratio"

    def test_loaded_weight_col_preserved(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df)
        assert loaded.weight_col == "earned_premium"

    def test_loaded_variable_configs_restored(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df)
        assert "driver_age" in loaded.variable_configs

    def test_load_overrides_target_col(self, sample_df, tmp_path):
        """Caller can override saved column names."""
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df, target_col="loss_ratio")
        assert loaded.target_col == "loss_ratio"

    def test_load_missing_file_raises(self, sample_df, tmp_path):
        from elastic_net_tool.io_utils import load_version
        with pytest.raises(FileNotFoundError):
            load_version(str(tmp_path / "nonexistent.pkl"), data=sample_df)

    def test_load_without_data_raises(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        with pytest.raises((ValueError, TypeError)):
            ModelingTool.load(path, data=None)

    def test_predictions_close_to_original(self, sample_df, tmp_path):
        """
        After save → load → refit, predictions should be close to original.
        Not identical (floating point, refitting) but within a reasonable range.
        """
        tool = _small_tool(sample_df)
        orig_preds = tool.model_versions["v1"].train_predictions.copy()
        path = str(tmp_path / "model.pkl")
        tool.save("v1", path)
        loaded = ModelingTool.load(path, data=sample_df)
        new_preds = loaded.model_versions["v1"].train_predictions
        # Correlation should be very high
        corr = float(np.corrcoef(orig_preds, new_preds)[0, 1])
        assert corr > 0.99


class TestLoadVersion:
    def test_creates_new_tool_with_saved_name(self, sample_df, tmp_path):
        source = _small_tool(sample_df)
        path = tmp_path / "model.pkl"
        source.save("v1", path)

        loaded = ModelingTool.load_version(path, data=sample_df)

        assert isinstance(loaded, ModelingTool)
        assert loaded.current_version == "v1"

    def test_appends_refitted_version_and_returns_same_tool(
        self, sample_df, tmp_path
    ):
        source = _small_tool(sample_df)
        path = tmp_path / "model.pkl"
        source.save("v1", path)
        loaded = ModelingTool.load_version(path, data=sample_df)

        result = ModelingTool.load_version(path, into=loaded, version_name="v2")

        assert result is loaded
        assert set(loaded.model_versions) == {"v1", "v2"}
        assert loaded.current_version == "v2"
        assert loaded.model_versions["v2"].name == "v2"
        np.testing.assert_allclose(
            loaded.model_versions["v1"].train_predictions,
            loaded.model_versions["v2"].train_predictions,
        )

    def test_append_rejects_data_argument(self, sample_df, tmp_path):
        source = _small_tool(sample_df)
        path = tmp_path / "model.pkl"
        source.save("v1", path)

        with pytest.raises(ValueError, match="cannot be supplied"):
            ModelingTool.load_version(
                path,
                data=sample_df,
                into=source,
                version_name="v2",
            )

    def test_duplicate_name_leaves_tool_unchanged(self, sample_df, tmp_path):
        source = _small_tool(sample_df)
        path = tmp_path / "model.pkl"
        source.save("v1", path)
        original_configs = dict(source.variable_configs)

        with pytest.raises(ValueError, match="already exists"):
            ModelingTool.load_version(path, into=source)

        assert list(source.model_versions) == ["v1"]
        assert source.variable_configs == original_configs
        assert source.current_version == "v1"

    def test_rejects_incompatible_tool_settings(self, sample_df, tmp_path):
        source = _small_tool(sample_df)
        path = tmp_path / "model.pkl"
        source.save("v1", path)
        destination = ModelingTool(sample_df, target_col="loss_ratio")

        with pytest.raises(ValueError, match="weight_col"):
            ModelingTool.load_version(path, into=destination, version_name="v2")

        assert destination.model_versions == {}
        assert destination.variable_configs == {}

    def test_rejects_incompatible_shared_config(self, sample_df, tmp_path):
        source = _small_tool(sample_df)
        path = tmp_path / "model.pkl"
        source.save("v1", path)
        destination = ModelingTool.load_version(path, data=sample_df)
        destination.variable_configs["driver_age"] = replace(
            destination.variable_configs["driver_age"], cap_upper=0.95
        )

        with pytest.raises(ValueError, match="driver_age"):
            ModelingTool.load_version(path, into=destination, version_name="v2")

        assert list(destination.model_versions) == ["v1"]

    def test_identical_loaded_custom_transforms_are_compatible(
        self, sample_df, tmp_path
    ):
        source = _chained_tool(sample_df)
        path = tmp_path / "chain.pkl"
        source.save("chain", path)
        destination = ModelingTool.load_version(path, data=sample_df)

        ModelingTool.load_version(path, into=destination, version_name="chain2")

        assert set(destination.model_versions) == {"chain", "chain2"}


# ── load_frozen ───────────────────────────────────────────────────────────────

class TestLoadFrozen:
    def test_load_frozen_returns_tool(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "frozen.pkl")
        tool.save("v1", path)
        frozen = ModelingTool.load_frozen(path)
        assert isinstance(frozen, ModelingTool)

    def test_load_frozen_has_v1(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "frozen.pkl")
        tool.save("v1", path)
        frozen = ModelingTool.load_frozen(path)
        assert "v1" in frozen.model_versions

    def test_load_frozen_predict_on_data(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        path = str(tmp_path / "frozen.pkl")
        tool.save("v1", path)
        frozen = ModelingTool.load_frozen(path)
        preds = frozen.model_versions["v1"].predict(sample_df)
        assert preds.shape == (len(sample_df),)
        assert np.all(preds > 0)

    def test_frozen_predictions_match_original(self, sample_df, tmp_path):
        tool = _small_tool(sample_df)
        orig_preds = tool.model_versions["v1"].predict(sample_df)
        path = str(tmp_path / "frozen.pkl")
        tool.save("v1", path)
        frozen = ModelingTool.load_frozen(path)
        frozen_preds = frozen.model_versions["v1"].predict(sample_df)
        np.testing.assert_allclose(frozen_preds, orig_preds, rtol=1e-6)


class TestLoadVersionFrozen:
    def test_uses_saved_name_when_creating(self, sample_df, tmp_path):
        source = _small_tool(sample_df)
        path = tmp_path / "frozen.pkl"
        source.save("v1", path)

        frozen = ModelingTool.load_version_frozen(path, data=sample_df)

        assert list(frozen.model_versions) == ["v1"]
        assert frozen.current_version == "v1"

    def test_appends_frozen_version_and_returns_same_tool(
        self, sample_df, tmp_path
    ):
        source = _small_tool(sample_df)
        path = tmp_path / "frozen.pkl"
        source.save("v1", path)
        frozen = ModelingTool.load_version_frozen(path, data=sample_df)

        result = ModelingTool.load_version_frozen(
            path, into=frozen, version_name="frozen2"
        )

        assert result is frozen
        assert set(frozen.model_versions) == {"v1", "frozen2"}
        assert frozen.current_version == "frozen2"
        np.testing.assert_allclose(
            frozen.model_versions["v1"].train_predictions,
            frozen.model_versions["frozen2"].train_predictions,
        )

    def test_appends_factor_version(self, sample_df, tmp_path):
        glm_source = _small_tool(sample_df)
        glm_path = tmp_path / "glm.pkl"
        glm_source.save("v1", glm_path)
        factor_source = _factor_tool(sample_df)
        factor_path = tmp_path / "factor.pkl"
        factor_source.save("factor", factor_path)
        destination = ModelingTool.load_version_frozen(glm_path, data=sample_df)

        ModelingTool.load_version_frozen(factor_path, into=destination)

        assert isinstance(destination.model_versions["factor"], FactorModelVersion)
        assert destination.current_version == "factor"
        assert destination.model_versions["factor"].train_predictions.shape == (
            len(sample_df),
        )

    def test_prediction_failure_is_atomic(self, sample_df, tmp_path):
        source = _small_tool(sample_df)
        path = tmp_path / "frozen.pkl"
        source.save("v1", path)
        incomplete_data = sample_df.drop("driver_age")
        destination = ModelingTool(
            incomplete_data,
            target_col="loss_ratio",
            weight_col="earned_premium",
        )

        with pytest.raises((ValueError, KeyError)):
            ModelingTool.load_version_frozen(
                path, into=destination, version_name="broken"
            )

        assert destination.model_versions == {}
        assert destination.variable_configs == {}
        assert destination.current_version is None

    def test_factor_version_cannot_be_refitted(self, sample_df, tmp_path):
        source = _factor_tool(sample_df)
        path = tmp_path / "factor.pkl"
        source.save("factor", path)

        with pytest.raises(ValueError, match="factor tables are not refittable"):
            ModelingTool.load_version(path, data=sample_df)


# ── chained derived variable persistence ──────────────────────────────────────

class TestChainedPersistence:
    def test_refit_round_trip_restores_dependency_configs(self, sample_df, tmp_path):
        tool = _chained_tool(sample_df)
        path = tmp_path / "chain.pkl"
        tool.save("chain", path)

        loaded = ModelingTool.load(
            path, data=sample_df, version_name="loaded_chain"
        )

        assert list(loaded.variable_configs) == ["age_twice", "age_chain"]
        assert loaded.current_version == "loaded_chain"
        assert loaded.model_versions["loaded_chain"].feature_names == ["age_chain"]
        assert loaded.predict(sample_df, "loaded_chain").shape == (len(sample_df),)

    def test_frozen_round_trip_predictions_match(self, sample_df, tmp_path):
        tool = _chained_tool(sample_df)
        expected = tool.predict(sample_df, "chain")
        path = tmp_path / "chain.pkl"
        tool.save("chain", path)

        frozen = ModelingTool.load_frozen(path)
        actual = frozen.predict(sample_df)

        assert list(frozen.variable_configs) == ["age_twice", "age_chain"]
        assert frozen.model_versions["v1"].feature_names == ["age_chain"]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
