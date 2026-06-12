"""
Tests for io_utils.py and ModelingTool.save / .load / .load_frozen.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.requires_glum

from elastic_net_tool import ModelingTool


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
