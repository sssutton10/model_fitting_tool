"""Persistence regression tests for chained derived variables (no real glum required)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from elastic_net_tool.io_utils import load_version, save_version
from elastic_net_tool.model import ModelVersion, _build_preprocessor
from elastic_net_tool.tool import ModelingTool
from elastic_net_tool.variable import VariableConfig


def derive_a(df: pl.DataFrame) -> pl.Series:
    return df["x"] * 2


def derive_b(df: pl.DataFrame) -> pl.Series:
    return df["a"] + 1


class DummyGLM:
    """Small pickleable predictor used to exercise frozen loading."""

    def predict(self, X, offset=None):
        result = np.asarray(X)[:, 0].astype(float)
        return result if offset is None else result + np.asarray(offset)


def _artifact_parts():
    data = pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "target": [1.0, 1.0, 1.0],
        "weight": [1.0, 1.0, 1.0],
    })
    configs = {
        "a": VariableConfig(
            "a", input_cols=["x"], custom_transform=derive_a,
            cap_upper=None, impute_strategy=None,
        ),
        "b": VariableConfig(
            "b", input_cols=["a"], custom_transform=derive_b,
            cap_upper=None, impute_strategy=None,
        ),
    }
    prep = _build_preprocessor(["b"], data, configs).fit(data)
    predictions = prep.transform(data)["b"].to_numpy()
    version = ModelVersion(
        name="chain",
        variables=["b"],
        preprocessor=prep,
        glm=DummyGLM(),
        feature_names=["b"],
        coefficients=pl.DataFrame({
            "feature": ["intercept", "b"],
            "coefficient": [0.0, 1.0],
        }),
        alpha=0.0,
        l1_ratio=0.0,
        family="poisson",
        link="identity",
        train_predictions=predictions,
        fit_info={"Fit_Time": "test"},
        gradient_tol=1e-7,
    )
    tool = SimpleNamespace(
        variable_configs=configs,
        target_col="target",
        weight_col="weight",
        offset_col=None,
        link="identity",
        drop_reference="max_weight",
    )
    return data, configs, version, tool


def test_snapshot_includes_dependency_configs_and_transform_sources(tmp_path):
    data, _, version, tool = _artifact_parts()
    path = tmp_path / "chain.pkl"

    save_version(version, tool, path)
    snapshot = load_version(path, refit=False)["snapshot"]

    saved_configs = snapshot["tool_settings"]["variable_configs"]
    saved_sources = snapshot["tool_settings"]["custom_transform_sources"]
    assert list(saved_configs) == ["a", "b"]
    assert set(saved_sources) == {"a", "b"}
    assert all(saved_configs[col].custom_transform is not None for col in ("a", "b"))

    saved_prep = snapshot["version"]["preprocessor"]
    assert saved_prep.output_cols == ["b"]
    assert list(saved_prep.configs) == ["a", "b"]
    np.testing.assert_allclose(
        saved_prep.transform(data)["b"].to_numpy(), [3.0, 5.0, 7.0]
    )

    metadata = json.loads(path.with_suffix(".json").read_text())
    assert list(metadata["variable_transformations"]) == ["a", "b"]


def test_frozen_round_trip_scores_chain_with_and_without_initial_data(tmp_path):
    data, _, version, tool = _artifact_parts()
    path = tmp_path / "chain.pkl"
    save_version(version, tool, path)

    frozen = ModelingTool.load_frozen(path, data)
    np.testing.assert_allclose(
        frozen.model_versions["v1"].train_predictions, [3.0, 5.0, 7.0]
    )
    assert frozen.current_version == "v1"
    assert frozen.drop_reference == "max_weight"
    assert list(frozen.variable_configs) == ["a", "b"]
    assert frozen.model_versions["v1"].gradient_tol == 1e-7

    data_free = ModelingTool.load_frozen(path)
    assert data_free.model_versions["v1"].train_predictions.size == 0
    np.testing.assert_allclose(
        data_free.model_versions["v1"].predict(data), [3.0, 5.0, 7.0]
    )


def test_frozen_round_trip_preserves_pipeline_alias(tmp_path):
    data = pl.DataFrame({
        "x": [1.0, np.e, np.e**2],
        "target": [1.0, 1.0, 1.0],
        "weight": [1.0, 1.0, 1.0],
    })
    config = VariableConfig(
        "x_logged", input_cols=["x"], log_transform=True,
        cap_upper=None, impute_strategy=None,
    )
    prep = _build_preprocessor(["x_logged"], data, {"x_logged": config}).fit(data)
    predictions = prep.transform(data)["x_logged"].to_numpy()
    version = ModelVersion(
        name="alias",
        variables=["x_logged"],
        preprocessor=prep,
        glm=DummyGLM(),
        feature_names=["x_logged"],
        coefficients=pl.DataFrame({
            "feature": ["intercept", "x_logged"],
            "coefficient": [0.0, 1.0],
        }),
        alpha=0.0,
        l1_ratio=0.0,
        family="poisson",
        link="identity",
        train_predictions=predictions,
        fit_info={"Fit_Time": "test"},
    )
    tool = SimpleNamespace(
        variable_configs={"x_logged": config},
        target_col="target",
        weight_col="weight",
        offset_col=None,
        link="identity",
        drop_reference="max_weight",
    )
    path = tmp_path / "alias.pkl"

    save_version(version, tool, path)
    frozen = ModelingTool.load_frozen(path)

    saved_config = frozen.variable_configs["x_logged"]
    assert saved_config.input_cols == ["x"]
    assert saved_config.custom_transform is None
    np.testing.assert_allclose(frozen.predict(data), [0.0, 1.0, 2.0], atol=1e-12)

    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["variable_transformations"]["x_logged"] == {
        "input_cols": ["x"],
        "log_transform": True,
    }


def test_refit_load_restores_dependency_closure(tmp_path, monkeypatch):
    data, _, version, tool = _artifact_parts()
    path = tmp_path / "chain.pkl"
    save_version(version, tool, path)
    observed = {}

    def fake_fit_model(self, variables, version, **kwargs):
        observed["variables"] = variables
        observed["configs"] = dict(self.variable_configs)
        prep = _build_preprocessor(variables, self.data, self.variable_configs).fit(self.data)
        observed["values"] = prep.transform(self.data)["b"].to_list()
        self.model_versions[version] = SimpleNamespace()
        self.current_version = version

    monkeypatch.setattr(ModelingTool, "fit_model", fake_fit_model)

    loaded = ModelingTool.load(path, data=data, version_name="refit_chain")

    assert observed["variables"] == ["b"]
    assert list(observed["configs"]) == ["a", "b"]
    assert observed["values"] == [3.0, 5.0, 7.0]
    assert loaded.current_version == "refit_chain"


def test_save_rejects_transform_without_retrievable_source(tmp_path):
    _, configs, version, tool = _artifact_parts()
    dynamic_transform = eval("lambda df: df['x'] * 2")
    configs["a"] = VariableConfig(
        "a", input_cols=["x"], custom_transform=dynamic_transform,
        cap_upper=None, impute_strategy=None,
    )
    version.preprocessor.configs["a"] = configs["a"]
    tool.variable_configs = configs

    with pytest.raises(ValueError, match="no retrievable source code"):
        save_version(version, tool, tmp_path / "broken.pkl")
