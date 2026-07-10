"""Regression coverage for strict validation and typed fitted preprocessing state."""

import numpy as np
import polars as pl
import pytest

from elastic_net_tool.metrics import double_lift_table, gini_coefficient, lift_table
from elastic_net_tool.variable import (
    FittedBinnedNumericParams,
    FittedCategoricalParams,
    Preprocessor,
    VariableConfig,
)


def test_variable_config_rejects_conflicting_binning_options():
    with pytest.raises(ValueError, match="both n_bins and bin_edges"):
        VariableConfig("x", n_bins=3, bin_edges=[1.0, 2.0])


def test_preprocessor_uses_typed_categorical_params_and_rejects_unseen_levels():
    train = pl.DataFrame({"state": ["CA", "NY", "CA"]})
    prep = Preprocessor([VariableConfig("state", is_categorical=True)])
    prep.fit(train)
    assert isinstance(prep._params["state"], FittedCategoricalParams)
    with pytest.raises(ValueError, match="unseen categorical levels"):
        prep.transform(pl.DataFrame({"state": ["TX"]}))


def test_preprocessor_uses_typed_binned_params():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    prep = Preprocessor([VariableConfig("x", bin_edges=[2.0, 3.0])])
    prep.fit(df)
    assert isinstance(prep._params["x"], FittedBinnedNumericParams)


def test_metrics_reject_bad_weights_and_zero_double_lift_denominator():
    with pytest.raises(ValueError, match="positive total"):
        gini_coefficient(np.array([1.0, 2.0]), np.array([1.0, 2.0]), weights=np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="must not contain zero"):
        double_lift_table(np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0]), n_buckets=2)


def test_lift_handles_tied_predictions_with_a_valid_schema():
    table = lift_table(np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4), n_buckets=2)
    assert set(table.columns) == {"bucket", "actual", "predicted", "exposure", "lift"}
    assert table["bucket"].to_list() == [1, 2]
