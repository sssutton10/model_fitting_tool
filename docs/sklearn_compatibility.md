# scikit-learn compatibility

`ModelingTool` is deliberately a stateful modelling workspace: it owns source
data, multiple named model versions, plots, comparison tables, and persistence.
That is useful for insurance modelling workflows but is not the lifecycle
expected by a scikit-learn estimator, so it should not be passed directly to
`Pipeline` or `GridSearchCV`.

`Preprocessor` already follows the core fitted-transformer pattern
(`fit`, `transform`, and `fit_transform`).  A future sklearn adapter should
be a separate, small class rather than changing the Polars-first API:

- `ElasticNetGLMEstimator(**constructor_hyperparameters)` with no data in its
  constructor; `fit(X, y, sample_weight=None)`, `predict(X)`, and `score(X, y)`.
- sklearn cloneability via `BaseEstimator`, `get_params`, and `set_params`.
- fitted trailing-underscore attributes such as `preprocessor_`, `model_`,
  `feature_names_in_`, `n_features_in_`, and `feature_names_out_`.
- explicit conversion at the boundary for pandas/NumPy input, while retaining
  Polars internally; preserve feature names and reject ambiguous positional
  column input.

With that adapter, standard sklearn tooling (`clone`, `Pipeline`,
`GridSearchCV`, `cross_validate`, and model-selection splitters) can be used
without weakening `ModelingTool`'s multi-version workflow.
