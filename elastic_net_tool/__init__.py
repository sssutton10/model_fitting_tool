"""
elastic_net_tool
================

Elastic net GLM modelling tool for insurance loss ratio analysis.
Uses **polars** for all DataFrame operations.

Quick start
-----------
>>> import polars as pl
>>> from elastic_net_tool import ModelingTool, VariableConfig
>>>
>>> tool = ModelingTool(df, target_col='loss_ratio', weight_col='earned_premium')
>>>
>>> # Variable creation
>>> tool.add_variable('vehicle_age', cap_upper=0.99, log_transform=True)
>>> tool.add_variable('driver_age', bin_edges=[16, 25, 35, 50, 65, 100])
>>> tool.add_variable('state', encoding='onehot')
>>> # Categorical remapping
>>> tool.add_variable('state', custom_transform=lambda v: 'South' if v in ('TX','FL') else v)
>>> # Multi-input derived variable
>>> tool.add_variable('age_x_veh', input_cols=['driver_age','vehicle_age'],
...                   custom_transform=lambda a, v: a * v, cap_upper=0.99)
>>> tool.univariate_plot('driver_age')
>>>
>>> # Model fitting
>>> tool.fit_model(['vehicle_age', 'state'], version='v1')
>>>
>>> # CV stability with user-defined fold column
>>> tool.fit_cv_stability(['vehicle_age', 'state'], fold_col='cv_fold', version='v1')
>>>
>>> # Evaluation
>>> tool.ae_chart('driver_age', version='v1')
>>>
>>> # Comparison
>>> tool.compare_models('v1', 'v2')
>>>
>>> # Persistence
>>> tool.save('v1', 'models/v1.pkl')
>>> tool2 = ModelingTool.load('models/v1.pkl', data=df)
"""

from .variable import (
    MISSING_SENTINEL,
    Preprocessor,
    VariableConfig,
    compute_quantile_bin_edges,
    default_config,
    make_bin_labels,
)
from .model import FactorModelVersion, ModelVersion, fit_cv_stability, fit_model
from .metrics import (
    bootstrap_metrics,
    compare_metrics,
    compute_metrics,
    double_lift_score,
    double_lift_table,
    gini_coefficient,
    lift_table,
    vif_table,
)
from .plots import (
    ae_chart,
    bootstrap_ci_plot,
    coefficient_plot,
    cv_stability_plot,
    double_lift_chart,
    importance_plot,
    interaction_heatmap,
    lorenz_chart,
    metrics_bar_chart,
    overfitting_plot,
    pd_plot_2d,
    regularization_path_plot,
    relativities_ci_plot,
    residual_chart,
    residual_heatmap,
    univariate_plot,
)
from .discovery import (
    boruta_select,
    fit_shadow_gbm,
    interaction_ranking,
    monotonicity_test,
    partial_dependence_2d,
    permutation_importance,
    residual_gbm,
    shap_dependence,
    shap_importance,
    shap_interaction_ranking,
    suggest_category_groups,
    tree_interaction_cooccurrence,
)
from .io_utils import load_version, save_version
from .bin_suggestor import (
    suggest_bins,
    suggest_bins_equal_width,
    suggest_bins_gbm,
    suggest_bins_optbin,
    suggest_bins_quantile,
)
from .tool import ModelingTool

__all__ = [
    # Main interface
    "ModelingTool",
    # Variable config
    "VariableConfig",
    "Preprocessor",
    "default_config",
    "compute_quantile_bin_edges",
    "MISSING_SENTINEL",
    # Model
    "ModelVersion",
    "FactorModelVersion",
    "fit_model",
    "fit_cv_stability",
    # Metrics
    "gini_coefficient",
    "lift_table",
    "double_lift_table",
    "double_lift_score",
    "compute_metrics",
    "compare_metrics",
    "vif_table",
    "bootstrap_metrics",
    # Plots
    "univariate_plot",
    "ae_chart",
    "residual_chart",
    "double_lift_chart",
    "lorenz_chart",
    "coefficient_plot",
    "cv_stability_plot",
    "metrics_bar_chart",
    "interaction_heatmap",
    "pd_plot_2d",
    "importance_plot",
    "residual_heatmap",
    "regularization_path_plot",
    "overfitting_plot",
    "bootstrap_ci_plot",
    "relativities_ci_plot",
    # Discovery
    "fit_shadow_gbm",
    "interaction_ranking",
    "partial_dependence_2d",
    "permutation_importance",
    "residual_gbm",
    "shap_importance",
    "shap_dependence",
    "shap_interaction_ranking",
    "tree_interaction_cooccurrence",
    "suggest_category_groups",
    "monotonicity_test",
    "boruta_select",
    # Bin suggestion
    "suggest_bins",
    "suggest_bins_quantile",
    "suggest_bins_equal_width",
    "suggest_bins_optbin",
    "suggest_bins_gbm",
    # IO
    "save_version",
    "load_version",
]
