# Public parameter inventory

This page is a machine-checked inventory of the primary CERM estimator
constructors. `CERMRegressor` now uses the fused residual V2 surface by default;
the historical finite-state Ridge estimator remains available explicitly as
`CERMRidgeRegressor`. `CERMFusedRegressor` is retained as a compatibility name
for the fused family.

CI compares these blocks with the live Python signatures. Adding or removing a
constructor parameter therefore requires an intentional documentation change.

## `CERMClassifier`

<!-- CERM-PARAMETERS:CERMClassifier:START -->
`max_features`, `max_bins`, `subsample`, `colsample`, `n_jobs`,
`search_effort`, `state_detail`, `feature_budget`,
`interaction_search_features`, `interaction_budget`, `selection_fraction`,
`feature_fraction`, `l2_regularization`, `memory_limit_mb`, `preset`,
`max_interaction_features`, `max_interactions`, `interaction_order`,
`reg_lambda`, `max_memory_mb`, `pair_feature_limit`, `search_profile`,
`replacement_objective`, `prediction_backend`, `random_state`,
`categorical_features`, `embedding_features`, `category_policy`,
`max_identity_categories`, `category_bins`, `category_smoothing`,
`category_identity`, `embedding_mode`, `embedding_pca`, `embedding_bins`,
`embedding_prototypes`, `retain_embedding_raw`, `missing_policy`,
`category_newton_l2`, `encoder_kind`, `newton_prebins`, `newton_gain_l2`,
`newton_min_hessian`, `ranking_kind`, `ranking_l2`,
`ranking_prefilter_multiplier`, `cost_per_byte`, `cost_per_operator`,
`block_cost_per_byte`, `block_cost_per_eval`, `selection_strategy`,
`selection_folds`, `selection_near_tie`, `selection_min_improvement`,
`calibration`, `calibration_folds`, `calibration_l2`,
`calibration_min_improvement`, `calibration_min_signal`,
`cache_training_statistics`, `resource_policy`,
`max_estimated_peak_memory_mb`, `max_pair_evaluations`,
`max_block_evaluations`, `max_knn_distance_evaluations`,
`multiclass_strategy`, `shared_multiclass_objective`,
`representation_strategy`, `class_specific_budget`.
<!-- CERM-PARAMETERS:CERMClassifier:END -->

## `CERMRegressor`

`CERMRegressor` is the default fused residual V2 estimator. Its capacity surface
is intentionally distinct from the historical Ridge vocabulary.

<!-- CERM-PARAMETERS:CERMRegressor:START -->
`n_bins`, `max_features`, `max_bins`, `max_interaction_features`, `max_pairs`,
`random_state`, `categorical_features`, `embedding_features`, `category_policy`,
`max_identity_categories`, `category_bins`, `category_smoothing`,
`category_identity`, `missing_policy`.
<!-- CERM-PARAMETERS:CERMRegressor:END -->

## `CERMFusedRegressor`

`CERMFusedRegressor` remains available as an explicit compatibility name for the
same fused regression family. It keeps the same constructor semantics as the new
default `CERMRegressor`.

<!-- CERM-PARAMETERS:CERMFusedRegressor:START -->
`n_bins`, `max_features`, `max_bins`, `max_interaction_features`, `max_pairs`,
`random_state`, `categorical_features`, `embedding_features`, `category_policy`,
`max_identity_categories`, `category_bins`, `category_smoothing`,
`category_identity`, `missing_policy`.
<!-- CERM-PARAMETERS:CERMFusedRegressor:END -->

## `CERMRidgeRegressor`

`CERMRidgeRegressor` preserves the historical finite-state Ridge estimator and
its previous `CERMRegressor` constructor surface for explicit compatibility.

<!-- CERM-PARAMETERS:CERMRidgeRegressor:START -->
`max_features`, `max_bins`, `subsample`, `colsample`, `n_jobs`,
`search_effort`, `state_detail`, `feature_budget`,
`interaction_search_features`, `interaction_budget`, `selection_fraction`,
`feature_fraction`, `l2_regularization`, `preset`,
`max_interaction_features`, `max_interactions`, `interaction_order`,
`reg_lambda`, `random_state`, `categorical_features`, `embedding_features`,
`category_policy`, `max_identity_categories`, `category_bins`,
`category_smoothing`, `category_identity`, `missing_policy`, `include_linear`.
<!-- CERM-PARAMETERS:CERMRidgeRegressor:END -->

## `CERMGeneralizedRegressor`

<!-- CERM-PARAMETERS:CERMGeneralizedRegressor:START -->
`loss`, `quantile`, `quantiles`, `non_crossing`, `huber_epsilon`, `head_alpha`,
`max_iter`, `tol`, `tweedie_power`, `representation_mode`, `max_features`,
`max_bins`, `subsample`, `colsample`, `n_jobs`, `search_effort`, `state_detail`,
`feature_budget`, `interaction_search_features`, `interaction_budget`,
`selection_fraction`, `feature_fraction`, `l2_regularization`, `preset`,
`max_interaction_features`, `max_interactions`, `interaction_order`,
`reg_lambda`, `random_state`, `categorical_features`, `embedding_features`,
`category_policy`, `max_identity_categories`, `category_bins`,
`category_smoothing`, `category_identity`, `missing_policy`, `include_linear`.
<!-- CERM-PARAMETERS:CERMGeneralizedRegressor:END -->

## Convenience aliases

These aliases remain exact compatibility names for direct constructor controls.
They apply to the historical estimator family (`CERMClassifier`,
`CERMRidgeRegressor`, and `CERMGeneralizedRegressor`). The default fused
`CERMRegressor` and compatibility `CERMFusedRegressor` intentionally do not
reuse Ridge-only constructor aliases.

<!-- CERM-ALIASES:START -->
| Convenience alias | Direct parameter |
|---|---|
| `search_effort` | `preset` |
| `state_detail` | `max_bins` |
| `feature_budget` | `max_features` |
| `interaction_search_features` | `max_interaction_features` |
| `interaction_budget` | `max_interactions` |
| `selection_fraction` | `subsample` |
| `feature_fraction` | `colsample` |
| `l2_regularization` | `reg_lambda` |
| `memory_limit_mb` | `max_memory_mb` |
<!-- CERM-ALIASES:END -->

`memory_limit_mb` is currently exposed by `CERMClassifier`. Fused regression's
compact convenience view maps its dedicated capacity semantics without adding
constructor synonyms.
