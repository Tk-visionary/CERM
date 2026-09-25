# API stability

CERM is an alpha research package. Public estimator constructor parameters follow
scikit-learn's cloneable estimator conventions, but minor releases may still
change experimental APIs.

The following are intended public surfaces:

- names exported from `cerm.__all__`;
- estimator constructor parameters documented in the API or parameter guides;
- standard estimator methods such as `fit`, `predict`, `predict_proba`,
  `get_params`, and `set_params` where applicable;
- fitted attributes documented in the README or API reference;
- `model_summary` and `CERMModelSummary` as compact inspection helpers;
- semantic, optimized, and compiled program objects returned by public methods.

The top-level parameter-layer helpers (`get_tunable_params`, `get_search_params`,
`get_resource_params`, `get_convenience_params`) and the compatibility estimator
helpers (`get_user_params`, `explain_params`, `parameter_summary`) provide compact
views for people. They do not replace scikit-learn's `get_params` / `set_params`,
which remain the complete cloneable constructor interface.

Everything under `cerm._internal` and `cerm._compat` is private. Experimental
modules whose names explicitly contain `experimental` may change without the
same deprecation window as the ordinary estimator surface.

Regression has one intentional alpha-stage default transition: top-level
`CERMRegressor` now denotes fused residual V2. The former finite-state Ridge
implementation remains public as `CERMRidgeRegressor`, and
`CERMFusedRegressor` remains available as a compatibility name for code that
explicitly adopted fused V2 before the default switch. This change is documented
as a model-default/API transition rather than hidden behind a strategy flag with
ambiguous parameters.

Portable/compiled fused program objects remain newer surfaces and may evolve
more quickly than the core estimator names. Pickled objects, training caches,
and native packages are not guaranteed to be forward- or backward-compatible
across minor versions.

A future stable release should provide at least one minor-release deprecation
window before removing an ordinary public parameter, method, or fitted
attribute.
