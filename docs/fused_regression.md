# Default regression: fused residual V2

`CERMRegressor` now uses CERM's fused residual regression V2 procedure by default.
`CERMFusedRegressor` remains available as an explicit compatibility name for the
same fused family. The historical finite-state Ridge estimator is preserved as
`CERMRidgeRegressor`.

```python
from cerm import CERMRegressor, CERMRidgeRegressor

model = CERMRegressor().fit(X_train, y_train)
prediction = model.predict(X_test)

legacy = CERMRidgeRegressor().fit(X_train, y_train)
```

This is a deliberate default change. Fused V2 is not just an unconditional
nonlinear correction: it contains the historical finite-state mean baseline and
can select `current_mean` when the residual correction does not earn promotion.
The previous Ridge estimator remains available for exact compatibility,
controlled comparisons, and legacy deployment workflows.

## Why there is no `regression_strategy` switch

The two engines do not assign the same meaning to their capacity parameters. The
historical Ridge estimator uses a broader finite-state vocabulary such as
`max_interactions`, `reg_lambda`, presets, subsampling, and column sampling. The
validated fused V2 configuration instead uses a residual-distribution candidate
bank centered on `max_features=24`, `max_interaction_features=12`, and
`max_pairs=4`.

Putting both engines behind one constructor would make several parameters valid
only for one strategy or silently reinterpret the same name. CERM therefore
uses explicit estimator identities:

- `CERMRegressor`: default fused residual V2;
- `CERMFusedRegressor`: compatibility name for fused residual V2;
- `CERMRidgeRegressor`: historical finite-state Ridge.

## Public defaults

| Parameter | Default | Meaning |
|---|---:|---|
| `n_bins` | 10 | Residual-survival threshold bins used by V2. |
| `max_bins` | 16 | Maximum nested resolution for the baseline/shared large-sample representation. |
| `max_features` | 24 | Feature cap for the baseline/shared large-sample candidate bank. |
| `max_interaction_features` | 12 | Retained features admitted to shared large-sample pair ranking. |
| `max_pairs` | 4 | Maximum shared-pair prefix considered by the large-sample fused branch. |

These controls are not global knobs for every internal V2 branch. When
`fit_diagnostics_["selection_mode"] == "small_n_crossfit_raw"`, the raw
correction keeps its separately validated 4/8/16 multi-resolution layout and
internal raw-pair cap. Those small-sample settings remain fixed implementation
details.

The remaining constructor arguments configure random state and the regression
typed-DataFrame contract. `embedding_features` is retained as a reserved slot
for regression-family compatibility, but non-empty embedding configuration is
explicitly rejected until the regression adapter has a real embedding
transformation contract. Numeric, categorical, and missing-value DataFrame
columns are supported.

## HPO and search layers

```python
from cerm import CERMRegressor, get_resource_params, get_search_params, get_tunable_params

get_tunable_params(CERMRegressor())
# {
#   "n_bins": 10,
#   "max_bins": 16,
#   "max_features": 24,
#   "max_interaction_features": 12,
#   "max_pairs": 4,
# }

get_search_params(CERMRegressor())
# {}

get_resource_params(CERMRegressor())
# {}
```

The empty search/resource views are intentional. The V2 candidate family,
small-sample promotion gate, threshold-head optimizer, and execution policy are
fixed implementation details rather than pretend HPO parameters.

For legacy Ridge HPO, construct `CERMRidgeRegressor` and use its historical
parameter surface.

## Typed input and sample weights

`fit(X, y, sample_weight=None)` accepts dense numeric arrays and pandas
DataFrames with the regression categorical/missing adapter. A fitted DataFrame
schema is checked at prediction time. Portable fused export/load preserves
JSON-native DataFrame column labels such as integer labels rather than coercing
them to strings. `get_feature_names_out()` returns the adapted feature names.

Sample weights reach baseline regression, threshold fitting,
validation/candidate selection, residual correction, small-N cross-fit
promotion, final fitting, and the large-N shared encoder and unary/pair MI
ranking.

The nested representation has an explicit two-part weight contract:

- `None` and exact all-one weights use the historical unweighted representation;
- zero-weight rows contribute no representation mass;
- non-negative integer weights use frequency semantics, matching explicit row duplication for weighted quantiles and finite-state MI at a fixed training set/structure;
- positive non-integer real weights use midpoint weighted-ECDF quantiles and weighted contingency mass.

Whole-estimator row-duplication equivalence is not promised because adaptive
splits can place duplicate copies differently and regularized objectives treat
total weight as statistical mass.

## Persistence and deployment

`CERMRegressor` exposes the fused contracts directly:

- `save` / `load` for Python persistence;
- `export` for the portable fused semantic program;
- `PortableFusedRegressionProgram.load(...)` for portable inference;
- `compile_native` for the platform-specific native predictor;
- `CompiledFusedRegressionProgram` as the returned native wrapper.

`CERMRidgeRegressor` preserves the historical Ridge persistence, semantic, and
native contracts through the previous implementation.

## Compatibility status

This default switch is intentionally made while CERM remains pre-stable. Code
that requires the former `CERMRegressor` behavior should migrate explicitly to
`CERMRidgeRegressor`. `CERMFusedRegressor` remains available so existing opt-in
fused code continues to work.

The next validation gate is not required to expose fused regression—the default
is now fused V2—but to characterize the external quality/cost tail and determine
where users may still prefer `CERMRidgeRegressor`.
