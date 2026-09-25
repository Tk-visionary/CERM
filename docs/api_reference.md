# API reference

This page describes the user-facing Python surface at a practical level. CERM
follows scikit-learn estimator conventions: constructor arguments describe
configuration, `fit` learns state, and learned public attributes end in `_`.
Everything under `cerm._internal` and `cerm._compat` is private.

## Parameter-layer helpers

These top-level helpers keep predictive HPO, search-family choices, resource
controls, and convenience aliases separate:

```python
from cerm import (
    get_convenience_params,
    get_resource_params,
    get_search_params,
    get_tunable_params,
)

get_tunable_params(model)
get_search_params(model)
get_resource_params(model)
get_convenience_params(model)
```

`get_user_params()` remains an estimator method for compatibility with the
earlier convenience-oriented parameter view.

## Top-level estimators

### `CERMClassifier`

```python
from cerm import CERMClassifier
model = CERMClassifier().fit(X, y)
```

Use for binary or multiclass classification. Binary classification uses one CERM
program. Multiclass defaults to independent one-vs-rest programs; the shared
representation is experimental and opt-in.

Common methods include `fit`, `predict`, `predict_proba`, `decision_function`,
`get_params`, `set_params`, `save`, `export`, and `compile_native`.

### `CERMRegressor`

```python
from cerm import CERMRegressor

model = CERMRegressor().fit(X, y, sample_weight=weights)
prediction = model.predict(X_test)
```

`CERMRegressor` is the ordinary regression estimator and now uses fused residual
V2 by default. The procedure contains a finite-state mean baseline and promotes
a residual-distribution correction only when its fitted gate selects one. The
fitted `selected_kind_` records `current_mean`, `fused`, or `raw_fused`.

Validated default capacity:

- `n_bins=10`
- `max_bins=16`
- `max_features=24`
- `max_interaction_features=12`
- `max_pairs=4`

`get_tunable_params(CERMRegressor())` returns exactly those five model controls.
`get_search_params()` and `get_resource_params()` are currently empty for this
surface because the V2 candidate family and execution policy remain fixed
implementation details rather than fake HPO knobs.

Common methods:

- `fit(X, y, sample_weight=None)`
- `predict(X)`
- `predict_survival(X)`
- `get_params(deep=True)` / `set_params(**params)`
- `get_feature_names_out()`
- `get_user_params()` / `explain_params()` / `parameter_summary()`
- `save(path)` / `CERMRegressor.load(path)`
- `export(path)`
- `compile_native(path)`

Portable fused exports are loaded with `PortableFusedRegressionProgram` and
native compilation returns `CompiledFusedRegressionProgram`.

### `CERMFusedRegressor`

`CERMFusedRegressor` remains available as an explicit compatibility name for the
same fused residual V2 family. Its constructor and fitted statistical semantics
match the new default regression family. Existing code that adopted
`CERMFusedRegressor` before the default switch does not need to rename the
estimator.

### `CERMRidgeRegressor`

```python
from cerm import CERMRidgeRegressor

legacy = CERMRidgeRegressor().fit(X, y)
```

`CERMRidgeRegressor` preserves the former `CERMRegressor` finite-state Ridge
implementation. Its historical constructor vocabulary includes controls such as
`max_interactions`, `reg_lambda`, presets, row/column subsampling, and
`include_linear`. Its previous semantic export, `PortableRegressionProgram`,
`CompiledRegressionProgram`, and native runtime remain available for explicit
compatibility and controlled Ridge-vs-fused comparisons.

There is deliberately no `regression_strategy` switch. Ridge and fused use
different capacity semantics, so separate estimator identities avoid parameters
that silently change meaning by engine.

### `CERMGeneralizedRegressor`

Use for objective-specific continuous outcomes. Supported contracts are
maintained in [task support](tasks.md); current objectives include quantile,
Huber, Poisson, Gamma, Tweedie, and multi-quantile modes.

### Multi-output estimators and search

`CERMMultiLabelClassifier` handles multilabel indicator targets and
`CERMMultiOutputRegressor` handles multi-output continuous targets. Their current
internal/base-estimator contracts are not silently migrated by the top-level
single-output regression default change. Use an explicit base estimator when the
choice matters.

`CERMSearchCV` follows the familiar fit/predict estimator pattern. Ordinary CERM
estimators expose standard `get_params` / `set_params` and can be cloned by
scikit-learn.

## Common model inspection

```python
from cerm import inspect_model, feature_table, interaction_table, model_summary

inspection = inspect_model(model)
print(inspection.summary)
```

`model_summary` is task-neutral and works with the default fused regressor. Fused
regression reports its engine, selected branch, fit time, adapted width, and
model-size information through fitted diagnostics.

`feature_table`, `interaction_table`, and `structure_table` expose explicit
finite-state structure when the fitted estimator has one globally meaningful
selected representation. They remain especially useful for classifiers,
`CERMRidgeRegressor`, and generalized finite-state representations. A composite
or fused procedure may return a compact/empty structure view rather than invent
one unified feature ranking.

## Configuration views

CERM exposes multiple views on purpose:

| API | Use it for |
|---|---|
| `estimator.get_params()` | Complete scikit-learn constructor state, cloning, pipelines, and grid search. |
| `get_tunable_params(estimator)` | Direct numeric/structural parameters recommended for HPO. |
| `get_search_params(estimator)` | Search-family controls where publicly exposed. |
| `get_resource_params(estimator)` | Execution/resource controls where publicly exposed. |
| `get_convenience_params(estimator)` | Compact human-language view. |
| `estimator.fit_diagnostics_` | Detailed post-fit engine/search/representation record. |

Historical estimator families additionally expose compatibility aliases and
`resolved_params_`. The default fused regressor intentionally has a smaller
constructor vocabulary instead of inheriting Ridge-only aliases.

The exhaustive constructor inventory is machine-checked against live signatures
in [Public parameter inventory](parameter_inventory.md).

## DataFrame schema and sample weights

For fitted DataFrame inputs, CERM stores the fitted schema and validates
prediction columns. Reordered columns are restored to fitted order; missing or
unexpected columns fail explicitly. Default fused regression supports numeric,
categorical, and missing-value regression adapter paths. Non-empty
`embedding_features` is currently rejected until regression embedding has a real
transformation contract.

Fused regression propagates sample weights through its mean baseline,
threshold/residual head, validation gate, small-N cross-fit path, and weighted
large-N nested representation. See [Default regression](fused_regression.md) for
the precise integer-frequency vs real-weight semantics.

## Persistence and deployment

CERM separates fitted statistical models from deployment lowering.

- `save` / `load` provide Python persistence.
- `export` writes versioned, checksummed portable state where supported.
- `compile_native` builds the supported platform-specific native predictor.
- historical finite-state programs may additionally expose optimization passes.

Because default regression changed model family, distinguish the program types:

- `CERMRegressor` / `CERMFusedRegressor` -> `PortableFusedRegressionProgram` and `CompiledFusedRegressionProgram`;
- `CERMRidgeRegressor` -> `PortableRegressionProgram` and `CompiledRegressionProgram`.

See [compatibility](compatibility.md) and [API stability](api_stability.md).

## Alpha-stage regression transition

CERM is still alpha, so the ordinary regression default has intentionally moved
to fused residual V2 before a stable API freeze. Code requiring the former
`CERMRegressor` behavior should use `CERMRidgeRegressor`. The next external
regression benchmark is a quality/cost-tail audit of the new default and a guide
to where Ridge may still be preferable; it is not a prerequisite for making
fused V2 accessible through the ordinary regression name.
