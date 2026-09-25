# Model, search, and resource parameters

CERM exposes four distinct parameter layers. Keeping these layers separate makes GridSearchCV/Optuna studies reproducible and prevents execution settings from being mistaken for model hyperparameters.

## 1. Preferred HPO surface

Use the direct numeric/structural parameters below for predictive tuning:

| Parameter | Meaning | Suggested search form |
|---|---|---|
| `max_bins` | Finite-state resolution | categorical `{4, 8, 16}` |
| `max_features` | Main-effect feature budget | integer/log scale |
| `max_interaction_features` | Features admitted to pair/block search | integer |
| `max_interactions` | Maximum retained interaction terms | integer including 0 |
| `interaction_order` | 1=main only, 2=pair/block interactions | categorical `{1, 2}` |
| `reg_lambda` | L2 shrinkage; positive float for HPO | log scale |
| `subsample` | Rows used for architecture selection | float `(0, 1]` |
| `colsample` | Adapted columns retained | float `(0, 1]` |

Example:

```python
from cerm import CERMClassifier, get_tunable_params

model = CERMClassifier(
    max_bins=8,
    max_features=64,
    max_interaction_features=24,
    max_interactions=16,
    reg_lambda=1.0,
    random_state=42,
)

print(get_tunable_params(model))
```

For Optuna, prefer numeric primitives directly:

```python
def suggest_cerm(trial):
    return {
        "max_bins": trial.suggest_categorical("max_bins", [4, 8, 16]),
        "max_features": trial.suggest_int("max_features", 16, 128, log=True),
        "max_interaction_features": trial.suggest_int("max_interaction_features", 8, 48),
        "max_interactions": trial.suggest_int("max_interactions", 0, 32),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }
```

`reg_lambda="auto"` remains a safe fixed/default behavior, but equal-budget HPO studies should generally search explicit numeric values so the tuning budget is observable.

## 2. Search-family controls

`preset`, `search_profile`, ranking strategy, selection strategy, and multiclass strategy choose *how* CERM searches or fits. They are algorithmic/categorical choices rather than a continuous complexity scale. In particular, `search_effort="balanced"` is a convenience alias for a candidate-family preset; it must not be interpreted as a numeric search budget.

Use `get_search_params(model)` to inspect this layer. A future true `search_budget` should only be introduced if candidate prefixes are nested and the number has a precise evaluation-budget meaning.

## 3. Convenience aliases

The human-language aliases remain supported for interactive use and backwards compatibility:

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

They are exact aliases and conflicting specifications still raise an error. Use `get_convenience_params(model)` to inspect them. They are no longer the recommended HPO vocabulary.

## 4. Resource controls

`n_jobs`, memory limits, `resource_policy`, and pair/block/KNN evaluation caps are execution guards. They should normally remain fixed while tuning predictive quality. Use `get_resource_params(model)` to inspect this layer.

`get_user_params()` remains for API compatibility and returns the old convenience-oriented view; new benchmark and HPO code should use the layer-specific helpers above.

## Historical and advanced parameter names

The sections below document the compatibility surface and advanced controls.
They remain supported for reproducibility, `GridSearchCV`, saved configurations,
and research experiments.

### `preset`

- `"accurate"`: historical exhaustive candidate family. This is the default.
- `"balanced"`: validated reduced candidate family. Pair search is unchanged.

The legacy `search_profile` parameter remains available as an explicit override.

### `max_features`

Maximum number of main features retained by the finite-state backbone.

### `max_bins`

Finite-state resolution. The current supported values are 4, 8, and 16.
This is a physical architecture parameter rather than an alias:

- `4` builds level `(4,)`.
- `8` builds levels `(4, 8)`.
- `16` builds levels `(4, 8, 16)`.

The choice changes state encoders, residual columns, pair/block states, semantic
IR, optimized graphs, native lowering, and design dimension.

### `max_interaction_features`

Maximum number of selected features entering pair and directed block ranking.
The undirected pair count grows as `p * (p - 1) / 2`. It overrides the legacy
`pair_feature_limit` alias.

### `max_interactions`

Maximum number of conditional block terms considered and retained. `0` disables
blocks. The candidate prefix schedule is clipped to this value.

### `interaction_order`

- `1`: main finite-state effects only.
- `2`: pair and conditional block interactions are enabled.

### `reg_lambda`

- `"auto"`: evaluate the historical regularization candidates.
- positive float: use one model-wide L2 penalty, with `C = 1 / reg_lambda`.

### `subsample`

Fraction of rows used for architecture and candidate selection. Sampling is
stratified and deterministic for a fixed `random_state`. The selected
architecture is then refit on all training rows. This differs from ordinary
per-tree GBDT sampling and is reported as `selection_row_subsample`.

### `colsample`

Fraction of adapted columns retained for the fitted model. The subset is
selected deterministically and is used consistently by semantic, optimized,
native, saved, and reloaded prediction paths. It changes the model space.

### `n_jobs`

Parallel job count for cross-fitted selection and exact KNN adapter operations.
`CERMSearchCV.n_jobs` separately controls outer candidate-fold evaluation.
`None` and `1` are serial in CERM's own scheduling; negative values follow
joblib conventions. Avoid nested large job counts.


### `multiclass_strategy` and `shared_multiclass_objective`

`multiclass_strategy="ovr"` is the default and lets each class select an
independent binary CERM representation. `multiclass_strategy="shared"` reuses one
finite-state main/pair design across all classes. In shared mode:

- `shared_multiclass_objective="ovr"` fits independent binary logits on the shared design.
- `"multinomial"` fits one joint softmax objective.
- `"auto"` evaluates OVR, multinomial, and same-design coefficient blends with weights
  0.25, 0.5, and 0.75. Intermediate blends pay a fixed 0.005 selection penalty to
  limit validation overfitting. A selected non-OVR candidate must additionally
  improve penalized validation log loss over the best OVR candidate by at least
  0.005 or `auto` falls back to the OVR candidate family. The selected
  coefficients are fused before inference.

`fit_diagnostics_` reports `pre_guard_multiclass_objective`,
`non_ovr_validation_advantage`, `objective_guard_margin`, and
`objective_guard_applied`. Shared mode remains experimental and is not the
default.

### `representation_strategy` and `class_specific_budget`

These parameters apply only to `multiclass_strategy="shared"`.
`representation_strategy="baseline"` preserves the 0.10 shared main/pair basis
and remains the default. `"adaptive"` screens residual class-state support on
task training rows, evaluates the fixed budget prefixes 0/4/8/12 on a
deterministic stratified internal split, and refits the selected prefix on all
training rows. `class_specific_budget` is a non-negative global cap; screening
also caps support at three operators per class. A prefix must improve validation
log loss by at least 0.001 and stay within the 2x semantic-size gate, otherwise
the adaptive path is an exact no-op.

`fit_diagnostics_` records the selected budget, every validation score, and the
training-only improvement. These diagnostics select an algorithmic candidate;
they are not a replacement for an untouched external test set. Adaptive models
use pickle-free finite-state IR v3/shared-program v2. Baseline models continue
to export v2/v1, and those older formats remain loadable.

### `max_memory_mb`

Pre-fit structural memory budget. It overrides the legacy
`max_estimated_peak_memory_mb`. The estimate is conservative, not a measured RSS
hard limit.

## Resolution and diagnostics

Call `resolve_params()` before fitting or inspect `resolved_params_` after fit.
Both resolved parameters and `fit_diagnostics_` expose:

- `active_reductions`
- `search_semantics`
- effective interaction budgets
- full and retained adapted-feature counts
- selection rows
- state resolution and job count
- candidate/design/solver counts and resource risk

The default has `active_reductions=()` and `search_semantics="full"`.

## Operational guidance

The 0.6.0 validation did not support changing the global defaults. Individual
reductions produced median speedups but material worst-task losses on some
high-dimensional and interaction-dominant datasets. Use reduced values only
with task-level validation that matches the production split and metric.
