# Advanced usage and implementation notes

This page preserves the detailed controls and implementation notes from the 0.6.1 quality build. Public API guarantees are limited to the symbols documented in `api_stability.md`.

## Intuitive model controls

The main controls follow the same design principle as GBDT libraries: each
parameter represents a recognizable source of model or search complexity.

| Parameter | Meaning | Similar GBDT intuition |
|---|---|---|
| `preset` | Candidate schedule | accuracy/speed preset |
| `max_features` | Maximum selected main features | feature budget |
| `max_bins` | Finite-state resolution, currently 4/8/16 | `max_bin` |
| `max_interaction_features` | Features entering quadratic pair/block search | interaction feature budget |
| `max_interactions` | Maximum retained conditional block terms | `max_leaves`-like capacity cap |
| `interaction_order` | `1` for main effects, `2` for pair/block interactions | `max_depth`-like interaction depth |
| `reg_lambda` | Model-wide L2 penalty, or `"auto"` | `reg_lambda` |
| `subsample` | Fraction of rows used for architecture selection; final refit uses all rows | row sampling budget |
| `colsample` | Fraction of adapted features retained for the fitted model | column sampling budget |
| `n_jobs` | Cross-fit/KNN parallelism; SearchCV has its own `n_jobs` | thread/job budget |
| `max_memory_mb` | Pre-fit structural memory budget | device/RAM budget |

`preset="accurate"` is the default and resolves to the historical
`full_exact` candidate family. `preset="balanced"` resolves to the validated
`practical` schedule: it keeps the pair-search range but reduces ordinary
primary candidates from 15 to 9.

```python
accurate = CERMClassifier(preset="accurate")
balanced = CERMClassifier(preset="balanced")
main_effects = CERMClassifier(interaction_order=1)
bounded = CERMClassifier(
    preset="balanced",
    max_interaction_features=20,
    max_interactions=16,
)
fixed_regularization = CERMClassifier(reg_lambda=5.0)
```

The fitted `resolved_params_` records the concrete search profile, effective
pair budget, interaction cap, regularization, and memory limit. Explicit
legacy parameters (`pair_feature_limit`, `search_profile`, and
`max_estimated_peak_memory_mb`) remain supported; the new names take precedence
when both are supplied.

`max_bins` is a real state-resolution control. Values 4, 8, and 16 build
physical hierarchies `(4,)`, `(4, 8)`, and `(4, 8, 16)` respectively, and the
choice propagates through block states, semantic IR, optimized graphs, and native
code generation. `subsample < 1` changes selection data only; the selected
architecture is refit on all rows. `colsample < 1` changes the fitted feature
subspace and is persisted across every prediction backend.

The reduced settings are explicit opt-ins. In the 0.6.0 validation grid they
provided material speedups, but high-dimensional and interaction-dominant tasks
showed non-negligible quality tails. The defaults remain `max_bins=16`,
`subsample=1.0`, and `colsample=1.0`. Inspect `active_reductions` and
`search_semantics` in `resolved_params_` or `fit_diagnostics_`.

Typed DataFrames, exact category states, missing indicators, embedding adapters,
joblib persistence, versioned export manifests, checksum verification, float32,
and bounded int16 native lowering are supported.

Optional cross-fitted affine calibration preserves score ordering while
adjusting logit scale and intercept:

```python
calibrated = CERMClassifier(
    calibration="affine",
    calibration_folds=3,
).fit(X_train, y_train)

print(calibrated.calibration_result_)
```

Finite-state Newton histograms can be retained during fitting or built later
for repeated ranking and regularization sweeps:

```python
model = CERMClassifier(cache_training_statistics=True).fit(X_train, y_train)
rankings = model.training_cache_.sweep_pair_rankings(
    [0.5, 1.0, 5.0, 20.0],
    max_pairs=20,
)
model.training_cache_.save("training_cache.npz")
```

```python
from cerm import verify_export

manifest = model.export("exported_model")
verify_export(manifest)
```

This remains a research package. Binary classification and dense inputs are the
current public scope. See `docs/limitations.md`, `docs/compatibility.md`,
`SECURITY.md`, and `docs/roadmap.md`.

## Cache-aware hyperparameter search

`CERMSearchCV` uses finite-state Newton histograms to prefilter compatible
`ranking_l2 × max_interaction_features` candidates, then evaluates the retained
candidates with ordinary cross-validation. The proxy never becomes the final
model-selection score.

```python
from cerm import CERMClassifier, CERMSearchCV

search = CERMSearchCV(
    CERMClassifier(ranking_kind="newton"),
    {
        "ranking_l2": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        "max_interaction_features": [4, 8, 16, 24],
    },
    cv=3,
    prefilter_top_k=8,
).fit(X_train, y_train)

print(search.best_params_)
print(search.search_diagnostics_)
```

Set `cache_prefilter=False` for exhaustive behavior. Candidates skipped by the
proxy remain visible in `cv_results_` with `prefilter_selected=False`.

### Experimental candidate diversification

`ranking_kind="diversified"` adds a bounded cross-fitted residual-Newton
candidate pool to the default mutual-information pool. It uses every training
row and changes only candidate discovery; it does not discard rows through a
GOSS-style shortcut. The mode is opt-in, is not eligible for the search-cache
proxy, and should be used only when a task-specific validation protocol can
justify the extra fit work:

```python
model = CERMClassifier(
    ranking_kind="diversified",
    search_effort="balanced",
    feature_budget=64,
    interaction_search_features=24,
)
```

The current evidence bundle keeps this mode experimental because the targeted
synthetic study found a quality regression tail without a compensating loss
improvement. See `evidence/candidate_expansion/REPORT.md` in the bundle.

## Training-graph optimization

CERM separates statistical model selection from exact fit-time rewrites. The
training graph reuses maximal hierarchical and block designs, evaluates smaller
candidates through column views, and fuses finite-state G/H statistics. Version
0.5.6 also lowers one-hot state designs directly to CSR, reuses ephemeral
backbone fit artifacts, writes semantic and block banks into preallocated
buffers, and invokes the validated binary LIBLINEAR kernel through a specialized
operator. Version 0.5.7 adds adaptive exact top-k ranking for extreme
pair/block searches, releases transformed training matrices after fitting, and
adds resource plans that stop predicted pair, block, memory, kNN, and full-CV
explosions before expensive work begins. These rewrites preserve the selected
semantic model and prediction function. Fit diagnostics expose stage times,
resource estimates, candidate/design counts, and fitted-object retention.


## Resource planning and scaling guards

CERM estimates structural fit work before target-dependent training. The plan is
inspectable and never silently changes the statistical search space:

```python
model = CERMClassifier(
    max_features=96,
    max_interaction_features=80,
)
plan = model.estimate_fit_resources(X_train)
print(plan.to_dict())
```

The default `resource_policy="raise"` stops a fit before execution when the
conservative plan exceeds any configured budget:

```python
model = CERMClassifier(
    max_memory_mb=4096,
    max_pair_evaluations=2_000_000,
    max_block_evaluations=5_000_000,
    max_knn_distance_evaluations=100_000_000,
)
```

Use `resource_policy="warn"` to continue with an explicit warning, or
`resource_policy="ignore"` only when the caller manages resources externally.
The estimates are structural upper-envelope diagnostics rather than measured
peak RSS.

`CERMSearchCV` separately limits the number of retained candidates multiplied by
CV splits:

```python
search = CERMSearchCV(
    model,
    param_grid,
    cv=5,
    max_full_cv_fits=128,
    budget_policy="raise",
)
```

High-dimensional rankings use bounded exact top-k operators only beyond the
regime where they reduce memory. Common feature counts retain the faster
materialized path, so the scaling protection does not impose a permanent Python
heap cost.

## Explicit scaling controls (0.6.0)

```python
reduced = CERMClassifier(
    preset="balanced",
    max_bins=8,
    subsample=0.75,
    colsample=0.85,
    n_jobs=2,
).fit(X_train, y_train)

print(reduced.fit_diagnostics_.active_reductions)
print(reduced.fit_diagnostics_.search_semantics)
```

These controls are real model/training changes, not cosmetic aliases. Validate
them on the deployment split strategy before use. `CERMSearchCV.n_jobs` controls
outer candidate-fold evaluation separately; avoid combining large inner and
outer job counts unless thread resources are managed explicitly.
