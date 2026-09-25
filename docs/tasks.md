# Task support

## Binary classification

`CERMClassifier` retains the full semantic, optimized, native, autotuned, and
quantized paths. This is the reference and most thoroughly tested task.

## Multiclass classification

The default `multiclass_strategy="ovr"` fits one binary CERM head per class.
`decision_function` returns an `n_samples × n_classes` logit matrix and
`predict_proba` applies softmax coupling. This is the safer alpha default because
each class may select its own state representation and interactions.

`multiclass_strategy="shared"` is an experimental alternative. It learns one
target-neutral nested-quantile state encoder, one selected main/pair design, and
a vector-valued multiclass head. `shared_multiclass_objective="ovr"` fits
independent binary logits, `"multinomial"` fits one joint softmax objective, and
`"auto"` evaluates both endpoints plus same-design coefficient blends. Intermediate
blends receive a fixed 0.005 validation penalty and are selected only when their
validation improvement is clear. In `auto`, a selected non-OVR candidate must
also beat the best OVR validation loss by at least 0.005; otherwise selection
falls back to the OVR candidate family. The final blend is collapsed into one coefficient
matrix, so it adds no inference or artifact-size overhead. Candidate designs reuse one
maximal encoded column bank. `n_jobs` parallelizes the output-head fits, and
compiled inference lowers the complete model to one multi-output C++ library.
The portable and compiled shared formats are pickle-free for supported adapters.

Shared mode currently requires `encoder_kind="quantile"`, no calibration, and a
target-neutral input adapter. Target-aware category quotient or embedding paths
fall back to OVR. A frozen 38-task external real-data evaluation found no
auto-versus-OVR log-loss deterioration after the guard, but `auto` still missed
the preregistered median fit-cost gate and is based on a single fixed holdout per
task. Shared mode therefore remains experimental and is not the default.

Set `multiclass_strategy="error"` to retain a binary-only contract.

## Multilabel classification

The default `representation_strategy="independent"` fits one binary CERM head
per label. `predict_proba` has shape `n_samples × n_labels` and contains positive
probabilities. Outputs are thresholded independently. The default threshold is
0.5. `threshold="auto"` uses a held-out threshold estimate shrunk toward 0.5;
it remains an experimental fit-time/quality tradeoff. Constant labels use a
portable constant program.

`representation_strategy="shared"` learns one finite-state main/pair design and
one logistic head per label. The heads can be fitted in parallel with `n_jobs`,
and the full program compiles to one multi-output native library. It has the same
target-neutral quantile/no-calibration restrictions as shared multiclass. It is
opt-in because some pilot datasets lost micro/macro F1 despite large speed and
artifact-size gains.

Label dependence, classifier chains, and structured joint decoding are not yet
implemented.

## Regression

`CERMRegressor` now uses fused residual V2. The procedure fits a compact
finite-state mean baseline, models the residual distribution through shared
survival thresholds, and promotes a residual correction only when its selection
gate supports it. Depending on sample size and validation outcome, the fitted
`selected_kind_` is `current_mean`, `fused`, or `raw_fused`.

The validated public capacity is intentionally compact: `max_features=24`,
`max_interaction_features=12`, and `max_pairs=4` by default. This vocabulary is
not the same as the historical Ridge estimator's `max_interactions` and
`reg_lambda` controls.

Typed DataFrames support numeric, missing, and categorical regression adapter
paths. `sample_weight` reaches the baseline, residual thresholds/head, selection
gate, small-N cross-fit path, and large-N weighted nested representation.
Portable fused programs use checked JSON/NPZ package state and native inference
uses the fused direct-lookup C++ lowering.

The former finite-state Ridge implementation remains available explicitly as
`CERMRidgeRegressor`. It preserves the old nested-state Ridge search, linear skip
connection, one-standard-error selection, `PortableRegressionProgram`, and
`CompiledRegressionProgram` deployment contracts. `CERMFusedRegressor` remains a
compatibility name for the same fused family now used by `CERMRegressor`.

### Generalized and robust regression

`CERMGeneralizedRegressor` is an experimental loss-backend extension. Its
representation backbone intentionally remains the historical finite-state Ridge
family rather than silently changing with the new top-level single-output
`CERMRegressor` default. It selects that finite-state representation and then
refits a compact loss-specific head on the sparse state design.

- `loss="huber"` adds robust regression for contaminated or heavy-tailed targets.
- `loss="quantile"` estimates a requested conditional quantile through the
  `quantile` parameter.
- `loss="poisson"` estimates a non-negative conditional mean with a log link.
- `loss="multi_quantile"` fits several requested quantiles on one shared
  representation/design. `quantiles` must be strictly increasing. The default
  `non_crossing="cumulative_max"` applies a deterministic monotone correction
  at prediction time.

Poisson accepts fixed additive `offset` and positive `exposure` arrays. The
implementation uses the exact rate-model transformation
`target=y/exposure`, `weight=weight*exposure`; an offset is incorporated through
`exp(offset)`. New-observation offsets/exposures must be supplied to `predict`.

All generalized paths support numeric and typed DataFrames. Their weighting
contract remains that of the historical generalized/finite-state backbone and is
not redefined by the top-level fused default transition.

This surface is alpha-stage. Joblib save/load is supported, but pickle-free
portable export and native compilation are not yet implemented for generalized
heads. Quantile and Poisson head fitting currently rely on scikit-learn solvers;
their fit-time scaling can therefore differ from ordinary fused regression.

## Multi-output regression

`CERMMultiOutputRegressor` accepts a two-dimensional continuous target matrix.
Its current implementation intentionally retains the historical finite-state
Ridge base contract; the top-level `CERMRegressor` default transition does not
silently replace existing multi-output internals. The conservative default
`representation_strategy="independent"` fits one historical Ridge head per
target unless an explicit estimator is supplied.

`representation_strategy="shared"` is the original opt-in compact path. It
standardizes outputs, extracts one deterministic weighted PC1 target for
finite-state selection, and fits one vector-valued Ridge head on the shared
design. It can reduce fit time and model size when outputs use related structure,
but one PC can miss nonlinear structure that is important in other output
directions.

`representation_strategy="shared_subspace"` is a newer experimental research
path. It standardizes the target matrix, estimates a low-dimensional output
subspace from singular values above a Marchenko-Pastur screening edge, and caps
the retained rank at six. Unary finite-state evidence and pair gains are then
screened across those retained directions while one shared representation and
one multi-target Ridge head are kept. If the spectrum has no retained spike, the
implementation falls back exactly to the independent estimator family instead of
forcing a shared representation.

The `shared_subspace` path is **not a speed claim or an automatic/default
selector**. Its current Python implementation scans direction-by-pair evidence
more expensively than the production shared caches. It is unweighted for now and
rejects `sample_weight` explicitly. Numeric and target-neutral typed paths use the
existing regression adaptation machinery; target-aware categorical adaptation
still enters through the historical PC1 bootstrap and is not yet an
R-dimensional preprocessing contract. Joblib save/load is available through the
estimator, but portable/native `shared_subspace` export is not yet implemented.

The default remains `independent`. The historical `shared` path retains its
existing `sample_weight` contract. Native and pickle-free shared multi-output
regression export remain future work.

### Gamma and Tweedie

`CERMGeneralizedRegressor(loss="gamma")` supports strictly positive targets.
`loss="tweedie"` supports non-negative targets for
`1 < tweedie_power < 2`. Both use a log link and exact fixed
`offset`/`exposure` transformations.

### Weighted classification

`sample_weight` is supported for binary classification, OVR multiclass, and
independent multilabel classification. Weighted shared-output public APIs remain
conservative even though common weighted representation primitives are now
available internally.
