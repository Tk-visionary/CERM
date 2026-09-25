# Current limitations

## Statistical scope

- Binary classification is the mature reference path.
- Multiclass defaults to independent one-vs-rest heads. Experimental shared mode supports OVR, joint multinomial, and conservative blended heads. A 38-task external fixed-holdout suite found no guarded `auto` loss versus shared OVR, but this is not a universal safety guarantee and `auto` still exceeded its median fit-cost promotion gate.
- Multilabel supports independent or shared-representation binary heads, but both assume conditionally independent labels and do not model label correlations.
- Multi-output regression supports a safe independent mode and an experimental PC1-guided shared representation. Shared mode can miss output-specific structure and remains opt-in.
- Ordinary regression uses a finite-state ridge head with a linear skip connection. Huber, single/multi-quantile, Poisson, Gamma, and Tweedie heads are experimental; generalized-head native and pickle-free export are not yet implemented.
- Regression and generalized regression accept `sample_weight`; classification, multilabel, calibration, and CRR do not yet expose a complete weight contract. Numeric quantile state boundaries remain feature-distribution based, so integer weights need not equal literal row replication.
- No ranking or survival heads yet.
- Cross-fitted intercept/affine calibration is available but remains optional.
  It increases fit time and has only a four-task repeated-CV pilot so far.
- Cross-fitted block selection is experimental and not the default.
- Residual-Newton category quotients require broader real-data validation.

## Input scope

- Public input is dense ndarray or pandas DataFrame.
- Sparse scipy matrices are rejected explicitly.
- Integer-coded categories require explicit `categorical_features`; automatic
  detection uses pandas dtypes.
- Numeric ndarray missing values are rejected. Missing-state learning currently
  requires a DataFrame.
- Categorical missing behavior is exact for identity states; ordered and Newton
  quotients still use their own reserved missing state.

## Systems scope

- Native compilation requires a working C++ compiler.
- Compiled shared libraries are reloadable through `CompiledProgram.load`, but
  remain architecture- and operating-system-specific.
- Common numeric/categorical compiled program v2 and compiled bundle v2
  packages are pickle-free. Embedding adapters and unsupported custom Python
  category objects fall back to trusted-joblib v1 packages.
- Binary semantic export v2 and task bundle export v1/v2 verify artifacts but
  do not yet reconstruct a complete portable binary estimator.
- Regression export v3 can be reconstructed for inference with
  `PortableRegressionProgram` from JSON/NPZ alone.
- Autotuning results are not persisted by model hash and CPU target.
- Most learning internals remain in `_internal`; they need gradual promotion to
  first-class package modules.

- The exact binary solver fast path uses sklearn's private vendored LIBLINEAR
  ABI and falls back to the public estimator when the ABI is unavailable.
- C values share a design plan but not solver iterates; true warm-start paths
  are not implemented yet.

## Training cache scope

- `NewtonHistogramCache` accelerates repeated L2/ranking queries after one
  state-histogram build.
- The cache is integrated with the estimator but not yet consumed
  automatically by GridSearchCV or an official equal-budget HPO driver.
- Retaining a training cache increases fitted-object memory and may preserve
  aggregate target statistics; it is disabled by default.

## Search limitations

`CERMSearchCV` cache prefiltering currently applies only when the effective
`ranking_kind` is `newton`, `residual_newton`, or `newton_prefilter_mi`, and
when candidate variation is confined to `ranking_l2` and
`max_interaction_features`. Other grids fall back to exhaustive CV. The proxy is a
candidate filter, not a final score, and broad task-level evidence is still
required before treating it as universally safe.

The experimental `ranking_kind="diversified"` pair bank combines a bounded
mutual-information pool with a cross-fitted residual-Newton pool. It is
deliberately excluded from search-cache prefiltering and from all defaults
until broader held-out evidence clears the quality and fit-cost gates.

`preset="balanced"` is an empirically validated candidate-family
reduction, not an algebraically exact TrainingGraph rewrite. It preserves the
pair-feature budget but removes the 4-block, 24-block, and strong cell-
shrinkage primary branches. The default remains `full_exact`. The
`aggressive` profile also caps pair features at 20 and is not recommended as a
general default because validation exposed material task-level risk tails.


## Resource-plan limitations

- `FitResourcePlan` is a conservative structural estimate. It is not a measured
  peak-RSS trace and may over- or under-estimate allocator, BLAS, compiler, or
  multiprocessing overhead on a particular machine.
- Default resource limits reject predicted expensive work before fit. They do
  not silently reduce `max_interaction_features`, block budgets, folds, or candidates.
  Callers may explicitly change the limits or policy.
- Pair and directed block exploration remains superlinear in the effective
  feature budget. Adaptive exact top-k ranking bounds candidate-object memory at
  high dimension, but it does not avoid histogram evaluation for every eligible
  pair.
- The adapted DataFrame matrix is no longer retained after fit, but the current
  public adapter still creates a transient dense float64 matrix. Very wide,
  sparse, or million-row workloads need a future sparse/chunked `AdaptedBatch`.
- `CERMSearchCV.max_full_cv_fits` counts retained candidates times CV splits. It
  does not yet estimate wall time, worker duplication, or nested outer-CV work.
- Exact kNN embedding is guarded by a distance-evaluation budget, but remains a
  high-cost option with near-quadratic scaling.

## Reduced scaling settings

`max_bins < 16`, `subsample < 1`, and `colsample < 1` are explicit model or
selection-space reductions. They are not certified safe-screening passes and
can change the selected model. Version 0.6.0 therefore keeps full-data settings
as defaults and records active reductions in diagnostics. The current
`max_bins` values are restricted to 4, 8, and 16.

Parallelism is available for cross-fit, KNN, search jobs, and shared output-head fits, but nested outer
and inner parallelism can oversubscribe CPU threads.


## 0.12.0a4 weighting boundary

Classification weights are used by feature and pair statistics, validation
risk, conditional-block gains, and final logistic heads. Numeric quantile state
boundaries remain based on the unweighted feature distribution, so integer
weights are not promised to equal literal row replication. Shared multiclass,
shared multilabel, cross-fitted selection with weights, calibration with
weights, and target-aware ordered/newton categorical preprocessing with weights
are explicit errors in this release.

Gamma and compound Poisson-Gamma Tweedie are development features. Their
portable/native exporters are not yet implemented.


## 0.12.0a5 representation-speed boundary

The default ``representation_mode="auto"`` is bitwise-compatible with 0.12.0a4
and retains the full validation search. ``representation_mode="linear"`` is an
explicit model restriction: it skips finite-state main/pair candidate validation
and directly fits the selected linear projection. It is exactly equal to auto
only when auto would select ``max_main_level=0, n_pairs=0``. Nonlinear and
interaction-dominated tasks can require the auto path.
