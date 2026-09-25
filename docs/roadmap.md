# Roadmap

## P0 — statistical evidence

1. Freeze a 30+ task binary IID tabular benchmark.
2. Run equal-wall-clock nested HPO against LightGBM, XGBoost, and CatBoost.
3. Add real high-cardinality categorical tasks for Residual-Newton quotients.
4. Report task-wise calibration, latency, bytes, and failure tails.

## P0 — training architecture

1. Freeze a 30+ task fit-time benchmark with measured peak RSS. Structural
   resource planning and stage timing are implemented in 0.5.7; broad task
   calibration remains open.
2. Prototype a true warm-start LIBLINEAR path through an explicit `init_sol`
   backend; the current specialized operator removes preprocessing but does not
   reuse solver iterates across C values.
3. Connect the estimator-level G/H histogram cache to an official HPO driver.
4. Expand calibration-aware final refitting beyond the four-task pilot.
5. Make missing/category state selection jointly validated with the core.
6. Replace fixed two-holdout selection only after repeated-CV evidence.
7. Derive certified pair/block gain upper bounds so feature scaling can be
   reduced without changing the exhaustive result.
8. Add a segmented design protocol before an implicit semantic solver.
9. Expand the 0.5.8 practical-profile validation from 11 datasets and three
   seeds to a frozen 30+ task suite before considering any default change.

## P1 — Python library

1. Extend the implemented regression `sample_weight` path to binary,
   multiclass, multilabel, CRR, ranking, and block selection.
2. Validate Huber, quantile, and Poisson generalized regression on frozen
   heterogeneous real-data suites; add native and portable generalized heads
   only after the statistical API stabilizes.
3. Promote alpha regression and multiclass heads after broader task-level validation.
4. Validate the experimental shared multiclass/multilabel representation on 30+ heterogeneous tasks before any default change.
5. Validate the joint multinomial and conservative blended objectives on 30+ independent external real tasks before considering any default change; correlated-label alternatives remain later work.
6. Promote semantic learner modules out of `_internal` behind stable protocols.
7. Add a stable loader for exported semantic packages.
8. Add persistent autotune cache keyed by model hash, CPU, compiler, and flags.
9. Add thread controls and measured peak-RSS profiling.
10. Add sparse and chunked `AdaptedBatch` inputs; 0.5.7 removes post-fit matrix
   retention but fit still creates a transient dense adapted matrix.
11. Calibrate resource-plan estimates against measured workloads and hardware.

## P1 — internal program optimizer

1. Keep exact semantic-to-execution equivalence as a property test.
2. Add mixed int8/int16 allocation under a global logit-error budget.
3. Add low-rank pair representations and measured hardware cost profiles.
4. Fuse numeric missing and categorical maps into native execution.

## Promotion gates

A feature moves from experimental to default only when it:

- improves or preserves task-wise risk under repeated CV;
- has a stable fallback;
- preserves semantic/optimized equivalence;
- passes package tests and sklearn estimator checks;
- has measured fit, prediction, memory, and artifact effects.

## Cache-aware search status

`CERMSearchCV` is implemented for grids whose cache-compatible dimensions are
`ranking_l2` and `max_interaction_features`. The final score always comes from full
cross-validation. Current 32-candidate validation retained the exhaustive best
candidate while reducing full fits by 78%, but broad task-level validation is
still required before making prefiltering the default in all search workflows.

Next search extensions:

- reuse adapter and quotient-state construction across structural groups;
- add successive-halving budgets for expensive calibration and cross-fitted selection;
- support multiple scorers and callable refit policies;
- persist fold caches by dataset/schema hash;
- integrate equal-wall-clock baseline search drivers.


## Resource-scaling status

Version 0.5.7 implements conservative preflight plans, default fail-fast
budgets, SearchCV full-fit caps, ephemeral adapted matrices, and adaptive exact
ranking memory. These controls make explosive requests visible and rejectable,
but they do not remove the underlying quadratic pair/block complexity. The next
architecture milestone is a certified screening layer followed by a segmented
or implicit solver representation.

## After 0.6.0 scaling controls

- Task-adaptive or certified safe choices for state resolution and feature/row reduction.
- Supervised feature screening or repeated subspaces to reduce colsample tail risk.
- Arbitrary nested bin schedules beyond 4/8/16 after IR versioning.
- Explicit nested-parallelism and thread-pool policy.
- Solver iteration/time diagnostics and a warm-start SolveGraph.
