# CERM binary classification production maturity audit

Audit integration base: `0a494a74d37163c5336d4404298dec3e48a8cc06`

Scope: current `CERMClassifier` robustness, execution, serialization, scikit-learn contracts, and the integrated weighted nested-representation contract. This audit does not add a new classification algorithm and does not change the unweighted/default model search or prediction semantics.

## Executive result

The binary classifier has a broad production-shaped contract: ndarray and DataFrame validation, typed categorical/missing adapters, string/bool labels, semantic and optimized programs, native compilation, estimator persistence, compiled-package persistence, model-size diagnostics, and scikit-learn parameter surfaces.

The regression-driven nested representation refactor is genuinely active in binary classification. `QuotientBlockCERM._make_backbone()` returns `_NestedEngineHierarchicalResidualCERM`, whose MRO includes `NestedRepresentationMixin`; binary main/pair 4/8/16 code materialization is therefore routed through the shared nested engine. The historical unweighted hierarchical implementation is retained as the compatibility authority, and unweighted/unit-weight nested fits delegate to it.

The audit fixed four contract bugs: semantic-alias clone safety, zero-weight preprocessing leakage, silent partially weighted target-aware typed preprocessing, and two dropped weights in the final hybrid refit.

## Maturity matrix

| Area | Case | Audit status | Evidence / contract |
|---|---|---|---|
| Input | dense ndarray | PASS | Public validation + focused execution tests |
| Input | pandas numeric | PASS | Exact fitted schema |
| Input | mixed categorical | PASS | Typed adapter state/quotient path |
| Input | missing / all-missing numeric | PASS | Median + missing-state path |
| Input | unseen category | PASS | Deterministic fitted fallback; no refit |
| Input | reordered DataFrame columns | PASS | Restored to fitted schema |
| Input | missing / extra DataFrame columns | EXPLICIT REJECT | Schema validation |
| Target | bool / int / string labels | PASS | LabelEncoder round-trip |
| Target | severe imbalance | PASS WITH BOUNDARY | Stratified selection; minimum class support applies |
| Target | constant / invalid / missing | EXPLICIT REJECT | Binary target validation |
| Weights | `None` | PASS | Historical path retained |
| Weights | all ones | PASS | Canonicalized to historical nested path |
| Weights | zero rows | PASS / FIXED | Removed before public preprocessing; ignored by weighted nested primitives |
| Weights | positive integer | PASS WITH DEFINED SCOPE | Frequency semantics: quantile/MI/fixed representation match explicit row duplication |
| Weights | positive non-integer real | PASS WITH DEFINED SCOPE | Midpoint weighted-ECDF quantiles + weighted contingency mass |
| Weights | invalid length / negative / NaN / all-zero | EXPLICIT REJECT | Public/internal validation |
| Weights + typed | identity categorical | PASS | Target-neutral categorical states |
| Weights + typed | ordered/newton/auto target-aware/embedding | EXPLICIT REJECT | Avoids silent partial weighting |
| Degeneracy | constant / near-constant / duplicate values | PASS | State collapse handled safely |
| Degeneracy | `p >> n` / `n >> p` | PASS | Budgeted representation / normal dense path |
| Execution | predict / predict_proba / decision_function | PASS | Public fitted/schema checks |
| Execution | batch=1 | PASS | Focused test |
| Execution | semantic / optimized parity | PASS | Existing parity + focused audit |
| Execution | native compile parity | PASS WHEN COMPILER PRESENT | Existing native contract |
| Serialization | estimator save/load | PASS | Atomic joblib persistence |
| Serialization | semantic export integrity | PASS | SHA/byte-count verification |
| Serialization | compiled package save/load | PASS | Versioned compiled contract |
| sklearn | constructor clone | PASS | Existing contract |
| sklearn | semantic alias set_params -> clone | FIXED | Aliases synchronized immediately |
| sklearn | GridSearchCV / Pipeline | PASS BY FOCUSED TEST | Representative smoke coverage |
| Representation | shared nested engine active | PASS | Shared mixin + retained legacy authority |
| Representation | integer-weight fixed-structure duplication | PASS | Encoder/MI/pair/final-head focused tests |

## Fixed issues

### 1. Semantic alias mutation could break sklearn cloning

After `set_params(state_detail="coarse")`, the semantic alias previously changed while `max_bins` could remain stale until fit validation. `set_params` now refreshes semantic aliases immediately. Default estimators are unaffected.

### 2. Zero-weight rows affected preprocessing

Zero weights were honored by objectives/statistics but rows could still reach representation and typed preprocessing. Public binary fit now removes exact zero-weight rows after validating original alignment and revalidates the effective binary target before preprocessing. The weighted nested encoder independently excludes zero-mass rows as well.

### 3. Target-aware typed preprocessing could be only partially weighted

`category_policy="auto"` could resolve to a target-aware quotient while the binary typed adapter did not provide a complete weighted contract; embedding preprocessing had the same issue. Weighted binary fitting now rejects target-aware categorical/embedding preprocessing explicitly. Identity categorical states remain supported.

### 4. Hybrid final refit dropped weights

Final stable block ranking omitted `sample_weight`, and final block centering/scaling used the unweighted helper. Weighted final fits now pass weights to stable ranking and use `BlockColumnBank(..., sample_weight=...)`. The unweighted final path remains the historical implementation.

## Sample-weight semantics after weighted nested integration

The nested quantile representation now has an explicit dual contract rather than an accidental partially weighted path.

- `sample_weight=None` and exact all-one weights execute the historical unweighted nested representation.
- Zero-weight observations contribute no representation or contingency mass.
- Non-negative **integer** weights are frequency weights. At a fixed training set/structure, quantile boundaries reproduce NumPy linear/Type-7 quantiles on explicitly repeated rows, and finite-state MI/pair statistics reproduce repeated-row contingency mass.
- Positive **non-integer real** weights use scale-invariant midpoint weighted-ECDF interpolation for numeric boundaries and weighted contingency mass for MI.

Integer frequency semantics intentionally do not imply global scale invariance: multiplying integer counts can change a finite-sample Type-7 quantile because it changes the virtual sample size. Conversely, the non-integer real-weight representation branch is scale-invariant. This distinction is part of the contract and should not be inferred from dtype alone as a promise that every full estimator fit is invariant to weight rescaling.

Most importantly, **full adaptive estimator training is not claimed to equal literal row duplication**. Candidate selection uses train/validation splits on the supplied rows; explicitly duplicated copies may be allocated to different folds. Regularized fitting also treats weight mass as statistical mass. The exact duplication claim is therefore limited to the weighted representation/statistical objective at a fixed training set/structure. The existing whole-estimator sample-weight equivalence xfail can remain for that reason, but its old explanation (unweighted quantile boundaries) no longer applies.

## Exact semantic-preservation evidence

The shared-representation migration predates this audit. Commit `252cfcc9a19a3facc165578cc8b79d3c2a336247` recorded bitwise prediction parity against the prior package for binary numeric and weighted cases when the task-neutral nested representation core was introduced. The integrated weighted facade goes further: unweighted and unit-weight nested fits delegate to a byte-preserved historical implementation, while focused tests compare shared/legacy code layout and integer-weight fixed-structure results.

## Unresolved limitations

1. **No whole-estimator row-replication invariant.** Fixed representation/objective integer-frequency equivalence is tested, but adaptive split histories are not identical to fitting a materialized duplicated dataset.
2. **Portable binary semantic export has integrity verification but no direct public `SemanticProgram.load` predictor API.** Estimator and compiled-package load paths exist.
3. **Standalone numeric lowered artifacts do not independently persist the estimator's original ndarray width.** The public estimator enforces it; typed DataFrame programs retain schema.
4. **Sparse matrices remain unsupported by design.**
5. **Embedding + sample weight remains explicitly unsupported rather than partially weighted.**
6. **Newton target-dependent encoder weighting is not redefined by the weighted quantile work.** The new exact frequency contract applies to the common nested quantile representation.

## Focused tests

`tests/test_binary_maturity_contract.py`, `tests/test_weighted_binary_nested_facade.py`, and `tests/test_weighted_nested_representation.py` cover the maturity fixes, historical unweighted/unit parity, zero-weight invariance, integer-frequency encoder/MI/fixed-structure equivalence, arbitrary-real-weight smoke, shared-vs-legacy code layout, semantic/optimized execution, persistence, and native parity when a compiler is available.

GitHub Actions are intentionally not used for this integration.