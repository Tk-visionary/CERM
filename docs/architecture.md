# Architecture

CERM separates four concerns.

1. **Typed adaptation** converts numerical, missing, nominal, and embedding
   inputs into continuous columns and finite states.
2. **Semantic learning** fits a hierarchical quotient model with main, pair,
   and conditional block operators.
3. **Program optimization** applies exact table composition, factorization,
   CSE, dead-term removal, and dense/mapped/sparse representation selection.
4. **Backend lowering** produces optimized NumPy evaluation or specialized
   native code.

`CERMClassifier.predict_proba` uses the exact optimized graph by default. The
semantic program is retained as `program_`, so compiler choices do not overwrite
or redefine the learned statistical model.

Approximate dtypes are separate deployment decisions:

- float64: exact reference lowering;
- float32: calibrated approximate lowering;
- int16: per-table symmetric quantization with an exported probability-error
  bound.

Finite-state labels are stored with the smallest lossless unsigned width
(`uint8`, `uint16`, or `uint32`) supported by the fitted cardinality. Native
direct-state bounds use `uint64_t`; this keeps ordinary binary graphs compact
without wrapping high-cardinality categorical states.

## Calibration and training statistics

An accepted calibrator is compiled into the semantic coefficients before graph
optimization. Therefore calibration does not add a runtime operator and exact
semantic/optimized/native equivalence is preserved.

The optional training cache stores the selected finite-state matrix and reusable
gradient/Hessian histograms. It is a training-side artifact and is deliberately
excluded from semantic export and deployment packages.

## Compiled artifact lifecycle

`CompiledProgram.save()` packages the native library, portable adapter IR,
input schema, class labels, backend metadata, and checksums. Common numeric and
categorical schemas use pickle-free compiled program v2. `CompiledProgram.load()`
verifies the package and rebinds the stable C ABI symbol without recompilation.
Compiled bundle v2 stores a compatible shared adapter once for all heads.
Embedding adapters or unusual Python category objects can fall back to the
legacy trusted-joblib v1 format. Compiled packages remain guarded by
operating-system and machine-architecture metadata.

## Scaling parameter lowering

The public scaling controls are lowered before semantic training:

```text
raw/adapted features
    -> deterministic colsample view
    -> dynamic 4/8/16 state hierarchy from max_bins
    -> stratified selection-row view from subsample
    -> candidate selection
    -> full-row fixed-architecture refit
    -> semantic/optimized/native programs with the same feature view
```

`n_jobs` affects only independent fold or neighbor work. It does not alter the
order of histogram accumulation inside one model, preserving deterministic
serial/parallel results in tested paths.

Portable categorical mappings use a compact typed parallel-array representation for homogeneous string, integer, Boolean, or floating-point categories. Existing row-wise mapping artifacts remain readable. This keeps the pickle-free runtime format competitive with compressed Python serialization without changing model semantics.

## Shared multi-output representation

The experimental shared path separates a target-neutral finite-state encoder
from vector-valued output heads. Candidate structures are views over maximal
encoded column banks. Head fitting is embarrassingly parallel, while validation
scoring is fused into one sparse matrix multiplication. Portable IR stores flat
arrays plus offsets, and native lowering emits one multi-output C++ library.
Default OVR/independent modes remain available when output-specific structure is
more important than shared execution efficiency.

The opt-in adaptive shared-multiclass path adds a bounded residual basis after
the base structure is fixed. Each `class_state_delta` node references one
existing encoder level, raw feature, state, and target class; its lookup is
masked to that class. The semantic evaluator and native compiler reuse the same
quantized state value, so no secondary encoder or Python callback enters the
runtime. Shared finite-state IR v3 stores typed descriptor arrays and packed
lookup offsets with strict range, shape, finiteness, and class-mask checks.

## Generalized objective backend

Generalized regression separates representation selection from the loss head:

```text
raw/typed input
    -> one CERM finite-state representation
    -> one shared sparse design
    -> ObjectiveBackend target transform and solver heads
    -> compact lookup/linear coefficients
```

The internal `ObjectiveBackend` protocol validates targets, constructs the
representation target, performs exact offset/exposure transformations, creates
loss solvers, and applies the inverse link. Multi-quantile heads share the same
representation and design; only the quantile solver calls are separate. The
prediction loop preserves the single-head arithmetic order for bitwise equality
with separately fitted quantile heads before optional non-crossing correction.

Shared multi-output regression uses a different target-reduction contract: a
weighted PC1 of standardized outputs selects the finite-state representation,
then one multi-target Ridge head is fitted. Independent output-specific CERM
models remain the default because a single PC1 representation is not universally
sufficient.
