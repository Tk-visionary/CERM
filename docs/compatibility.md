# Compatibility policy

CERM 1.0.0a1 targets Python 3.10–3.13 and scikit-learn 1.6–1.x. The public CI matrix is the authoritative tested range for release revisions.

Public scikit-learn APIs are preferred. The optional optimized binary
LIBLINEAR operator accesses a private extension through `cerm._compat`; an ABI
failure must fall back to public `LogisticRegression` rather than preventing
fit.

Serialized `joblib` objects are not guaranteed to load across CERM minor
versions. Versioned directory exports include manifests and checksums, but they
remain trusted-code artifacts rather than a secure interchange format.

The semantic Python backend is the reference implementation. Optimized and
native backends must remain numerically equivalent within their documented
error bounds.

Version 0.10.0a2 preserves fixed shared OVR and fixed shared multinomial
probabilities exactly relative to 0.10.0a1 on the frozen 38-task external
suite. The `auto` objective intentionally changes on five low-validation-margin
tasks by falling back to an OVR candidate. The public default remains unchanged.

Version 0.11.0a1 keeps `representation_strategy="baseline"` as the default and
retains shared model IR v2/program v1 for that path. Adaptive shared models use
model IR v3/program v2. The 0.11 loader accepts both generations; compiled
native libraries remain platform-specific and joblib compatibility is not
promised across minor versions.

## 0.6.1 private-module migration

The private implementation package changed from `cerm._vendor` to
`cerm._internal` in 0.7.0a1. Joblib artifacts that serialize classes from the
old private path may not load. This is an intentional alpha-stage private API
break. Refit from source data or regenerate exports with the new version.


## Task-program compatibility

Binary semantic and compiled program formats remain unchanged from 0.7.0a1.
Version 0.8.0a1 added `cerm-program-bundle-v1` and regression package v3.
Version 0.8.1a1 adds `cerm-compiled-program-v2` and
`cerm-compiled-bundle-v2` for pickle-free common-schema native packages,
`cerm-program-bundle-v2` for deduplicated shared adapters,
`cerm-finite-state-regression-v2` with fitted pair cardinalities and direct-state
metadata, `cerm-regression-adapter-ir-v1`, and the architecture-specific,
pickle-free `cerm-compiled-regression-v2` format. The loader retains read
compatibility with experimental compiled regression v1. `PortableRegressionProgram` can execute
checked regression JSON/NPZ exports without joblib. These alpha formats may
change before 1.0.

Compiled binary program v1 and compiled regression v1 remain loadable for
alpha-stage backward compatibility. Embedding adapters and unusual custom
Python category objects may still use the v1 joblib fallback; manifests mark
that fallback with `unsafe_pickle_state`.
