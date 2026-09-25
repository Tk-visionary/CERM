# Third-party notices

CERM includes a small adapted implementation of the TRON optimization routine
and L2-regularized binary logistic objective used by LIBLINEAR and the
scikit-learn vendored LIBLINEAR integration. CERM's modification reads SciPy
CSR storage directly and receives the same SciPy Cython BLAS callbacks used by
scikit-learn, avoiding the intermediate LIBLINEAR `feature_node` expansion.

The adapted code is contained in
`src/cerm/_internal/native/cerm_training_core.cpp`. It is used only for the
package's exact sparse binary L2R_LR training fast path. The historical
scikit-learn low-level LIBLINEAR path remains the runtime fallback.

The relevant BSD-3-Clause notices are distributed with source and binary
artifacts in:

- `LICENSES/LIBLINEAR-BSD-3-Clause.txt`
- `LICENSES/SCIKIT-LEARN-BSD-3-Clause.txt`

LIBLINEAR project provenance: https://github.com/cjlin1/liblinear

scikit-learn provenance: https://github.com/scikit-learn/scikit-learn

All other runtime and development dependencies are installed separately and
remain licensed by their respective copyright holders. The dependency list is
defined in `pyproject.toml`.
