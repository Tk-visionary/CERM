from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import numpy as np

from .cerm_graph_optimizer import _state_index_ctype, _state_index_dtype
from .cerm_graph_optimizer_v2 import CERMGraphProgramV2
from .cerm_native_build import compile_shared_library


def _arr(values, ctype="double"):
    values = np.asarray(values).ravel()
    if len(values) == 0:
        return "{0}"
    if ctype == "double":
        return "{" + ",".join(f"{float(x):.17g}" for x in values) + "}"
    if ctype == "float":
        literals = []
        for x in values:
            text = f"{float(x):.9g}"
            if "e" not in text.lower() and "." not in text:
                text += ".0"
            literals.append(text + "f")
        return "{" + ",".join(literals) + "}"
    if ctype == "int16_t":
        return "{" + ",".join(str(int(x)) for x in values) + "}"
    return "{" + ",".join(str(int(x)) for x in values) + "}"


def _key(array: np.ndarray):
    array = np.ascontiguousarray(array)
    return (
        array.dtype.str,
        tuple(array.shape),
        hashlib.sha256(array.view(np.uint8)).hexdigest(),
    )


def _is_identity_map(array: np.ndarray) -> bool:
    array = np.asarray(array, dtype=np.int64)
    return np.array_equal(array, np.arange(len(array), dtype=np.int64))


class _TableRegistry:
    """Initializer CSE with transpose canonicalization for rank-2 tables."""

    def __init__(self, declarations: list[str], ctype: str):
        self.declarations = declarations
        self.ctype = ctype
        self.entries: dict[tuple, tuple[str, bool, tuple[int, ...]]] = {}
        self.count = 0

    def register(self, array: np.ndarray, stem: str = "table") -> tuple[str, bool]:
        arr = np.asarray(array)
        direct_key = (self.ctype, _key(arr))
        if direct_key in self.entries:
            name, transposed, _ = self.entries[direct_key]
            return name, transposed

        if arr.ndim == 2:
            transposed_arr = np.ascontiguousarray(arr.T)
            transpose_key = (self.ctype, _key(transposed_arr))
            if transpose_key in self.entries:
                name, stored_transposed, _ = self.entries[transpose_key]
                # The existing initializer represents arr.T relative to this request.
                return name, not stored_transposed

            # Store the lexicographically smaller binary representation. Register both
            # orientations so future transpose-equivalent tables share one initializer.
            direct_digest = direct_key[1][2]
            transpose_digest = transpose_key[1][2]
            if transpose_digest < direct_digest:
                stored = transposed_arr
                requested_transposed = True
            else:
                stored = np.ascontiguousarray(arr)
                requested_transposed = False
            name = f"{stem}_{self.count}"
            self.count += 1
            self.declarations.append(
                f"static const {self.ctype} {name}[{max(1, stored.size)}]="
                f"{_arr(stored, self.ctype)};"
            )
            stored_shape = tuple(stored.shape)
            self.entries[direct_key] = (name, requested_transposed, stored_shape)
            self.entries[transpose_key] = (name, not requested_transposed, stored_shape)
            return name, requested_transposed

        name = f"{stem}_{self.count}"
        self.count += 1
        stored = np.ascontiguousarray(arr)
        self.declarations.append(
            f"static const {self.ctype} {name}[{max(1, stored.size)}]="
            f"{_arr(stored, self.ctype)};"
        )
        self.entries[direct_key] = (name, False, tuple(stored.shape))
        return name, False


def compile_graph_native_v2(program: CERMGraphProgramV2, prefix, *, projected_input: bool = False):
    prefix = Path(prefix)
    cpp = prefix.with_suffix(".cpp")
    so = prefix.with_suffix(".so")
    feature_count = len(program.feature_idx)
    table_ctype = "float" if program.table_dtype == "float32" else "double"
    state_max = int(np.max(program.fine_cardinalities, initial=1)) - 1
    state_dtype = _state_index_dtype(state_max)
    state_ctype = _state_index_ctype(state_max)

    declarations = [
        "#include <cmath>",
        "#include <cstdint>",
        "#include <cstddef>",
        f"static constexpr int NF={feature_count};",
        f"static const int32_t feature_idx[NF]={_arr(program.feature_idx, 'int')};",
        f"static const uint8_t direct_state[NF]={_arr(program.direct_state_mask.astype(np.uint8), 'int')};",
        f"static const uint64_t direct_card[NF]={_arr(program.direct_state_cardinalities, 'int')};",
    ]

    threshold_registry = {}
    threshold_names = []
    threshold_lengths = []
    for threshold in program.thresholds:
        threshold = np.asarray(threshold, dtype=np.float64)
        key = _key(threshold)
        if key not in threshold_registry:
            name = f"threshold_{len(threshold_registry)}"
            threshold_registry[key] = name
            declarations.append(
                f"static const double {name}[{max(1, len(threshold))}]={_arr(threshold)};"
            )
        threshold_names.append(threshold_registry[key])
        threshold_lengths.append(len(threshold))
    declarations.append(
        "static const double* threshold_ptr[NF]={" + ",".join(threshold_names) + "};"
    )
    declarations.append(
        f"static const uint8_t threshold_len[NF]={_arr(threshold_lengths, 'int')};"
    )

    tables = _TableRegistry(declarations, table_ctype)
    unary_names: dict[int, str] = {}
    for feature, table in sorted(program.unary_tables.items()):
        name, transposed = tables.register(np.asarray(table), "table")
        if transposed:
            raise RuntimeError("rank-1 unary table cannot be transposed")
        unary_names[feature] = name

    map_initializer_registry = {}
    map_value_registry = {}
    map_compute_lines = []

    def register_map(feature: int, array: np.ndarray) -> str:
        array = np.asarray(array, dtype=state_dtype)
        if _is_identity_map(array):
            return f"s[{feature}]"
        init_key = _key(array)
        if init_key not in map_initializer_registry:
            init_name = f"map_init_{len(map_initializer_registry)}"
            map_initializer_registry[init_key] = init_name
            declarations.append(
                f"static const {state_ctype} {init_name}[{len(array)}]={_arr(array, 'int')};"
            )
        value_key = (feature, init_key)
        if value_key not in map_value_registry:
            value_name = f"m{len(map_value_registry)}"
            map_value_registry[value_key] = value_name
            map_compute_lines.append(
                f"  const {state_ctype} {value_name}={map_initializer_registry[init_key]}[s[{feature}]];"
            )
        return map_value_registry[value_key]

    pair_code = []
    sparse_slot_registry = {}

    def register_sparse_slots(array: np.ndarray) -> str:
        arr = np.asarray(array, dtype=np.int16)
        key = _key(arr)
        if key not in sparse_slot_registry:
            name = f"row_slot_{len(sparse_slot_registry)}"
            sparse_slot_registry[key] = name
            declarations.append(
                f"static const int16_t {name}[{len(arr)}]={_arr(arr, 'int16_t')};"
            )
        return sparse_slot_registry[key]

    for pair in program.pair_programs:
        if pair.representation == "dense":
            table = np.asarray(pair.dense_table)
            name, transposed = tables.register(table, "table")
            if not transposed:
                pair_code.append(
                    f"score += {name}[(int)s[{pair.left}]*{table.shape[1]} + (int)s[{pair.right}]];"
                )
            else:
                pair_code.append(
                    f"score += {name}[(int)s[{pair.right}]*{table.shape[0]} + (int)s[{pair.left}]];"
                )
        elif pair.representation == "components":
            for component in pair.components:
                left_value = register_map(pair.left, component.left_map)
                right_value = register_map(pair.right, component.right_map)
                table = np.asarray(component.table)
                name, transposed = tables.register(table, "table")
                if not transposed:
                    pair_code.append(
                        f"score += {name}[(int){left_value}*{table.shape[1]} + (int){right_value}];"
                    )
                else:
                    pair_code.append(
                        f"score += {name}[(int){right_value}*{table.shape[0]} + (int){left_value}];"
                    )
        else:
            sparse = pair.sparse
            assert sparse is not None
            slot_name = register_sparse_slots(sparse.row_slot)
            table = np.asarray(sparse.row_table)
            table_name, transposed = tables.register(table, "table")
            slot_var = f"slot_{len(pair_code)}"
            pair_code.append(f"const int16_t {slot_var}={slot_name}[s[{pair.left}]];")
            if not transposed:
                pair_code.append(
                    f"if({slot_var}>=0) score += {table_name}[(int){slot_var}*{table.shape[1]} + (int)s[{pair.right}]];"
                )
            else:
                # Stored table is transposed: [right_card, active_rows].
                pair_code.append(
                    f"if({slot_var}>=0) score += {table_name}[(int)s[{pair.right}]*{table.shape[0]} + (int){slot_var}];"
                )

    body = [
        'extern "C" __attribute__((visibility("default"))) void cerm_graph_predict(const double* X,int n,int d,double* out){',
        " for(int i=0;i<n;++i){",
        "  const double* row=X+(size_t)i*d;",
        f"  {state_ctype} s[NF];",
    ]
    # Specialize state generation to the fitted model.  This removes the
    # generic feature loop, indirect threshold pointers, and data-dependent
    # threshold while-loop while preserving the exact finite-state code.
    for feature in range(feature_count):
        raw_feature = int(program.feature_idx[feature])
        input_feature = int(feature) if projected_input else raw_feature
        value_name = f"v_{feature}"
        body.append(f"  const double {value_name}=row[{input_feature}];")
        if bool(program.direct_state_mask[feature]):
            card = int(program.direct_state_cardinalities[feature])
            body.append(f"  const long q_{feature}=std::lround({value_name});")
            body.append(
                f"  s[{feature}]=({state_ctype})((std::isfinite({value_name}) && "
                f"q_{feature}>=0 && q_{feature}<{card})?q_{feature}:0);"
            )
        else:
            threshold_name = threshold_names[feature]
            threshold_count = int(threshold_lengths[feature])
            if threshold_count:
                expression = " + ".join(
                    f"({value_name}>={threshold_name}[{index}])"
                    for index in range(threshold_count)
                )
                body.append(f"  s[{feature}]=({state_ctype})({expression});")
            else:
                body.append(f"  s[{feature}]=({state_ctype})0;")
    body.extend(map_compute_lines)
    body.append(f"  double score={program.intercept:.17g};")
    for feature, name in sorted(unary_names.items()):
        body.append(f"  score += {name}[s[{feature}]];")
    body.extend("  " + line for line in pair_code)
    body.extend(
        [
            "  out[i]=1.0/(1.0+std::exp(-score));",
            " }",
            "}",
        ]
    )
    cpp.write_text("\n".join(declarations + body), encoding="utf-8")
    compile_shared_library(cpp, so)

    library = ctypes.CDLL(str(so))
    function = library.cerm_graph_predict
    function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = None

    def predict(X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        out = np.empty(len(X), dtype=np.float64)
        function(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(X),
            X.shape[1],
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out

    return predict, cpp, so
