from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import numpy as np

from .cerm_graph_optimizer import (
    CERMGraphProgram,
    _state_index_ctype,
    _state_index_dtype,
)
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
    return "{" + ",".join(str(int(x)) for x in values) + "}"


def _key(array: np.ndarray):
    array = np.ascontiguousarray(array)
    return (array.dtype.str, array.shape, hashlib.sha256(array.view(np.uint8)).hexdigest())


def _is_identity_map(array: np.ndarray) -> bool:
    array = np.asarray(array, dtype=np.int64)
    return np.array_equal(array, np.arange(len(array), dtype=np.int64))


def compile_graph_native(program: CERMGraphProgram, prefix):
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

    # Threshold initializer CSE. Binary/one-hot features often share {0.5}.
    threshold_registry = {}
    threshold_names = []
    threshold_lengths = []
    for threshold in program.thresholds:
        key = _key(np.asarray(threshold, dtype=np.float64))
        if key not in threshold_registry:
            name = f"threshold_{len(threshold_registry)}"
            threshold_registry[key] = name
            declarations.append(
                f"static const double {name}[{max(1, len(threshold))}]={_arr(threshold)};"
            )
        threshold_names.append(threshold_registry[key])
        threshold_lengths.append(len(threshold))
    declarations.append(
        "static const double* threshold_ptr[NF]={"
        + ",".join(threshold_names)
        + "};"
    )
    declarations.append(
        f"static const uint8_t threshold_len[NF]={_arr(threshold_lengths, 'int')};"
    )

    # All constant tables share one initializer registry.
    table_registry = {}

    def register_table(array: np.ndarray, stem: str) -> str:
        array = np.asarray(array)
        key = (table_ctype, _key(array))
        if key not in table_registry:
            name = f"{stem}_{len(table_registry)}"
            table_registry[key] = name
            declarations.append(
                f"static const {table_ctype} {name}[{max(1, array.size)}]="
                f"{_arr(array, table_ctype)};"
            )
        return table_registry[key]

    unary_names = {}
    for feature, table in sorted(program.unary_tables.items()):
        unary_names[feature] = register_table(table, "table")

    # Map CSE is keyed by (input state, map), not only by the initializer.
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
    for pair in program.pair_programs:
        if pair.representation == "dense":
            table = np.asarray(pair.dense_table)
            name = register_table(table, "table")
            pair_code.append(
                f"score += {name}[(int)s[{pair.left}]*{table.shape[1]} + (int)s[{pair.right}]];"
            )
        else:
            for component in pair.components:
                left_value = register_map(pair.left, component.left_map)
                right_value = register_map(pair.right, component.right_map)
                table = register_table(component.table, "table")
                pair_code.append(
                    f"score += {table}[(int){left_value}*{component.table.shape[1]} + (int){right_value}];"
                )

    body = [
        'extern "C" __attribute__((visibility("default"))) void cerm_graph_predict(const double* X,int n,int d,double* out){',
        " for(int i=0;i<n;++i){",
        "  const double* row=X+(size_t)i*d;",
        f"  {state_ctype} s[NF];",
    ]
    for feature in range(feature_count):
        raw_feature = int(program.feature_idx[feature])
        value_name = f"v_{feature}"
        body.append(f"  const double {value_name}=row[{raw_feature}];")
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
        out = np.empty(len(X), dtype=np.float64)
        function(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(X),
            X.shape[1],
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out

    return predict, cpp, so
