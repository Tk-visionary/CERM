from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import numpy as np

from .cerm_graph_quantization import CERMInt16GraphProgram
from .cerm_graph_optimizer import _state_index_ctype, _state_index_dtype
from .cerm_native_build import compile_shared_library


def _arr(values, ctype="double"):
    values = np.asarray(values).ravel()
    if len(values) == 0:
        return "{0}"
    if ctype == "double":
        return "{" + ",".join(f"{float(x):.17g}" for x in values) + "}"
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


class _QRegistry:
    def __init__(self, declarations: list[str]):
        self.declarations = declarations
        self.registry = {}
        self.count = 0

    def register(self, array: np.ndarray, stem="qtable") -> tuple[str, bool]:
        arr = np.asarray(array, dtype=np.int16)
        key = _key(arr)
        if key in self.registry:
            return self.registry[key]
        if arr.ndim == 2:
            trans = np.ascontiguousarray(arr.T)
            tkey = _key(trans)
            if tkey in self.registry:
                name, stored_transposed = self.registry[tkey]
                return name, not stored_transposed
            if tkey[2] < key[2]:
                stored = trans
                requested_transposed = True
            else:
                stored = np.ascontiguousarray(arr)
                requested_transposed = False
            name = f"{stem}_{self.count}"
            self.count += 1
            self.declarations.append(
                f"static const int16_t {name}[{max(1, stored.size)}]={_arr(stored, 'int')};"
            )
            self.registry[key] = (name, requested_transposed)
            self.registry[tkey] = (name, not requested_transposed)
            return name, requested_transposed
        name = f"{stem}_{self.count}"
        self.count += 1
        self.declarations.append(
            f"static const int16_t {name}[{max(1, arr.size)}]={_arr(arr, 'int')};"
        )
        self.registry[key] = (name, False)
        return name, False


def compile_graph_int16_native(program: CERMInt16GraphProgram, prefix):
    prefix = Path(prefix)
    cpp = prefix.with_suffix('.cpp')
    so = prefix.with_suffix('.so')
    nf = len(program.feature_idx)
    state_max = int(np.max(
        np.where(
            program.direct_state_mask,
            program.direct_state_cardinalities,
            np.asarray([len(t) + 1 for t in program.thresholds], dtype=np.int64),
        ),
        initial=1,
    )) - 1
    state_dtype = _state_index_dtype(state_max)
    state_ctype = _state_index_ctype(state_max)
    declarations = [
        '#include <cmath>',
        '#include <cstdint>',
        '#include <cstddef>',
        f'static constexpr int NF={nf};',
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
    declarations.append('static const double* threshold_ptr[NF]={' + ','.join(threshold_names) + '};')
    declarations.append(f"static const uint8_t threshold_len[NF]={_arr(threshold_lengths, 'int')};")

    qregistry = _QRegistry(declarations)
    unary_code = []
    for feature, table in sorted(program.unary_tables.items()):
        name, transposed = qregistry.register(table.values)
        if transposed:
            raise RuntimeError('rank-1 table transpose')
        unary_code.append(
            f"score += (double){name}[s[{feature}]]*{table.scale:.17g};"
        )

    map_initializer_registry = {}
    map_value_registry = {}
    map_compute_lines = []

    def register_map(feature: int, array: np.ndarray) -> str:
        arr = np.asarray(array, dtype=state_dtype)
        if _is_identity_map(arr):
            return f"s[{feature}]"
        key = _key(arr)
        if key not in map_initializer_registry:
            name = f"map_init_{len(map_initializer_registry)}"
            map_initializer_registry[key] = name
            declarations.append(
                f"static const {state_ctype} {name}[{len(arr)}]={_arr(arr, 'int')};"
            )
        value_key = (feature, key)
        if value_key not in map_value_registry:
            value = f"m{len(map_value_registry)}"
            map_value_registry[value_key] = value
            map_compute_lines.append(
                f"  const {state_ctype} {value}={map_initializer_registry[key]}[s[{feature}]];"
            )
        return map_value_registry[value_key]

    slot_registry = {}
    def register_slot(array: np.ndarray) -> str:
        arr = np.asarray(array, dtype=np.int16)
        key = _key(arr)
        if key not in slot_registry:
            name = f"row_slot_{len(slot_registry)}"
            slot_registry[key] = name
            declarations.append(
                f"static const int16_t {name}[{len(arr)}]={_arr(arr, 'int')};"
            )
        return slot_registry[key]

    pair_code = []
    counter = 0
    for pair in program.pair_programs:
        if pair.representation == 'dense':
            table = pair.dense_table
            name, transposed = qregistry.register(table.values)
            if not transposed:
                index = f"(int)s[{pair.left}]*{table.values.shape[1]} + (int)s[{pair.right}]"
            else:
                index = f"(int)s[{pair.right}]*{table.values.shape[0]} + (int)s[{pair.left}]"
            pair_code.append(f"score += (double){name}[{index}]*{table.scale:.17g};")
        elif pair.representation == 'components':
            for component in pair.components:
                left = register_map(pair.left, component.left_map)
                right = register_map(pair.right, component.right_map)
                table = component.table
                name, transposed = qregistry.register(table.values)
                if not transposed:
                    index = f"(int){left}*{table.values.shape[1]} + (int){right}"
                else:
                    index = f"(int){right}*{table.values.shape[0]} + (int){left}"
                pair_code.append(f"score += (double){name}[{index}]*{table.scale:.17g};")
        else:
            sparse = pair.sparse
            slot_name = register_slot(sparse.row_slot)
            table = sparse.row_table
            qname, transposed = qregistry.register(table.values)
            slot = f"slot_{counter}"
            counter += 1
            pair_code.append(f"const int16_t {slot}={slot_name}[s[{pair.left}]];")
            if not transposed:
                index = f"(int){slot}*{table.values.shape[1]} + (int)s[{pair.right}]"
            else:
                index = f"(int)s[{pair.right}]*{table.values.shape[0]} + (int){slot}"
            pair_code.append(
                f"if({slot}>=0) score += (double){qname}[{index}]*{table.scale:.17g};"
            )

    body = [
        'extern "C" __attribute__((visibility("default"))) void cerm_graph_predict(const double* X,int n,int d,double* out){',
        ' for(int i=0;i<n;++i){',
        '  const double* row=X+(size_t)i*d;',
        f'  {state_ctype} s[NF];',
        '  for(int j=0;j<NF;++j){',
        '   const double v=row[feature_idx[j]];',
        f'   if(direct_state[j]){{ long q=std::lround(v); s[j]=({state_ctype})((std::isfinite(v) && q>=0 && q<direct_card[j])?q:0); }}',
        f'   else {{ const double* t=threshold_ptr[j]; int f=0; while(f<threshold_len[j] && v>=t[f]) ++f; s[j]=({state_ctype})f; }}',
        '  }',
    ]
    body.extend(map_compute_lines)
    body.append(f"  double score={program.intercept:.17g};")
    body.extend('  ' + line for line in unary_code)
    body.extend('  ' + line for line in pair_code)
    body.extend([
        '  out[i]=1.0/(1.0+std::exp(-score));',
        ' }',
        '}',
    ])
    cpp.write_text('\n'.join(declarations + body), encoding='utf-8')
    compile_shared_library(cpp, so)

    library = ctypes.CDLL(str(so))
    fn = library.cerm_graph_predict
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)
    ]
    fn.restype = None

    def predict(X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        fn(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(X), X.shape[1],
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return out

    return predict, cpp, so
