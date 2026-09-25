from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from .cerm_graph_native_codegen_v2 import _TableRegistry, _arr, _is_identity_map, _key
from .cerm_graph_optimizer import _state_index_ctype, _state_index_dtype
from .cerm_graph_optimizer_v2 import CERMGraphProgramV2
from .cerm_native_build import compile_shared_library
from .cerm_typed_quotient_adapters_v4 import EmbeddingPrototypeAdapter


def compile_embedding_graph_fused(
    adapter: EmbeddingPrototypeAdapter,
    program: CERMGraphProgramV2,
    prefix,
):
    """Fuse a prototype embedding adapter and finite-state graph into one kernel."""

    if not isinstance(adapter, EmbeddingPrototypeAdapter):
        raise TypeError("only EmbeddingPrototypeAdapter is supported")
    prefix = Path(prefix)
    cpp = prefix.with_suffix('.cpp')
    so = prefix.with_suffix('.so')

    mean = np.asarray(adapter.mean_, dtype=np.float64)
    scale = np.asarray(adapter.scale_, dtype=np.float64)
    pca_mean = np.asarray(adapter.pca_.mean_, dtype=np.float64)
    pca = np.asarray(adapter.pca_.components_, dtype=np.float64)
    lda_coef = np.asarray(adapter.lda_.coef_[0], dtype=np.float64)
    lda_intercept = float(adapter.lda_.intercept_[0])
    proto0 = np.asarray(adapter.prototypes_[0], dtype=np.float64)
    proto1 = np.asarray(adapter.prototypes_[1], dtype=np.float64)
    adapter_thresholds = [np.asarray(t, dtype=np.float64) for t in adapter.thresholds_]

    d = len(mean)
    npc = pca.shape[0]
    nad = len(adapter_thresholds)
    if nad != npc + 5:
        raise ValueError("unexpected prototype adapter output layout")
    max_adapter_thresholds = max(1, max(len(t) for t in adapter_thresholds))
    adapter_nth = np.asarray([len(t) for t in adapter_thresholds], dtype=np.int32)
    adapter_state_max = max((len(t) for t in adapter_thresholds), default=0)
    adapter_state_dtype = _state_index_dtype(adapter_state_max)
    adapter_state_ctype = _state_index_ctype(adapter_state_max)
    graph_state_max = int(np.max(program.fine_cardinalities, initial=1)) - 1
    graph_state_dtype = _state_index_dtype(graph_state_max)
    graph_state_ctype = _state_index_ctype(graph_state_max)
    adapter_th = np.zeros((nad, max_adapter_thresholds), dtype=np.float64)
    for j, th in enumerate(adapter_thresholds):
        adapter_th[j, : len(th)] = th

    nf = len(program.feature_idx)
    table_ctype = 'float' if program.table_dtype == 'float32' else 'double'
    declarations = [
        '#include <cmath>', '#include <cstdint>', '#include <cstddef>',
        f'static constexpr int D={d}, NPC={npc}, NAD={nad}, NF={nf};',
        f'static constexpr int N0={len(proto0)}, N1={len(proto1)}, AMAX={max_adapter_thresholds};',
        f'static const double emb_mean[D]={_arr(mean)};',
        f'static const double emb_scale[D]={_arr(scale)};',
        f'static const double pca_mean[D]={_arr(pca_mean)};',
        'static const double pca[NPC][D]={' + ','.join(_arr(r) for r in pca) + '};',
        f'static const double lda_coef[D]={_arr(lda_coef)};',
        f'static const double lda_intercept={lda_intercept:.17g};',
        'static const double proto0[N0][D]={' + ','.join(_arr(r) for r in proto0) + '};',
        'static const double proto1[N1][D]={' + ','.join(_arr(r) for r in proto1) + '};',
        f'static const int32_t adapter_nth[NAD]={_arr(adapter_nth, "int")};',
        'static const double adapter_th[NAD][AMAX]={' + ','.join(_arr(r) for r in adapter_th) + '};',
        f'static const double soft_scale={float(adapter.soft_scale_):.17g};',
        f'static const int32_t graph_feature_idx[NF]={_arr(program.feature_idx, "int")};',
    ]

    # Compose adapter-state quantization with the core quantizer. The adapter
    # already emits integer states, so the second threshold chain can be
    # replaced by one exact state map per retained graph feature.
    graph_map_registry = {}
    graph_map_names = []
    for local_j, adapted_j in enumerate(program.feature_idx):
        adapted_j = int(adapted_j)
        cardinality = len(adapter_thresholds[adapted_j]) + 1
        state_values = np.arange(cardinality, dtype=np.float64)
        if program.direct_state_mask[local_j]:
            core_card = int(program.direct_state_cardinalities[local_j])
            mapping = np.where(state_values < core_card, state_values, 0).astype(graph_state_dtype)
        else:
            mapping = np.searchsorted(
                np.asarray(program.thresholds[local_j], dtype=np.float64),
                state_values,
                side="right",
            ).astype(graph_state_dtype)
        key = _key(mapping)
        if key not in graph_map_registry:
            name = f"adapter_to_graph_{len(graph_map_registry)}"
            graph_map_registry[key] = name
            declarations.append(
                f"static const {graph_state_ctype} {name}[{len(mapping)}]={_arr(mapping, 'int')};"
            )
        graph_map_names.append(graph_map_registry[key])
    declarations.append(
        f"static const {graph_state_ctype}* adapter_to_graph_ptr[NF]={{"
        + ",".join(graph_map_names)
        + "};"
    )

    tables = _TableRegistry(declarations, table_ctype)
    unary_code = []
    for feature, table in sorted(program.unary_tables.items()):
        name, transposed = tables.register(np.asarray(table), 'table')
        if transposed:
            raise RuntimeError('rank-1 transpose')
        unary_code.append(f'score += {name}[s[{feature}]];')

    map_initializer_registry = {}
    map_value_registry = {}
    map_compute_lines = []

    def register_map(feature: int, array: np.ndarray) -> str:
        arr = np.asarray(array, dtype=graph_state_dtype)
        if _is_identity_map(arr):
            return f's[{feature}]'
        key = _key(arr)
        if key not in map_initializer_registry:
            name = f'map_init_{len(map_initializer_registry)}'
            map_initializer_registry[key] = name
            declarations.append(f'static const {graph_state_ctype} {name}[{len(arr)}]={_arr(arr,"int")};')
        value_key = (feature, key)
        if value_key not in map_value_registry:
            value = f'm{len(map_value_registry)}'
            map_value_registry[value_key] = value
            map_compute_lines.append(f'  const {graph_state_ctype} {value}={map_initializer_registry[key]}[s[{feature}]];')
        return map_value_registry[value_key]

    slot_registry = {}
    def register_slot(array):
        arr = np.asarray(array, dtype=np.int16)
        key = _key(arr)
        if key not in slot_registry:
            name = f'row_slot_{len(slot_registry)}'
            slot_registry[key] = name
            declarations.append(f'static const int16_t {name}[{len(arr)}]={_arr(arr,"int")};')
        return slot_registry[key]

    pair_code = []
    slot_counter = 0
    for pair in program.pair_programs:
        if pair.representation == 'dense':
            table = np.asarray(pair.dense_table)
            name, transposed = tables.register(table, 'table')
            if not transposed:
                idx = f'(int)s[{pair.left}]*{table.shape[1]} + (int)s[{pair.right}]'
            else:
                idx = f'(int)s[{pair.right}]*{table.shape[0]} + (int)s[{pair.left}]'
            pair_code.append(f'score += {name}[{idx}];')
        elif pair.representation == 'components':
            for component in pair.components:
                left = register_map(pair.left, component.left_map)
                right = register_map(pair.right, component.right_map)
                table = np.asarray(component.table)
                name, transposed = tables.register(table, 'table')
                if not transposed:
                    idx = f'(int){left}*{table.shape[1]} + (int){right}'
                else:
                    idx = f'(int){right}*{table.shape[0]} + (int){left}'
                pair_code.append(f'score += {name}[{idx}];')
        else:
            sparse = pair.sparse
            slot_name = register_slot(sparse.row_slot)
            table = np.asarray(sparse.row_table)
            name, transposed = tables.register(table, 'table')
            slot = f'slot_{slot_counter}'; slot_counter += 1
            pair_code.append(f'const int16_t {slot}={slot_name}[s[{pair.left}]];')
            if not transposed:
                idx = f'(int){slot}*{table.shape[1]} + (int)s[{pair.right}]'
            else:
                idx = f'(int)s[{pair.right}]*{table.shape[0]} + (int){slot}'
            pair_code.append(f'if({slot}>=0) score += {name}[{idx}];')

    body = [
        'extern "C" __attribute__((visibility("default"))) void cerm_embedding_graph_predict(const double* X,int n,int input_d,double* out){',
        ' for(int i=0;i<n;++i){',
        '  const double* row=X+(size_t)i*input_d;',
        '  double z[D];',
        '  for(int j=0;j<D;++j){ double v=row[j]; if(!std::isfinite(v)) v=emb_mean[j]; z[j]=(v-emb_mean[j])/emb_scale[j]; }',
        '  double feat[NAD];',
        '  for(int q=0;q<NPC;++q){ double value=0.0; for(int j=0;j<D;++j) value+=(z[j]-pca_mean[j])*pca[q][j]; feat[q]=value; }',
        '  double lda=lda_intercept; for(int j=0;j<D;++j) lda+=z[j]*lda_coef[j]; feat[NPC]=lda;',
        '  double d0=1.0e300,d1=1.0e300;',
        '  for(int q=0;q<N0;++q){ double value=0.0; for(int j=0;j<D;++j){ double e=z[j]-proto0[q][j]; value+=e*e; } if(value<d0)d0=value; }',
        '  for(int q=0;q<N1;++q){ double value=0.0; for(int j=0;j<D;++j){ double e=z[j]-proto1[q][j]; value+=e*e; } if(value<d1)d1=value; }',
        '  feat[NPC+1]=d0; feat[NPC+2]=d1; feat[NPC+3]=d0-d1;',
        '  double a=(d1-d0)/soft_scale; if(a>35.0)a=35.0; if(a<-35.0)a=-35.0; feat[NPC+4]=1.0/(1.0+std::exp(a));',
        f'  {adapter_state_ctype} adapter_state[NAD];',
        f'  for(int q=0;q<NAD;++q){{ int state=0; while(state<adapter_nth[q] && feat[q]>=adapter_th[q][state]) ++state; adapter_state[q]=({adapter_state_ctype})state; }}',
        f'  {graph_state_ctype} s[NF];',
        '  for(int j=0;j<NF;++j) s[j]=adapter_to_graph_ptr[j][adapter_state[graph_feature_idx[j]]];',
    ]
    body.extend(map_compute_lines)
    body.append(f'  double score={program.intercept:.17g};')
    body.extend('  ' + line for line in unary_code)
    body.extend('  ' + line for line in pair_code)
    body.extend([
        '  out[i]=1.0/(1.0+std::exp(-score));',
        ' }',
        '}',
    ])

    cpp.write_text('\n'.join(declarations + body), encoding='utf-8')
    compile_shared_library(cpp, so)
    lib = ctypes.CDLL(str(so))
    fn = lib.cerm_embedding_graph_predict
    fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    fn.restype = None

    def predict(X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        fn(X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(X), X.shape[1], out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out

    return predict, cpp, so
