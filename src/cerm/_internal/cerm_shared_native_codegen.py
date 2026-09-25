from __future__ import annotations

from pathlib import Path

import numpy as np

from .cerm_native_build import compile_shared_library


def _arr(values, ctype: str = "double") -> str:
    array = np.asarray(values).ravel()
    if array.size == 0:
        return "{0}"
    if ctype == "double":
        values = []
        for value in array:
            number = float(value)
            if np.isnan(number):
                values.append("NAN")
            elif np.isposinf(number):
                values.append("INFINITY")
            elif np.isneginf(number):
                values.append("-INFINITY")
            else:
                values.append(f"{number:.17g}")
        return "{" + ",".join(values) + "}"
    return "{" + ",".join(str(int(value)) for value in array) + "}"


def compile_shared_native(model, prefix: str | Path):
    """Lower a shared multiclass/multilabel finite-state model to one library."""

    from ..native_runtime import load_native_matrix_predictor

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    source = prefix.with_suffix(".cpp")
    library = prefix.with_suffix(".so")
    n_outputs = int(len(model.intercept_))
    levels = tuple(int(level) for level in model.levels)
    coarse = levels[0]
    main_levels = tuple(level for level in levels if level <= int(model.config_.max_main_level))
    pair_levels = tuple(level for level in levels if level <= min(int(model.max_bins), 8))
    fine_pairs = set(tuple(pair) for pair in model.fine_pairs_)

    declarations = [
        "#include <cmath>",
        "#include <cstdint>",
        "#include <cstddef>",
        "#include <algorithm>",
        f"static constexpr int NO={n_outputs};",
        f"static const double intercept[{n_outputs}]={_arr(model.intercept_)};",
    ]
    body = [
        'extern "C" __attribute__((visibility("default"))) void cerm_shared_predict(const double* X,int n,int d,double* out){',
        " for(int i=0;i<n;++i){",
        "  const double* row=X+(size_t)i*d;",
        "  double score[NO];",
        "  for(int o=0;o<NO;++o) score[o]=intercept[o];",
    ]

    # Selected base features only need the state resolutions referenced by the
    # fitted lookup program.  Quantize once at the maximum required resolution
    # and recover lower parents through tiny integer maps.  The lookup traversal
    # below remains in the historical order.
    required_levels = set(main_levels)
    if len(model.pairs_) > 0:
        required_levels.update(pair_levels)
    if 16 in levels and len(fine_pairs) > 0:
        required_levels.add(16)
    execution_level = int(max(required_levels)) if required_levels else int(coarse)

    state_names: dict[int, list[str]] = {level: [] for level in levels}
    raw_state_names: dict[tuple[int, int], str] = {}
    for position, raw_feature in enumerate(np.asarray(model.feature_idx_, dtype=np.int32)):
        raw_feature = int(raw_feature)
        direct = bool(model.encoder_.direct_state_mask_[raw_feature])
        direct_card = int(model.encoder_.direct_state_cardinalities_[raw_feature])
        exec_map = np.asarray(model.encoder_.maps_[execution_level][raw_feature], dtype=np.int32)
        exec_name = f"exec_{position}"
        if direct:
            exec_map_name = f"exec_map_{position}"
            declarations.append(
                f"static const int32_t {exec_map_name}[{max(1, len(exec_map))}]={_arr(exec_map, 'int')};"
            )
            raw_name = f"raw_state_{position}"
            body.extend([
                f"  long {raw_name}=std::lround(row[{raw_feature}]);",
                f"  if(!std::isfinite(row[{raw_feature}]) || {raw_name}<0 || {raw_name}>={direct_card}) {raw_name}=0;",
                f"  const int32_t {exec_name}={exec_map_name}[{raw_name}];",
            ])
        else:
            direct_thresholds = model.encoder_._direct_level_thresholds(raw_feature, execution_level)
            if direct_thresholds is not None:
                thresholds = np.asarray(direct_thresholds, dtype=np.float64)
                threshold_name = f"exec_threshold_{position}"
                declarations.append(
                    f"static const double {threshold_name}[{max(1, len(thresholds))}]={_arr(thresholds)};"
                )
                if len(thresholds):
                    expression = " + ".join(
                        f"(row[{raw_feature}]>={threshold_name}[{index}])"
                        for index in range(len(thresholds))
                    )
                    body.append(f"  const int32_t {exec_name}={expression};")
                else:
                    body.append(f"  const int32_t {exec_name}=0;")
            else:
                thresholds = np.asarray(model.encoder_.thresholds_[raw_feature], dtype=np.float64)
                threshold_name = f"threshold_{position}"
                declarations.append(
                    f"static const double {threshold_name}[{max(1, len(thresholds))}]={_arr(thresholds)};"
                )
                if len(thresholds):
                    expression = " + ".join(
                        f"(row[{raw_feature}]>={threshold_name}[{index}])"
                        for index in range(len(thresholds))
                    )
                    fine_name = f"fine_{position}"
                    body.append(f"  const int {fine_name}={expression};")
                else:
                    fine_name = f"fine_{position}"
                    body.append(f"  const int {fine_name}=0;")
                exec_map_name = f"exec_map_{position}"
                declarations.append(
                    f"static const int32_t {exec_map_name}[{max(1, len(exec_map))}]={_arr(exec_map, 'int')};"
                )
                body.append(f"  const int32_t {exec_name}={exec_map_name}[{fine_name}];")

        exec_card = int(model.encoder_.cardinalities_[execution_level][raw_feature])
        for level in sorted(required_levels):
            state_name = f"s_{level}_{position}"
            if int(level) == execution_level:
                body.append(f"  const int16_t {state_name}=(int16_t){exec_name};")
            else:
                level_map = np.asarray(model.encoder_.maps_[level][raw_feature], dtype=np.int32)
                parent = np.zeros(exec_card, dtype=np.int32)
                seen = np.zeros(exec_card, dtype=bool)
                for fine_index, child in enumerate(exec_map):
                    child = int(child)
                    value = int(level_map[fine_index])
                    if seen[child] and int(parent[child]) != value:
                        raise RuntimeError("non-nested shared quotient map")
                    parent[child] = value
                    seen[child] = True
                parent_name = f"parent_{level}_{position}"
                declarations.append(
                    f"static const int32_t {parent_name}[{max(1, len(parent))}]={_arr(parent, 'int')};"
                )
                body.append(f"  const int16_t {state_name}=(int16_t){parent_name}[{exec_name}];")
            state_names[level].append(state_name)
            raw_state_names[(level, raw_feature)] = state_name

    # Experimental residual-basis programs use the same exact state encoder but
    # may reference raw features that were not selected by the base program.
    # Emit each required quantization once so conditional blocks share common
    # subexpressions in native code just as they do in the semantic evaluator.
    extra_descriptors = tuple(getattr(model, "extra_descriptors_", ()))
    extra_lookup = tuple(getattr(model, "extra_lookup_", ()))
    if len(extra_descriptors) != len(extra_lookup):
        raise RuntimeError("residual descriptor/lookup layout mismatch")
    required_states: set[tuple[int, int]] = set()
    for descriptor in extra_descriptors:
        level = int(descriptor.level)
        required_states.add((level, int(descriptor.raw0)))
        if str(descriptor.kind) == "conditional_pair":
            required_states.add((level, int(descriptor.raw1)))
            required_states.add((level, int(descriptor.raw2)))
    for level, raw_feature in sorted(required_states):
        if (level, raw_feature) in raw_state_names:
            continue
        if level not in levels:
            raise RuntimeError(f"residual basis references unavailable level {level}")
        thresholds = np.asarray(model.encoder_.thresholds_[raw_feature], dtype=np.float64)
        direct = bool(model.encoder_.direct_state_mask_[raw_feature])
        direct_card = int(model.encoder_.direct_state_cardinalities_[raw_feature])
        token = f"extra_{level}_{raw_feature}"
        threshold_name = f"threshold_{token}"
        declarations.append(
            f"static const double {threshold_name}[{max(1, len(thresholds))}]={_arr(thresholds)};"
        )
        fine_name = f"fine_{token}"
        if direct:
            body.extend([
                f"  long {fine_name}=std::lround(row[{raw_feature}]);",
                f"  if(!std::isfinite(row[{raw_feature}]) || {fine_name}<0 || {fine_name}>={direct_card}) {fine_name}=0;",
            ])
        else:
            if len(thresholds):
                expression = " + ".join(
                    f"(row[{raw_feature}]>={threshold_name}[{index}])"
                    for index in range(len(thresholds))
                )
                body.append(f"  const int {fine_name}={expression};")
            else:
                body.append(f"  const int {fine_name}=0;")
        mapping = np.asarray(model.encoder_.maps_[level][raw_feature], dtype=np.int16)
        mapping_name = f"map_{token}"
        state_name = f"s_{token}"
        declarations.append(
            f"static const int16_t {mapping_name}[{max(1, len(mapping))}]={_arr(mapping, 'int')};"
        )
        body.append(f"  const int16_t {state_name}={mapping_name}[{fine_name}];")
        raw_state_names[(level, raw_feature)] = state_name

    lookup_index = 0
    for feature_position in range(len(model.feature_idx_)):
        body.append(f"  int code_{lookup_index}=(int){state_names[coarse][feature_position]};")
        for position, level in enumerate(main_levels):
            if position == 0:
                code_expr = state_names[level][feature_position]
            else:
                residual = np.asarray(model.main_residuals_[level][feature_position], dtype=np.int32)
                residual_name = f"main_res_{level}_{feature_position}"
                declarations.append(
                    f"static const int32_t {residual_name}[{max(1, len(residual))}]={_arr(residual, 'int')};"
                )
                code_expr = f"{residual_name}[(int){state_names[level][feature_position]}]"
            table = np.asarray(model.lookup_[lookup_index], dtype=np.float64)
            table_name = f"lookup_{lookup_index}"
            declarations.append(
                f"static const double {table_name}[{max(1, table.size)}]={_arr(table)};"
            )
            body.append(f"  int code_lookup_{lookup_index}=(int)({code_expr});")
            body.append(f"  if(code_lookup_{lookup_index}>=0 && code_lookup_{lookup_index}<{table.shape[0]}) for(int o=0;o<NO;++o) score[o]+={table_name}[(size_t)code_lookup_{lookup_index}*NO+o];")
            lookup_index += 1

    for pair in model.pairs_:
        j, k = int(pair[0]), int(pair[1])
        raw_k = int(model.feature_idx_[k])
        for level in pair_levels:
            card_k = int(model.encoder_.cardinalities_[level][raw_k])
            joint_name = f"joint_{lookup_index}"
            body.append(
                f"  int {joint_name}=(int){state_names[level][j]}*{card_k}+(int){state_names[level][k]};"
            )
            if level == coarse:
                code_expr = joint_name
            else:
                residual = np.asarray(model.pair_residuals_[level][tuple(pair)], dtype=np.int32)
                residual_name = f"pair_res_{level}_{lookup_index}"
                declarations.append(
                    f"static const int32_t {residual_name}[{max(1, len(residual))}]={_arr(residual, 'int')};"
                )
                code_expr = f"{residual_name}[{joint_name}]"
            table = np.asarray(model.lookup_[lookup_index], dtype=np.float64)
            table_name = f"lookup_{lookup_index}"
            declarations.append(
                f"static const double {table_name}[{max(1, table.size)}]={_arr(table)};"
            )
            body.append(f"  int code_lookup_{lookup_index}=(int)({code_expr});")
            body.append(f"  if(code_lookup_{lookup_index}>=0 && code_lookup_{lookup_index}<{table.shape[0]}) for(int o=0;o<NO;++o) score[o]+={table_name}[(size_t)code_lookup_{lookup_index}*NO+o];")
            lookup_index += 1
        if 16 in levels and tuple(pair) in fine_pairs:
            raw_k = int(model.feature_idx_[k])
            card_k = int(model.encoder_.cardinalities_[16][raw_k])
            joint_name = f"joint_{lookup_index}"
            body.append(
                f"  int {joint_name}=(int){state_names[16][j]}*{card_k}+(int){state_names[16][k]};"
            )
            residual = np.asarray(model.pair_residuals_[16][tuple(pair)], dtype=np.int32)
            residual_name = f"pair_res_16_{lookup_index}"
            declarations.append(
                f"static const int32_t {residual_name}[{max(1, len(residual))}]={_arr(residual, 'int')};"
            )
            table = np.asarray(model.lookup_[lookup_index], dtype=np.float64)
            table_name = f"lookup_{lookup_index}"
            declarations.append(
                f"static const double {table_name}[{max(1, table.size)}]={_arr(table)};"
            )
            body.append(f"  int code_lookup_{lookup_index}=(int){residual_name}[{joint_name}];")
            body.append(f"  if(code_lookup_{lookup_index}>=0 && code_lookup_{lookup_index}<{table.shape[0]}) for(int o=0;o<NO;++o) score[o]+={table_name}[(size_t)code_lookup_{lookup_index}*NO+o];")
            lookup_index += 1

    if lookup_index != len(model.lookup_):
        raise RuntimeError(
            f"shared lookup layout mismatch: emitted {lookup_index}, expected {len(model.lookup_)}"
        )

    for extra_index, (descriptor, raw_table) in enumerate(
        zip(extra_descriptors, extra_lookup)
    ):
        table = np.asarray(raw_table, dtype=np.float64)
        if table.ndim != 2 or table.shape[1] != n_outputs:
            raise RuntimeError("residual lookup output layout mismatch")
        level = int(descriptor.level)
        kind = str(descriptor.kind)
        code_name = f"extra_code_{extra_index}"
        if kind == "state_refinement":
            state_name = raw_state_names[(level, int(descriptor.raw0))]
            body.extend([
                f"  int {code_name}=0;",
                f"  if((int){state_name}=={int(descriptor.state)} && std::isfinite(row[{int(descriptor.raw0)}])) {code_name}=row[{int(descriptor.raw0)}] < {float(descriptor.threshold):.17g} ? 1 : 2;",
            ])
        elif kind == "class_state_delta":
            state_name = raw_state_names[(level, int(descriptor.raw0))]
            body.append(
                f"  int {code_name}=((int){state_name}=={int(descriptor.state)}) ? 1 : 0;"
            )
        elif kind == "conditional_pair":
            gate_name = raw_state_names[(level, int(descriptor.raw0))]
            left_name = raw_state_names[(level, int(descriptor.raw1))]
            right_name = raw_state_names[(level, int(descriptor.raw2))]
            right_card = int(
                model.encoder_.cardinalities_[level][int(descriptor.raw2)]
            )
            joint_values = np.asarray(descriptor.joint_values, dtype=np.int64)
            joint_name = f"extra_joint_values_{extra_index}"
            declarations.append(
                f"static const int64_t {joint_name}[{max(1, len(joint_values))}]={_arr(joint_values, 'int')};"
            )
            body.extend([
                f"  int {code_name}=0;",
                f"  if((int){gate_name}=={int(descriptor.state)}){{",
                f"   int64_t extra_joint=(int64_t){left_name}*{right_card}+(int64_t){right_name};",
                f"   const int64_t* extra_it=std::lower_bound({joint_name},{joint_name}+{len(joint_values)},extra_joint);",
                f"   if(extra_it!={joint_name}+{len(joint_values)} && *extra_it==extra_joint) {code_name}=(int)(extra_it-{joint_name})+1;",
                "  }",
            ])
        else:
            raise RuntimeError(f"unsupported residual basis kind: {kind}")
        table_name = f"extra_lookup_{extra_index}"
        declarations.append(
            f"static const double {table_name}[{max(1, table.size)}]={_arr(table)};"
        )
        body.append(
            f"  if({code_name}>=0 && {code_name}<{table.shape[0]}) for(int o=0;o<NO;++o) score[o]+={table_name}[(size_t){code_name}*NO+o];"
        )

    if model.task_type == "multiclass":
        body.extend([
            "  double mx=score[0]; for(int o=1;o<NO;++o) mx=std::max(mx,score[o]);",
            "  double total=0.0; for(int o=0;o<NO;++o){ score[o]=std::exp(std::max(-50.0,std::min(50.0,score[o]-mx))); total+=score[o]; }",
            "  for(int o=0;o<NO;++o) out[(size_t)i*NO+o]=score[o]/total;",
        ])
    else:
        constants = np.asarray(model.constant_outputs_, dtype=np.float64)
        declarations.append(f"static const double constants[{n_outputs}]={_arr(constants)};")
        body.extend([
            "  for(int o=0;o<NO;++o){",
            "   if(std::isfinite(constants[o])) out[(size_t)i*NO+o]=constants[o];",
            "   else out[(size_t)i*NO+o]=1.0/(1.0+std::exp(-std::max(-40.0,std::min(40.0,score[o]))));",
            "  }",
        ])
    body.extend([" }", "}"])
    source.write_text("\n".join(declarations + body), encoding="utf-8")
    compile_shared_library(source, library)
    predictor = load_native_matrix_predictor(
        library, n_outputs=n_outputs, symbol="cerm_shared_predict"
    )
    return predictor, source, library
