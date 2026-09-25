from __future__ import annotations

from pathlib import Path

import numpy as np

from .cerm_native_build import compile_shared_library


def _arr(values, ctype: str = "double") -> str:
    array = np.asarray(values).ravel()
    if array.size == 0:
        return "{0}"
    if ctype == "double":
        return "{" + ",".join(f"{float(value):.17g}" for value in array) + "}"
    if ctype == "int16_t":
        return "{" + ",".join(str(int(value)) for value in array) + "}"
    return "{" + ",".join(str(int(value)) for value in array) + "}"


def compile_regression_native(model, prefix: str | Path, *, projected_input: bool = False):
    """Compile a fitted ``FiniteStateRidgeRegressor`` into a scalar C ABI.

    The generated function consumes the already-adapted numeric matrix.  The
    Python runtime remains responsible for DataFrame schema validation and any
    categorical/missing-value adapter, exactly as for binary compiled programs.
    """

    from ..native_runtime import load_native_predictor

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    source = prefix.with_suffix(".cpp")
    library = prefix.with_suffix(".so")

    feature_idx = np.asarray(model.feature_idx_, dtype=np.int32)
    n_features = int(len(feature_idx))
    levels = tuple(int(level) for level in model.levels)
    active_levels = tuple(
        level for level in levels if level <= int(model.config_.max_main_level)
    )
    pair_level = (
        max(level for level in active_levels if level <= min(int(model.config_.max_main_level), 8))
        if active_levels
        else None
    )

    declarations = [
        "#include <cmath>",
        "#include <cstdint>",
        "#include <cstddef>",
        f"static constexpr int NF={n_features};",
        f"static const int32_t feature_idx[{max(1, n_features)}]={_arr(feature_idx, 'int')};",
    ]
    body = [
        'extern "C" __attribute__((visibility("default"))) void cerm_regression_predict(const double* X,int n,int d,double* out){',
        " for(int i=0;i<n;++i){",
        "  const double* row=X+(size_t)i*d;",
    ]

    state_names: dict[int, list[str]] = {level: [] for level in active_levels}
    if active_levels:
        model._ensure_execution_maps()
        execution_level = int(model.execution_level_)
    else:
        execution_level = 0

    for position, raw_feature in enumerate(feature_idx) if active_levels else ():
        raw_feature = int(raw_feature)
        input_feature = int(position) if projected_input else raw_feature
        direct = bool(model.encoder_.direct_state_mask_[raw_feature])
        direct_cardinality = int(model.encoder_.direct_state_cardinalities_[raw_feature])
        execution_map = np.asarray(
            model.encoder_.maps_[execution_level][raw_feature], dtype=np.int32
        )
        execution_name = f"exec_{position}"

        if direct:
            map_name = f"exec_map_{position}"
            declarations.append(
                f"static const int32_t {map_name}[{max(1, len(execution_map))}]="
                f"{_arr(execution_map, 'int')};"
            )
            raw_name = f"raw_state_{position}"
            body.extend(
                [
                    f"  long {raw_name}=std::lround(row[{input_feature}]);",
                    f"  if(!std::isfinite(row[{input_feature}]) || {raw_name}<0 || {raw_name}>={direct_cardinality}) {raw_name}=0;",
                    f"  const int32_t {execution_name}={map_name}[{raw_name}];",
                ]
            )
        else:
            direct_thresholds = model.encoder_._direct_level_thresholds(
                raw_feature, execution_level
            )
            if direct_thresholds is None:
                thresholds = np.asarray(
                    model.encoder_.thresholds_[raw_feature], dtype=np.float64
                )
                threshold_name = f"threshold_{position}"
                declarations.append(
                    f"static const double {threshold_name}[{max(1, len(thresholds))}]={_arr(thresholds)};"
                )
                if len(thresholds):
                    expression = " + ".join(
                        f"(row[{input_feature}]>={threshold_name}[{index}])"
                        for index in range(len(thresholds))
                    )
                    fine_name = f"fine_{position}"
                    body.append(f"  const int {fine_name}={expression};")
                else:
                    fine_name = f"fine_{position}"
                    body.append(f"  const int {fine_name}=0;")
                map_name = f"exec_map_{position}"
                declarations.append(
                    f"static const int32_t {map_name}[{max(1, len(execution_map))}]="
                    f"{_arr(execution_map, 'int')};"
                )
                body.append(
                    f"  const int32_t {execution_name}={map_name}[{fine_name}];"
                )
            else:
                thresholds = np.asarray(direct_thresholds, dtype=np.float64)
                threshold_name = f"exec_threshold_{position}"
                declarations.append(
                    f"static const double {threshold_name}[{max(1, len(thresholds))}]={_arr(thresholds)};"
                )
                if len(thresholds):
                    expression = " + ".join(
                        f"(row[{input_feature}]>={threshold_name}[{index}])"
                        for index in range(len(thresholds))
                    )
                    body.append(f"  const int32_t {execution_name}={expression};")
                else:
                    body.append(f"  const int32_t {execution_name}=0;")

        feature_maps = model.execution_parent_maps_[position]
        for level_index, level in enumerate(active_levels):
            state_name = f"s_{level}_{position}"
            parent_map = np.asarray(feature_maps[level_index], dtype=np.int32)
            if int(level) == execution_level:
                body.append(
                    f"  const int32_t {state_name}={execution_name};"
                )
            else:
                parent_name = f"parent_{level}_{position}"
                declarations.append(
                    f"static const int32_t {parent_name}[{max(1, len(parent_map))}]="
                    f"{_arr(parent_map, 'int')};"
                )
                body.append(
                    f"  const int32_t {state_name}={parent_name}[{execution_name}];"
                )
            state_names[level].append(state_name)

    body.append(f"  double score={float(model.intercept_):.17g};")

    lookup_index = 0
    if active_levels:
        for feature_position in range(n_features):
            for level in active_levels:
                table = np.asarray(model.lookup_[lookup_index], dtype=np.float64)
                table_name = f"lookup_{lookup_index}"
                declarations.append(
                    f"static const double {table_name}[{max(1, len(table))}]={_arr(table)};"
                )
                body.append(
                    f"  score += {table_name}[(int){state_names[level][feature_position]}];"
                )
                lookup_index += 1

        assert pair_level is not None
        pair_states = state_names[pair_level]
        for pair_index, (left, right) in enumerate(model.pairs_):
            table = np.asarray(model.lookup_[lookup_index], dtype=np.float64)
            table_name = f"lookup_{lookup_index}"
            declarations.append(
                f"static const double {table_name}[{max(1, len(table))}]={_arr(table)};"
            )
            right_cardinality = int(model.pair_cardinalities_[pair_index])
            body.append(
                f"  score += {table_name}[(int){pair_states[int(left)]}*{right_cardinality}+(int){pair_states[int(right)]}];"
            )
            lookup_index += 1

    if lookup_index != len(model.lookup_):
        raise RuntimeError(
            f"regression lookup layout mismatch: emitted {lookup_index}, expected {len(model.lookup_)}"
        )

    linear_positions = np.asarray(model.linear_positions_, dtype=np.int32)
    linear_mean = np.asarray(model.linear_mean_, dtype=np.float64)
    linear_scale = np.asarray(model.linear_scale_, dtype=np.float64)
    linear_coef = np.asarray(model.linear_coef_, dtype=np.float64)
    if len(linear_positions):
        body.append(f"  score += {float(model.linear_intercept_):.17g};")
        for index, position in enumerate(linear_positions):
            raw_feature = int(feature_idx[int(position)])
            input_linear_feature = int(position) if projected_input else raw_feature
            body.append(
                "  score += "
                f"((row[{input_linear_feature}]-({float(linear_mean[index]):.17g}))/({float(linear_scale[index]):.17g}))"
                f"*({float(linear_coef[index]):.17g});"
            )

    body.extend(["  out[i]=score;", " }", "}"])
    source.write_text("\n".join(declarations + body), encoding="utf-8")
    compile_shared_library(source, library)
    predictor = load_native_predictor(library, symbol="cerm_regression_predict")
    return predictor, source, library
