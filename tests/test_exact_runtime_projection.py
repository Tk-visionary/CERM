from __future__ import annotations

import numpy as np

from cerm import CERMRidgeRegressor as CERMRegressor, PortableRegressionProgram
from cerm.regression import (
    FiniteStateRidgeRegressor,
    _between_group_score,
    _regression_target_stats,
)


def _fit_pair_model(seed: int = 20260811):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(420, 36))
    y = (
        1.8 * ((X[:, 0] > 0) ^ (X[:, 1] > 0))
        + 1.1 * ((X[:, 2] > 0.25) & (X[:, 3] < -0.2))
        + 0.4 * X[:, 8]
        + rng.normal(scale=0.25, size=len(X))
    )
    model = CERMRegressor(
        max_features=16,
        max_interaction_features=12,
        max_interactions=6,
        include_linear=True,
        n_jobs=1,
        random_state=seed,
    )
    model._forced_representation_config = (16, 6, 1.0)
    model.fit(X, y)
    return model, X


def _legacy_regression_predict(model: FiniteStateRidgeRegressor, X: np.ndarray):
    array = np.asarray(X, dtype=np.float64)
    prediction = np.full(len(array), model.intercept_, dtype=np.float64)
    if model.config_.max_main_level > 0:
        all_states = model.encoder_.transform(array)
        states = {
            level: matrix[:, model.feature_idx_] for level, matrix in all_states.items()
        }
        codes = model._codes(
            states, model.pairs_, model.config_, model.pair_cardinalities_
        )
        for column, table in enumerate(model.lookup_):
            state = codes[:, column]
            valid = (state >= 0) & (state < len(table))
            prediction[valid] += table[state[valid]]
    if len(model.linear_positions_):
        selected = array[:, model.feature_idx_]
        linear = (
            selected[:, model.linear_positions_] - model.linear_mean_
        ) / model.linear_scale_
        prediction += model.linear_intercept_ + linear @ model.linear_coef_
    return prediction


def _legacy_portable_predict(program: PortableRegressionProgram, X):
    matrix = program._matrix(X)
    selected_indices = program.arrays["feature_indices"].astype(np.int64)
    selected = matrix[:, selected_indices]
    prediction = np.full(
        len(matrix), float(program.arrays["intercept"][0]), dtype=np.float64
    )
    max_main_level = int(program.model_metadata["max_main_level"])
    if max_main_level > 0:
        direct_mask = program.arrays["direct_state_mask"].astype(bool)
        direct_cards = program.arrays["direct_state_cardinalities"].astype(np.int64)
        fine = np.empty(matrix.shape, dtype=np.int16)
        for feature in range(matrix.shape[1]):
            values = matrix[:, feature]
            if direct_mask[feature]:
                state = np.rint(values).astype(np.int64)
                if np.any(~np.isfinite(values)) or np.any(
                    (state < 0) | (state >= direct_cards[feature])
                ):
                    raise ValueError
                fine[:, feature] = state
            else:
                fine[:, feature] = np.searchsorted(
                    program.arrays[f"thresholds_{feature}"], values, side="right"
                )
        states_all = {}
        for level in program.model_metadata["levels"]:
            transformed = np.empty_like(fine)
            for feature in range(matrix.shape[1]):
                transformed[:, feature] = program.arrays[
                    f"map_{int(level)}_{feature}"
                ][fine[:, feature]]
            states_all[int(level)] = transformed
        states = {
            level: values[:, selected_indices] for level, values in states_all.items()
        }
        active_levels = [
            int(level)
            for level in program.model_metadata["levels"]
            if int(level) <= max_main_level
        ]
        pair_level = max(
            level for level in active_levels if level <= min(max_main_level, 8)
        )
        columns = []
        for position in range(len(selected_indices)):
            for level in active_levels:
                columns.append(states[level][:, position].astype(np.int64, copy=False))
        pairs = program.arrays["pairs"].astype(np.int64).reshape(-1, 2)
        cards = program.arrays["pair_cardinalities"].astype(np.int64)
        for pair_index, (left, right) in enumerate(pairs):
            columns.append(
                states[pair_level][:, left].astype(np.int64) * int(cards[pair_index])
                + states[pair_level][:, right].astype(np.int64)
            )
        codes = np.column_stack(columns)
        offsets = program.arrays["lookup_offsets"].astype(np.int64)
        values = program.arrays["lookup_values"].astype(np.float64)
        for column in range(codes.shape[1]):
            table = values[offsets[column] : offsets[column + 1]]
            state = codes[:, column]
            valid = (state >= 0) & (state < len(table))
            prediction[valid] += table[state[valid]]
    positions = program.arrays["linear_positions"].astype(np.int64)
    if len(positions):
        linear = (
            selected[:, positions] - program.arrays["linear_mean"]
        ) / program.arrays["linear_scale"]
        prediction += float(program.arrays["linear_intercept"][0])
        prediction += linear @ program.arrays["linear_coef"]
    return prediction


def test_regression_projected_stream_matches_legacy_bitwise():
    model, X = _fit_pair_model()
    probe = np.vstack([X[:80], X[-80:]])
    expected = _legacy_regression_predict(model.model_, probe)
    actual = model.model_.predict(probe)
    assert np.array_equal(actual, expected)


def test_portable_regression_projected_stream_matches_legacy_bitwise(tmp_path):
    model, X = _fit_pair_model(20260812)
    manifest = model.export(tmp_path / "portable")
    portable = PortableRegressionProgram.load(manifest.parent)
    probe = np.vstack([X[:70], X[-70:]])
    expected = _legacy_portable_predict(portable, probe)
    actual = portable.predict(probe)
    assert np.array_equal(actual, expected)


def test_regression_pair_rank_fortran_projection_preserves_order():
    rng = np.random.default_rng(99)
    states = rng.integers(0, 16, size=(600, 14), dtype=np.int16)
    y = rng.normal(size=len(states))
    model = FiniteStateRidgeRegressor(
        max_features=14,
        max_interaction_features=14,
        max_interactions=12,
        n_jobs=1,
    )
    stats = _regression_target_stats(y)
    singles = np.asarray(
        [_between_group_score(states[:, j], y, target_stats=stats) for j in range(14)]
    )
    actual = model._pair_rank(
        states, y, target_stats=stats, single_scores=singles
    )
    cards = states.max(axis=0, initial=0).astype(np.int64) + 1
    rows = []
    for j in range(14):
        for k in range(j + 1, 14):
            joint = states[:, j].astype(np.int64) * int(cards[k]) + states[:, k]
            score = _between_group_score(joint, y, target_stats=stats) - singles[j] - singles[k]
            rows.append((float(score), j, k))
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    expected = [(j, k) for _, j, k in rows[:12]]
    assert actual == expected


def test_regression_native_codegen_is_branchless_and_semantically_equal(tmp_path):
    model, X = _fit_pair_model(20260813)
    compiled = model.compile_native(tmp_path / "native")
    source = compiled.source_path.read_text(encoding="utf-8")
    assert "while(" not in source
    np.testing.assert_allclose(
        compiled.predict(X[:100]), model.predict(X[:100]), rtol=0.0, atol=2e-12
    )
