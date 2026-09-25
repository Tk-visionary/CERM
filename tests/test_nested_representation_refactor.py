import numpy as np

from cerm._internal.cerm_hierarchical_residual import NestedQuantileEncoder
from cerm._internal.cerm_nested_representation import (
    build_nested_codes,
    prepare_nested_residual_maps,
)


def _legacy_residual_state_map(child_to_parent):
    child_to_parent = np.asarray(child_to_parent, dtype=np.int64)
    n = len(child_to_parent)
    mapping = np.zeros(n, dtype=np.int32)
    if n == 0:
        return mapping
    child = np.arange(n, dtype=np.int64)
    order = np.lexsort((child, child_to_parent))
    ordered_parent = child_to_parent[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = ordered_parent[1:] != ordered_parent[:-1]
    residual_children = order[~first]
    mapping[residual_children] = np.arange(
        1, len(residual_children) + 1, dtype=np.int32
    )
    return mapping


def _legacy_parent(child_map, parent_map, child_card):
    child_map = np.asarray(child_map, dtype=np.int64)
    parent_map = np.asarray(parent_map, dtype=np.int64)
    first_index = np.full(int(child_card), len(child_map), dtype=np.int64)
    np.minimum.at(first_index, child_map, np.arange(len(child_map), dtype=np.int64))
    return parent_map[first_index]


def _legacy_maps(encoder, feature_idx, levels, pairs, fine_pairs):
    main = {}
    for level, parent in zip(levels[1:], levels[:-1]):
        values = []
        for raw_j in feature_idx:
            child_card = int(encoder.cardinalities_[level][raw_j])
            labels = _legacy_parent(
                encoder.maps_[level][raw_j],
                encoder.maps_[parent][raw_j],
                child_card,
            )
            values.append(_legacy_residual_state_map(labels))
        main[level] = values
    pair = {level: {} for level in levels[1:]}
    fine = set(fine_pairs)
    for j, k in pairs:
        raw_j, raw_k = int(feature_idx[j]), int(feature_idx[k])
        for level, parent in zip(levels[1:], levels[:-1]):
            if level > 8 and (j, k) not in fine:
                continue
            cj = int(encoder.cardinalities_[level][raw_j])
            ck = int(encoder.cardinalities_[level][raw_k])
            parent_ck = int(encoder.cardinalities_[parent][raw_k])
            pj = _legacy_parent(
                encoder.maps_[level][raw_j], encoder.maps_[parent][raw_j], cj
            )
            pk = _legacy_parent(
                encoder.maps_[level][raw_k], encoder.maps_[parent][raw_k], ck
            )
            parent_code = (pj[:, None] * parent_ck + pk[None, :]).ravel()
            pair[level][(j, k)] = _legacy_residual_state_map(parent_code)
    return main, pair


def _legacy_codes(states, encoder, feature_idx, levels, max_bins, max_main_level, pairs, fine_pairs, main, pair):
    coarse = levels[0]
    main_levels = [x for x in levels if x <= max_main_level]
    pair_levels = [x for x in levels if x <= min(max_bins, 8)]
    columns = []
    for j in range(states[coarse].shape[1]):
        columns.append(states[coarse][:, j])
        for level in main_levels[1:]:
            columns.append(main[level][j][states[level][:, j]])
    fine = set(fine_pairs)
    for j, k in pairs:
        raw_k = int(feature_idx[k])
        for level in pair_levels:
            card_k = int(encoder.cardinalities_[level][raw_k])
            joint = states[level][:, j].astype(np.int64) * card_k + states[level][:, k]
            columns.append(joint if level == coarse else pair[level][(j, k)][joint])
        if 16 in levels and (j, k) in fine:
            card_k = int(encoder.cardinalities_[16][raw_k])
            joint = states[16][:, j].astype(np.int64) * card_k + states[16][:, k]
            columns.append(pair[16][(j, k)][joint])
    return np.column_stack(columns)


def test_nested_engine_matches_legacy_semantic_layout():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(600, 7))
    encoder = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(X)
    feature_idx = np.asarray([0, 2, 4, 5, 6], dtype=np.int64)
    states_all = encoder.transform(X)
    states = {level: values[:, feature_idx] for level, values in states_all.items()}
    pairs = [(0, 1), (1, 3), (2, 4)]
    fine_pairs = [(0, 1)]
    legacy_main, legacy_pair = _legacy_maps(
        encoder, feature_idx, (4, 8, 16), pairs, fine_pairs
    )
    legacy = _legacy_codes(
        states,
        encoder,
        feature_idx,
        (4, 8, 16),
        16,
        16,
        pairs,
        fine_pairs,
        legacy_main,
        legacy_pair,
    )
    maps = prepare_nested_residual_maps(
        encoder=encoder,
        feature_idx=feature_idx,
        levels=(4, 8, 16),
        pairs=pairs,
        fine_pairs=fine_pairs,
    )
    actual = build_nested_codes(
        states,
        encoder=encoder,
        feature_idx=feature_idx,
        levels=(4, 8, 16),
        max_bins=16,
        max_main_level=16,
        pairs=pairs,
        fine_pairs=fine_pairs,
        maps=maps,
        dtype=np.int64,
    )
    assert np.array_equal(actual, legacy)


def test_binary_hybrid_backbone_uses_shared_nested_engine():
    from sklearn.datasets import make_classification
    from cerm import CERMClassifier

    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=6,
        n_redundant=1,
        random_state=19,
    )
    model = CERMClassifier(
        preset="balanced",
        max_features=10,
        max_interaction_features=8,
        max_interactions=6,
        calibration="none",
        n_jobs=1,
        random_state=23,
    ).fit(X, y)
    assert hasattr(model.model_.base_, "_nested_representation_maps_")
