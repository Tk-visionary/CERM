from __future__ import annotations

"""Task-neutral nested finite-state representation primitives.

This module owns the structural part of CERM's 4/8/16 nested state programs:
parent-residual maps, main/pair code materialization, and exact candidate-prefix
metadata.  Task learners remain responsible for ranking features/pairs and for
fitting their statistical heads.

The functions are deliberately pure with respect to the fitted encoder.  This
lets binary classification, shared multi-output learners, and regression
corrections reuse the same semantic code layout without sharing task losses.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def residual_state_map(child_to_parent: np.ndarray) -> np.ndarray:
    """Return the exact residual code map in parent-major order."""
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


def parent_labels_from_fine(
    child_map: np.ndarray,
    parent_map: np.ndarray,
    child_cardinality: int,
) -> np.ndarray:
    """Compose child and parent quotient maps without repeated searches."""
    child_map = np.asarray(child_map, dtype=np.int64)
    parent_map = np.asarray(parent_map, dtype=np.int64)
    first_index = np.full(int(child_cardinality), len(child_map), dtype=np.int64)
    np.minimum.at(
        first_index, child_map, np.arange(len(child_map), dtype=np.int64)
    )
    if np.any(first_index == len(child_map)):
        raise ValueError("child quotient contains an unreachable state")
    return parent_map[first_index]


def pair_parent_map(
    child_card_j: int,
    child_card_k: int,
    map_j: np.ndarray,
    map_k: np.ndarray,
    parent_card_k: int,
) -> np.ndarray:
    """Map child pair codes to their parent pair codes."""
    map_j = np.asarray(map_j, dtype=np.int64)[: int(child_card_j)]
    map_k = np.asarray(map_k, dtype=np.int64)[: int(child_card_k)]
    return (
        map_j[:, None] * int(parent_card_k) + map_k[None, :]
    ).ravel()


@dataclass(frozen=True)
class NestedResidualMaps:
    main: dict[int, tuple[np.ndarray, ...]]
    pair: dict[int, dict[tuple[int, int], np.ndarray]]


def prepare_nested_residual_maps(
    *,
    encoder,
    feature_idx: Sequence[int],
    levels: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    fine_pairs: Sequence[tuple[int, int]],
) -> NestedResidualMaps:
    """Build deterministic nested residual maps for main and pair states."""
    feature_idx = np.asarray(feature_idx, dtype=np.int64)
    levels = tuple(int(level) for level in levels)

    main: dict[int, tuple[np.ndarray, ...]] = {}
    for level, parent_level in zip(levels[1:], levels[:-1]):
        maps = []
        for raw_j in feature_idx:
            raw_j = int(raw_j)
            child_map = encoder.maps_[level][raw_j]
            parent_map = encoder.maps_[parent_level][raw_j]
            child_card = int(encoder.cardinalities_[level][raw_j])
            labels = parent_labels_from_fine(child_map, parent_map, child_card)
            maps.append(residual_state_map(labels))
        main[level] = tuple(maps)

    pair: dict[int, dict[tuple[int, int], np.ndarray]] = {
        level: {} for level in levels[1:]
    }
    fine = set(tuple(map(int, value)) for value in fine_pairs)
    for j, k in pairs:
        j, k = int(j), int(k)
        raw_j, raw_k = int(feature_idx[j]), int(feature_idx[k])
        for level, parent_level in zip(levels[1:], levels[:-1]):
            if level > 8 and (j, k) not in fine:
                continue
            child_card_j = int(encoder.cardinalities_[level][raw_j])
            child_card_k = int(encoder.cardinalities_[level][raw_k])
            parent_card_k = int(encoder.cardinalities_[parent_level][raw_k])
            parent_j = parent_labels_from_fine(
                encoder.maps_[level][raw_j],
                encoder.maps_[parent_level][raw_j],
                child_card_j,
            )
            parent_k = parent_labels_from_fine(
                encoder.maps_[level][raw_k],
                encoder.maps_[parent_level][raw_k],
                child_card_k,
            )
            parent = pair_parent_map(
                child_card_j,
                child_card_k,
                parent_j,
                parent_k,
                parent_card_k,
            )
            pair[level][(j, k)] = residual_state_map(parent)

    return NestedResidualMaps(main=main, pair=pair)


def build_nested_codes(
    states: dict[int, np.ndarray],
    *,
    encoder,
    feature_idx: Sequence[int],
    levels: Sequence[int],
    max_bins: int,
    max_main_level: int,
    pairs: Sequence[tuple[int, int]],
    fine_pairs: Sequence[tuple[int, int]],
    maps: NestedResidualMaps,
    dtype=np.int64,
) -> np.ndarray:
    """Materialize the canonical nested CERM semantic code matrix.

    Column order is exactly:
      feature-major main states (coarse, then residual refinements),
      pair-major states through level 8,
      optional level-16 pair residuals.
    """
    levels = tuple(int(level) for level in levels)
    coarse = levels[0]
    n_rows = len(states[coarse])
    main_levels = [level for level in levels if level <= int(max_main_level)]
    pair_levels = [level for level in levels if level <= min(int(max_bins), 8)]
    n_columns = (
        states[coarse].shape[1] * len(main_levels)
        + len(pairs) * len(pair_levels)
        + len(fine_pairs) * int(16 in levels)
    )
    if n_columns == 0:
        return np.zeros((n_rows, 1), dtype=dtype)

    codes = np.empty((n_rows, n_columns), dtype=dtype)
    column = 0
    for j in range(states[coarse].shape[1]):
        codes[:, column] = states[coarse][:, j]
        column += 1
        for level in main_levels[1:]:
            codes[:, column] = maps.main[level][j][states[level][:, j]]
            column += 1

    fine = set(tuple(map(int, value)) for value in fine_pairs)
    joint = np.empty(n_rows, dtype=np.int64)
    feature_idx = np.asarray(feature_idx, dtype=np.int64)
    for j, k in pairs:
        j, k = int(j), int(k)
        raw_j, raw_k = int(feature_idx[j]), int(feature_idx[k])
        for level in pair_levels:
            card_k = int(encoder.cardinalities_[level][raw_k])
            np.multiply(states[level][:, j], card_k, out=joint, casting="unsafe")
            joint += states[level][:, k]
            if level == coarse:
                codes[:, column] = joint
            else:
                codes[:, column] = maps.pair[level][(j, k)][joint]
            column += 1
        if 16 in levels and (j, k) in fine:
            card_k = int(encoder.cardinalities_[16][raw_k])
            np.multiply(states[16][:, j], card_k, out=joint, casting="unsafe")
            joint += states[16][:, k]
            codes[:, column] = maps.pair[16][(j, k)][joint]
            column += 1

    if column != n_columns:
        raise RuntimeError("nested finite-state code layout mismatch")
    return codes


def maximal_code_metadata(
    *,
    levels: Sequence[int],
    n_features: int,
    n_pairs: int,
    n_fine_pairs: int,
) -> tuple[tuple[str, int, int], ...]:
    """Describe columns of a maximal nested code bank canonically."""
    metadata: list[tuple[str, int, int]] = []
    levels = tuple(int(level) for level in levels)
    for feature in range(int(n_features)):
        for level in levels:
            metadata.append(("main", level, feature))
    for pair_index in range(int(n_pairs)):
        for level in levels:
            if level <= 8 or pair_index < int(n_fine_pairs):
                metadata.append(("pair", level, pair_index))
    return tuple(metadata)


def candidate_code_columns(
    metadata: Sequence[tuple[str, int, int]],
    *,
    max_main_level: int,
    n_pairs: int,
    n_fine_pairs: int,
) -> list[int]:
    """Return exact maximal-bank code columns for one nested prefix."""
    selected: list[int] = []
    for column, (kind, level, index) in enumerate(metadata):
        if kind == "main" and int(level) <= int(max_main_level):
            selected.append(column)
        elif kind == "pair":
            if int(level) <= 8 and int(index) < int(n_pairs):
                selected.append(column)
            elif int(level) > 8 and int(index) < int(n_fine_pairs):
                selected.append(column)
    return selected


class NestedRepresentationMixin:
    """Mixin that preserves the legacy nested semantic-code contract exactly."""

    def _prepare_residual_maps(self):
        maps = prepare_nested_residual_maps(
            encoder=self.encoder_,
            feature_idx=self.feature_idx_,
            levels=self.levels,
            pairs=self.pairs_,
            fine_pairs=self.fine_pairs_,
        )
        self._nested_representation_maps_ = maps
        # Keep historical mutable/list-shaped attributes for downstream code.
        self.main_residuals_ = {
            level: list(values) for level, values in maps.main.items()
        }
        self.pair_residuals_ = {
            level: dict(values) for level, values in maps.pair.items()
        }
        self.main_res8_ = self.main_residuals_.get(8, [])
        self.main_res16_ = self.main_residuals_.get(16, [])
        self.pair_res8_ = self.pair_residuals_.get(8, {})
        self.pair_res16_ = self.pair_residuals_.get(16, {})

    def _build_codes_from_states(self, states):
        maps = getattr(self, "_nested_representation_maps_", None)
        if maps is None:
            self._prepare_residual_maps()
            maps = self._nested_representation_maps_
        return build_nested_codes(
            states,
            encoder=self.encoder_,
            feature_idx=self.feature_idx_,
            levels=self.levels,
            max_bins=self.max_bins,
            max_main_level=self.config_.max_main_level,
            pairs=self.pairs_,
            fine_pairs=self.fine_pairs_,
            maps=maps,
            dtype=np.int64,
        )
