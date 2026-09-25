from __future__ import annotations

"""Portable semantic IR for the internal fused residual regression V2.

The outer package shape intentionally follows the existing regression export
contract: JSON metadata + deterministic NPZ arrays referenced by a
``cerm-python-package-v3`` manifest.  The execution object below depends only on
NumPy and the serialized IR, not on fitted sklearn/CERM estimator objects.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_MODEL_FORMAT = "cerm-fused-residual-regression-v1"
_PACKAGE_FORMAT = "cerm-python-package-v3"
_KINDS = {"current_mean", "fused", "raw_fused"}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _integrate_survival(
    lower: float,
    upper: float,
    thresholds: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    boundaries = np.r_[lower, thresholds, upper]
    survival = np.column_stack(
        [
            np.ones(len(probability), dtype=np.float64),
            probability,
            np.zeros(len(probability), dtype=np.float64),
        ]
    )
    return lower + np.sum(
        np.diff(boundaries)[None, :]
        * 0.5
        * (survival[:, :-1] + survival[:, 1:]),
        axis=1,
    )


def _file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "file": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _require(metadata: Mapping[str, Any], key: str) -> Any:
    if key not in metadata:
        raise ValueError(f"missing fused regression IR field: {key}")
    return metadata[key]


def _array(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"missing fused regression IR array: {key}")
    return np.asarray(arrays[key])


def _finite_float(value: Any, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _flatten_reference_state_coefficients(state_encoder, state_coef: np.ndarray):
    state_coef = np.asarray(state_coef, dtype=np.float64)
    offsets = [0]
    tables = []
    cursor = 0
    for cardinality in np.asarray(state_encoder.cardinalities_, dtype=np.int64):
        cardinality = int(cardinality)
        table = np.zeros((cardinality, state_coef.shape[1]), dtype=np.float64)
        width = max(cardinality - 1, 0)
        if width:
            table[1:] = state_coef[cursor : cursor + width]
        cursor += width
        tables.append(table)
        offsets.append(offsets[-1] + cardinality)
    if cursor != state_coef.shape[0]:
        raise RuntimeError("fused reference-state coefficient layout mismatch")
    values = (
        np.vstack(tables)
        if tables
        else np.empty((0, state_coef.shape[1]), dtype=np.float64)
    )
    return np.asarray(offsets, dtype=np.int64), values


def _flatten_raw_state_coefficients(raw, state_coef: np.ndarray):
    state_coef = np.asarray(state_coef, dtype=np.float64)
    offsets = [0]
    tables = []
    cursor = 0
    for resolution in raw.resolutions:
        cards = np.asarray(raw.state_sets_[int(resolution)][2], dtype=np.int64)
        for cardinality in cards:
            cardinality = int(cardinality)
            tables.append(state_coef[cursor : cursor + cardinality])
            cursor += cardinality
            offsets.append(offsets[-1] + cardinality)
    pair_cards = np.asarray(
        raw.state_sets_[int(raw.pair_resolution)][2], dtype=np.int64
    )
    for left, right in raw.pairs_:
        cardinality = int(pair_cards[int(left)]) * int(pair_cards[int(right)])
        tables.append(state_coef[cursor : cursor + cardinality])
        cursor += cardinality
        offsets.append(offsets[-1] + cardinality)
    if cursor != state_coef.shape[0]:
        raise RuntimeError("raw fused coefficient layout mismatch")
    values = (
        np.vstack(tables)
        if tables
        else np.empty((0, state_coef.shape[1]), dtype=np.float64)
    )
    return np.asarray(offsets, dtype=np.int64), values


def _extract_baseline(model) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    lookup_offsets = np.concatenate(
        [[0], np.cumsum([len(table) for table in model.lookup_])]
    ).astype(np.int64)
    lookup_values = (
        np.concatenate(model.lookup_)
        if model.lookup_
        else np.empty(0, dtype=np.float64)
    )
    arrays: dict[str, np.ndarray] = {
        "feature_indices": np.asarray(model.feature_idx_, dtype=np.int32),
        "pairs": np.asarray(model.pairs_, dtype=np.int32).reshape(-1, 2),
        "pair_cardinalities": np.asarray(
            model.pair_cardinalities_, dtype=np.int32
        ),
        "direct_state_mask": np.asarray(
            model.encoder_.direct_state_mask_, dtype=np.uint8
        ),
        "direct_state_cardinalities": np.asarray(
            model.encoder_.direct_state_cardinalities_, dtype=np.int32
        ),
        "lookup_offsets": lookup_offsets,
        "lookup_values": np.asarray(lookup_values, dtype=np.float64),
        "intercept": np.asarray([model.intercept_], dtype=np.float64),
        "linear_positions": np.asarray(model.linear_positions_, dtype=np.int32),
        "linear_mean": np.asarray(model.linear_mean_, dtype=np.float64),
        "linear_scale": np.asarray(model.linear_scale_, dtype=np.float64),
        "linear_coef": np.asarray(model.linear_coef_, dtype=np.float64),
        "linear_intercept": np.asarray(
            [model.linear_intercept_], dtype=np.float64
        ),
    }
    for index, thresholds in enumerate(model.encoder_.thresholds_):
        arrays[f"thresholds_{index}"] = np.asarray(
            thresholds, dtype=np.float64
        )
    for level in model.levels:
        for index, mapping in enumerate(model.encoder_.maps_[level]):
            arrays[f"map_{int(level)}_{index}"] = np.asarray(
                mapping, dtype=np.int32
            )
    metadata = {
        "format": "cerm-finite-state-regression-v2",
        "levels": [int(level) for level in model.levels],
        "max_main_level": int(model.config_.max_main_level),
        "n_pairs": int(model.config_.n_pairs),
        "alpha": float(model.config_.alpha),
        "n_features_in": int(model.encoder_.n_features_in_),
        "design_dimension": int(model.design_dim_),
    }
    return metadata, arrays


def _prefix_arrays(
    target: dict[str, np.ndarray], prefix: str, source: Mapping[str, np.ndarray]
) -> None:
    for name, value in source.items():
        target[f"{prefix}{name}"] = np.asarray(value)


def _unprefix_arrays(
    arrays: Mapping[str, np.ndarray], prefix: str
) -> dict[str, np.ndarray]:
    return {
        key[len(prefix) :]: np.asarray(value)
        for key, value in arrays.items()
        if key.startswith(prefix)
    }


def _extract_head(head) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    thresholds = np.asarray(head.thresholds_, dtype=np.float64)
    basis = np.asarray(
        head._basis(thresholds, float(head.lower_), float(head.upper_)),
        dtype=np.float64,
    )
    metadata = {
        "lower": float(head.lower_),
        "upper": float(head.upper_),
        "rank": int(head.threshold_basis_dim_),
        "n_thresholds": int(len(thresholds)),
    }
    arrays = {
        "thresholds": thresholds,
        "threshold_basis": basis,
        "threshold_coef": np.asarray(head.threshold_coef_, dtype=np.float64),
        "intercept": np.asarray([head.intercept_], dtype=np.float64),
    }
    return metadata, arrays


def semantic_program_from_fused_regressor(model) -> "FusedRegressionSemanticProgram":
    """Lower a fitted ``FusedResidualCERMRegressor`` to portable semantic IR."""

    selected_kind = str(getattr(model, "selected_kind_", ""))
    if selected_kind not in _KINDS:
        raise ValueError("fused regression model is not fitted or has unknown branch")
    baseline_metadata, baseline_arrays = _extract_baseline(model.baseline_)
    arrays: dict[str, np.ndarray] = {}
    _prefix_arrays(arrays, "baseline__", baseline_arrays)
    residual_thresholds = np.asarray(model.thresholds_, dtype=np.float64)
    arrays["residual_thresholds"] = residual_thresholds
    metadata: dict[str, Any] = {
        "format": _MODEL_FORMAT,
        "task_type": "regression",
        "engine": "fused-residual-v2",
        "selected_kind": selected_kind,
        "n_features_in": int(model.n_features_in_),
        "residual_scale": float(model.residual_scale_),
        "residual_lower": float(model.lower_),
        "residual_upper": float(model.upper_),
        "baseline": baseline_metadata,
        "shared": None,
        "raw": None,
    }

    if selected_kind == "fused":
        representation = model.representation_
        head = model.head_
        head_metadata, head_arrays = _extract_head(head)
        feature_idx = np.asarray(representation.feature_idx, dtype=np.int32)
        pairs = np.asarray(representation.pairs, dtype=np.int32).reshape(-1, 2)
        fine_pairs = np.asarray(
            representation.fine_pairs, dtype=np.int32
        ).reshape(-1, 2)
        arrays["shared__feature_indices"] = feature_idx
        arrays["shared__pairs"] = pairs
        arrays["shared__fine_pairs"] = fine_pairs
        arrays["shared__direct_state_mask"] = np.asarray(
            representation.encoder.direct_state_mask_[feature_idx], dtype=np.uint8
        )
        arrays["shared__direct_state_cardinalities"] = np.asarray(
            representation.encoder.direct_state_cardinalities_[feature_idx],
            dtype=np.int32,
        )
        for position, raw_feature in enumerate(feature_idx.astype(np.int64)):
            arrays[f"shared__thresholds_{position}"] = np.asarray(
                representation.encoder.thresholds_[int(raw_feature)],
                dtype=np.float64,
            )
            for level in representation.levels:
                arrays[f"shared__map_{int(level)}_{position}"] = np.asarray(
                    representation.encoder.maps_[int(level)][int(raw_feature)],
                    dtype=np.int32,
                )
        for level, values in representation.maps.main.items():
            for feature, mapping in enumerate(values):
                arrays[f"shared__main_residual_{int(level)}_{feature}"] = (
                    np.asarray(mapping, dtype=np.int32)
                )
        pair_to_index = {
            tuple(map(int, pair)): index
            for index, pair in enumerate(representation.pairs)
        }
        for level, mappings in representation.maps.pair.items():
            for pair, mapping in mappings.items():
                pair_index = pair_to_index[tuple(map(int, pair))]
                arrays[
                    f"shared__pair_residual_{int(level)}_{pair_index}"
                ] = np.asarray(mapping, dtype=np.int32)
        lookup_offsets, lookup_values = _flatten_reference_state_coefficients(
            representation.state_encoder, head.state_coef_
        )
        arrays["shared__lookup_offsets"] = lookup_offsets
        arrays["shared__lookup_values"] = lookup_values
        _prefix_arrays(arrays, "shared__head__", head_arrays)
        metadata["shared"] = {
            "levels": [int(level) for level in representation.levels],
            "max_bins": int(representation.max_bins),
            "max_main_level": int(representation.max_main_level),
            "n_features": int(len(feature_idx)),
            "n_pairs": int(len(pairs)),
            "n_fine_pairs": int(len(fine_pairs)),
            "head": head_metadata,
        }

    if selected_kind == "raw_fused":
        raw = model.raw_correction_
        head = raw.head_
        head_metadata, head_arrays = _extract_head(head)
        pairs = np.asarray(raw.pairs_, dtype=np.int32).reshape(-1, 2)
        arrays["raw__pairs"] = pairs
        for resolution in raw.resolutions:
            edges, _, cards = raw.state_sets_[int(resolution)]
            arrays[f"raw__cards_{int(resolution)}"] = np.asarray(
                cards, dtype=np.int32
            )
            for feature, feature_edges in enumerate(edges):
                arrays[f"raw__edges_{int(resolution)}_{feature}"] = np.asarray(
                    feature_edges, dtype=np.float64
                )
        if int(raw.pair_resolution) not in raw.resolutions:
            edges, _, cards = raw.state_sets_[int(raw.pair_resolution)]
            arrays[f"raw__cards_{int(raw.pair_resolution)}"] = np.asarray(
                cards, dtype=np.int32
            )
            for feature, feature_edges in enumerate(edges):
                arrays[
                    f"raw__edges_{int(raw.pair_resolution)}_{feature}"
                ] = np.asarray(feature_edges, dtype=np.float64)
        lookup_offsets, lookup_values = _flatten_raw_state_coefficients(
            raw, head.state_coef_
        )
        arrays["raw__lookup_offsets"] = lookup_offsets
        arrays["raw__lookup_values"] = lookup_values
        _prefix_arrays(arrays, "raw__head__", head_arrays)
        metadata["raw"] = {
            "resolutions": [int(value) for value in raw.resolutions],
            "pair_resolution": int(raw.pair_resolution),
            "n_features": int(model.n_features_in_),
            "n_pairs": int(len(pairs)),
            "head": head_metadata,
        }

    return FusedRegressionSemanticProgram(metadata=metadata, arrays=arrays)


@dataclass
class FusedRegressionSemanticProgram:
    """Pure-NumPy reference executor for fused regression semantic IR."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata)
        self.arrays = {key: np.asarray(value) for key, value in self.arrays.items()}
        self.validate()

    @property
    def selected_kind(self) -> str:
        return str(self.metadata["selected_kind"])

    def validate(self) -> "FusedRegressionSemanticProgram":
        if _require(self.metadata, "format") != _MODEL_FORMAT:
            raise ValueError("unsupported fused regression IR format")
        if _require(self.metadata, "task_type") != "regression":
            raise ValueError("fused regression IR task_type must be regression")
        selected_kind = str(_require(self.metadata, "selected_kind"))
        if selected_kind not in _KINDS:
            raise ValueError(f"unsupported fused regression branch: {selected_kind!r}")
        n_features = int(_require(self.metadata, "n_features_in"))
        if n_features < 1:
            raise ValueError("n_features_in must be positive")
        _finite_float(_require(self.metadata, "residual_scale"), "residual_scale")
        lower = _finite_float(
            _require(self.metadata, "residual_lower"), "residual_lower"
        )
        upper = _finite_float(
            _require(self.metadata, "residual_upper"), "residual_upper"
        )
        if upper < lower:
            raise ValueError("residual integration bounds are reversed")
        residual_thresholds = _array(self.arrays, "residual_thresholds")
        if residual_thresholds.ndim != 1:
            raise ValueError("residual_thresholds must be one-dimensional")
        if np.any(np.diff(residual_thresholds) <= 0):
            raise ValueError("residual_thresholds must be strictly increasing")
        self._validate_baseline(_require(self.metadata, "baseline"), n_features)
        if selected_kind == "fused":
            shared = _require(self.metadata, "shared")
            if not isinstance(shared, dict):
                raise ValueError("shared fused branch requires shared metadata")
            self._validate_shared(shared, n_features)
        if selected_kind == "raw_fused":
            raw = _require(self.metadata, "raw")
            if not isinstance(raw, dict):
                raise ValueError("raw fused branch requires raw metadata")
            self._validate_raw(raw, n_features)
        return self

    def _validate_baseline(self, metadata: Any, n_features: int) -> None:
        if not isinstance(metadata, dict):
            raise ValueError("baseline metadata must be an object")
        levels = tuple(int(value) for value in _require(metadata, "levels"))
        if not levels:
            raise ValueError("baseline levels must not be empty")
        max_main = int(_require(metadata, "max_main_level"))
        feature_idx = _array(self.arrays, "baseline__feature_indices").astype(
            np.int64, copy=False
        )
        if feature_idx.ndim != 1 or np.any(
            (feature_idx < 0) | (feature_idx >= n_features)
        ):
            raise ValueError("baseline feature_indices are invalid")
        pairs = _array(self.arrays, "baseline__pairs")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("baseline pairs must have shape (n, 2)")
        pair_cards = _array(self.arrays, "baseline__pair_cardinalities")
        if len(pair_cards) != len(pairs):
            raise ValueError("baseline pair cardinality width mismatch")
        offsets = _array(self.arrays, "baseline__lookup_offsets").astype(
            np.int64, copy=False
        )
        values = _array(self.arrays, "baseline__lookup_values")
        if offsets.ndim != 1 or len(offsets) < 1 or offsets[0] != 0:
            raise ValueError("baseline lookup offsets are invalid")
        if np.any(np.diff(offsets) < 0) or offsets[-1] != len(values):
            raise ValueError("baseline lookup values do not match offsets")
        active_levels = [level for level in levels if level <= max_main]
        expected_tables = len(feature_idx) * len(active_levels) + len(pairs)
        if len(offsets) != expected_tables + 1:
            raise ValueError("baseline lookup table count mismatch")
        direct_mask = _array(self.arrays, "baseline__direct_state_mask")
        direct_cards = _array(
            self.arrays, "baseline__direct_state_cardinalities"
        )
        if len(direct_mask) != n_features or len(direct_cards) != n_features:
            raise ValueError("baseline encoder width mismatch")
        for raw_feature in feature_idx:
            raw_feature = int(raw_feature)
            _array(self.arrays, f"baseline__thresholds_{raw_feature}")
            for level in levels:
                _array(self.arrays, f"baseline__map_{level}_{raw_feature}")
        for key in (
            "baseline__intercept",
            "baseline__linear_positions",
            "baseline__linear_mean",
            "baseline__linear_scale",
            "baseline__linear_coef",
            "baseline__linear_intercept",
        ):
            _array(self.arrays, key)

    def _validate_head(self, prefix: str, metadata: Mapping[str, Any]) -> int:
        rank = int(_require(metadata, "rank"))
        n_thresholds = int(_require(metadata, "n_thresholds"))
        if rank < 1 or n_thresholds < 1:
            raise ValueError("fused threshold head dimensions must be positive")
        lower = _finite_float(_require(metadata, "lower"), f"{prefix} lower")
        upper = _finite_float(_require(metadata, "upper"), f"{prefix} upper")
        if upper <= lower:
            raise ValueError("fused threshold head bounds must be increasing")
        thresholds = _array(self.arrays, f"{prefix}thresholds")
        basis = _array(self.arrays, f"{prefix}threshold_basis")
        threshold_coef = _array(self.arrays, f"{prefix}threshold_coef")
        intercept = _array(self.arrays, f"{prefix}intercept")
        if thresholds.shape != (n_thresholds,):
            raise ValueError("threshold head threshold count mismatch")
        if basis.shape != (n_thresholds, rank):
            raise ValueError("threshold basis shape mismatch")
        if threshold_coef.shape != (rank,) or intercept.shape != (1,):
            raise ValueError("threshold head coefficient shape mismatch")
        if not all(
            np.all(np.isfinite(value))
            for value in (thresholds, basis, threshold_coef, intercept)
        ):
            raise ValueError("threshold head contains non-finite values")
        return rank

    def _validate_shared(self, metadata: Mapping[str, Any], n_features: int) -> None:
        levels = tuple(int(value) for value in _require(metadata, "levels"))
        if not levels:
            raise ValueError("shared levels must not be empty")
        n_selected = int(_require(metadata, "n_features"))
        feature_idx = _array(self.arrays, "shared__feature_indices").astype(
            np.int64, copy=False
        )
        if feature_idx.shape != (n_selected,) or np.any(
            (feature_idx < 0) | (feature_idx >= n_features)
        ):
            raise ValueError("shared feature_indices are invalid")
        direct_mask = _array(self.arrays, "shared__direct_state_mask")
        direct_cards = _array(
            self.arrays, "shared__direct_state_cardinalities"
        )
        if direct_mask.shape != (n_selected,) or direct_cards.shape != (n_selected,):
            raise ValueError("shared encoder width mismatch")
        for feature in range(n_selected):
            _array(self.arrays, f"shared__thresholds_{feature}")
            for level in levels:
                _array(self.arrays, f"shared__map_{level}_{feature}")
        pairs = _array(self.arrays, "shared__pairs").astype(np.int64, copy=False)
        fine_pairs = _array(self.arrays, "shared__fine_pairs").astype(
            np.int64, copy=False
        )
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("shared pairs must have shape (n, 2)")
        if fine_pairs.ndim != 2 or fine_pairs.shape[1] != 2:
            raise ValueError("shared fine_pairs must have shape (n, 2)")
        if len(pairs) != int(_require(metadata, "n_pairs")):
            raise ValueError("shared pair count mismatch")
        if len(fine_pairs) != int(_require(metadata, "n_fine_pairs")):
            raise ValueError("shared fine pair count mismatch")
        if np.any((pairs < 0) | (pairs >= n_selected)):
            raise ValueError("shared pair index is out of range")
        max_main = int(_require(metadata, "max_main_level"))
        coarse = levels[0]
        main_levels = [level for level in levels if level <= max_main]
        for level in main_levels[1:]:
            for feature in range(n_selected):
                _array(self.arrays, f"shared__main_residual_{level}_{feature}")
        fine_set = {tuple(map(int, pair)) for pair in fine_pairs}
        for pair_index, pair in enumerate(pairs):
            pair_tuple = tuple(map(int, pair))
            for level in levels:
                if level <= 8 and level != coarse:
                    _array(
                        self.arrays,
                        f"shared__pair_residual_{level}_{pair_index}",
                    )
                if level > 8 and pair_tuple in fine_set:
                    _array(
                        self.arrays,
                        f"shared__pair_residual_{level}_{pair_index}",
                    )
        rank = self._validate_head("shared__head__", _require(metadata, "head"))
        offsets = _array(self.arrays, "shared__lookup_offsets").astype(
            np.int64, copy=False
        )
        values = _array(self.arrays, "shared__lookup_values")
        expected_codes = n_selected * len(main_levels)
        expected_codes += len(pairs) * len([level for level in levels if level <= 8])
        expected_codes += len(fine_pairs) * int(16 in levels)
        if len(offsets) != expected_codes + 1 or offsets[0] != 0:
            raise ValueError("shared lookup table count mismatch")
        if np.any(np.diff(offsets) < 0) or values.shape != (offsets[-1], rank):
            raise ValueError("shared lookup values do not match offsets")

    def _validate_raw(self, metadata: Mapping[str, Any], n_features: int) -> None:
        resolutions = tuple(int(value) for value in _require(metadata, "resolutions"))
        pair_resolution = int(_require(metadata, "pair_resolution"))
        if not resolutions:
            raise ValueError("raw resolutions must not be empty")
        if int(_require(metadata, "n_features")) != n_features:
            raise ValueError("raw feature width mismatch")
        needed_resolutions = tuple(dict.fromkeys((*resolutions, pair_resolution)))
        for resolution in needed_resolutions:
            cards = _array(self.arrays, f"raw__cards_{resolution}")
            if cards.shape != (n_features,) or np.any(cards < 1):
                raise ValueError("raw state cardinalities are invalid")
            for feature in range(n_features):
                edges = _array(self.arrays, f"raw__edges_{resolution}_{feature}")
                if edges.ndim != 1 or len(edges) + 1 != int(cards[feature]):
                    raise ValueError("raw state edge/cardinality mismatch")
                if np.any(np.diff(edges) <= 0):
                    raise ValueError("raw state edges must be strictly increasing")
        pairs = _array(self.arrays, "raw__pairs").astype(np.int64, copy=False)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("raw pairs must have shape (n, 2)")
        if len(pairs) != int(_require(metadata, "n_pairs")):
            raise ValueError("raw pair count mismatch")
        if np.any((pairs < 0) | (pairs >= n_features)):
            raise ValueError("raw pair index is out of range")
        rank = self._validate_head("raw__head__", _require(metadata, "head"))
        offsets = _array(self.arrays, "raw__lookup_offsets").astype(
            np.int64, copy=False
        )
        values = _array(self.arrays, "raw__lookup_values")
        expected = len(resolutions) * n_features + len(pairs)
        if len(offsets) != expected + 1 or offsets[0] != 0:
            raise ValueError("raw lookup table count mismatch")
        if np.any(np.diff(offsets) < 0) or values.shape != (offsets[-1], rank):
            raise ValueError("raw lookup values do not match offsets")

    @staticmethod
    def _matrix(X: Any, width: int) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != int(width):
            raise ValueError("fused regression semantic input width mismatch")
        return matrix

    def _baseline_predict(self, matrix: np.ndarray) -> np.ndarray:
        metadata = self.metadata["baseline"]
        arrays = _unprefix_arrays(self.arrays, "baseline__")
        feature_idx = arrays["feature_indices"].astype(np.int64, copy=False)
        selected = matrix[:, feature_idx]
        levels = tuple(int(value) for value in metadata["levels"])
        max_main = int(metadata["max_main_level"])
        active_levels = tuple(level for level in levels if level <= max_main)
        offsets = arrays["lookup_offsets"].astype(np.int64, copy=False)
        lookup_values = arrays["lookup_values"].astype(np.float64, copy=False)
        direct_mask = arrays["direct_state_mask"].astype(bool, copy=False)
        direct_cards = arrays["direct_state_cardinalities"].astype(
            np.int64, copy=False
        )
        states: dict[int, np.ndarray] = {}
        if active_levels:
            fine = np.empty((len(matrix), len(feature_idx)), dtype=np.int64)
            for position, raw_feature in enumerate(feature_idx):
                raw_feature = int(raw_feature)
                values = matrix[:, raw_feature]
                if direct_mask[raw_feature]:
                    state = np.rint(values).astype(np.int64)
                    if np.any(~np.isfinite(values)) or np.any(
                        (state < 0) | (state >= direct_cards[raw_feature])
                    ):
                        raise ValueError(
                            f"feature {raw_feature} contains an unknown finite state"
                        )
                    fine[:, position] = state
                else:
                    fine[:, position] = np.searchsorted(
                        arrays[f"thresholds_{raw_feature}"], values, side="right"
                    )
            for level in active_levels:
                mapped = np.empty_like(fine)
                for position, raw_feature in enumerate(feature_idx):
                    mapping = arrays[f"map_{level}_{int(raw_feature)}"]
                    mapped[:, position] = mapping[fine[:, position]]
                states[level] = mapped
        prediction = np.full(
            len(matrix), float(arrays["intercept"][0]), dtype=np.float64
        )
        lookup_index = 0

        def accumulate(state: np.ndarray) -> None:
            nonlocal lookup_index, prediction
            table = lookup_values[offsets[lookup_index] : offsets[lookup_index + 1]]
            valid = (state >= 0) & (state < len(table))
            prediction[valid] += table[state[valid]]
            lookup_index += 1

        for feature in range(len(feature_idx)):
            for level in active_levels:
                accumulate(states[level][:, feature])
        pairs = arrays["pairs"].astype(np.int64, copy=False).reshape(-1, 2)
        if len(pairs):
            pair_level = max(
                level for level in active_levels if level <= min(max_main, 8)
            )
            pair_cards = arrays["pair_cardinalities"].astype(np.int64, copy=False)
            for pair_index, (left, right) in enumerate(pairs):
                code = (
                    states[pair_level][:, int(left)] * int(pair_cards[pair_index])
                    + states[pair_level][:, int(right)]
                )
                accumulate(code)
        if lookup_index != len(offsets) - 1:
            raise RuntimeError("baseline semantic lookup layout mismatch")
        linear_positions = arrays["linear_positions"].astype(np.int64, copy=False)
        if len(linear_positions):
            linear = (
                selected[:, linear_positions] - arrays["linear_mean"]
            ) / arrays["linear_scale"]
            prediction += float(arrays["linear_intercept"][0])
            prediction += linear @ arrays["linear_coef"]
        return prediction

    def _shared_states(self, matrix: np.ndarray):
        metadata = self.metadata["shared"]
        levels = tuple(int(value) for value in metadata["levels"])
        feature_idx = self.arrays["shared__feature_indices"].astype(
            np.int64, copy=False
        )
        values = matrix[:, feature_idx]
        direct_mask = self.arrays["shared__direct_state_mask"].astype(
            bool, copy=False
        )
        direct_cards = self.arrays[
            "shared__direct_state_cardinalities"
        ].astype(np.int64, copy=False)
        fine = np.empty(values.shape, dtype=np.int64)
        for feature in range(len(feature_idx)):
            column = values[:, feature]
            if direct_mask[feature]:
                state = np.rint(column).astype(np.int64)
                if np.any(~np.isfinite(column)) or np.any(
                    (state < 0) | (state >= direct_cards[feature])
                ):
                    raise ValueError(
                        "feature "
                        f"{int(feature_idx[feature])} contains an unknown finite state"
                    )
                fine[:, feature] = state
            else:
                fine[:, feature] = np.searchsorted(
                    self.arrays[f"shared__thresholds_{feature}"],
                    column,
                    side="right",
                )
        states = {}
        for level in levels:
            mapped = np.empty_like(fine)
            for feature in range(len(feature_idx)):
                mapping = self.arrays[f"shared__map_{level}_{feature}"]
                mapped[:, feature] = mapping[fine[:, feature]]
            states[level] = mapped
        return states

    def _shared_codes(self, matrix: np.ndarray) -> np.ndarray:
        metadata = self.metadata["shared"]
        levels = tuple(int(value) for value in metadata["levels"])
        coarse = levels[0]
        states = self._shared_states(matrix)
        n_features = int(metadata["n_features"])
        main_levels = [
            level for level in levels if level <= int(metadata["max_main_level"])
        ]
        pairs = self.arrays["shared__pairs"].astype(np.int64, copy=False)
        fine_pairs = self.arrays["shared__fine_pairs"].astype(
            np.int64, copy=False
        )
        pair_levels = [level for level in levels if level <= 8]
        n_columns = (
            n_features * len(main_levels)
            + len(pairs) * len(pair_levels)
            + len(fine_pairs) * int(16 in levels)
        )
        codes = np.empty((len(matrix), n_columns), dtype=np.int64)
        column = 0
        for feature in range(n_features):
            codes[:, column] = states[coarse][:, feature]
            column += 1
            for level in main_levels[1:]:
                residual = self.arrays[
                    f"shared__main_residual_{level}_{feature}"
                ]
                codes[:, column] = residual[states[level][:, feature]]
                column += 1
        fine_set = {tuple(map(int, pair)) for pair in fine_pairs}
        for pair_index, (left, right) in enumerate(pairs):
            pair = (int(left), int(right))
            for level in pair_levels:
                right_map = self.arrays[f"shared__map_{level}_{pair[1]}"]
                card_right = int(np.max(right_map, initial=0)) + 1
                joint = (
                    states[level][:, pair[0]] * card_right
                    + states[level][:, pair[1]]
                )
                if level == coarse:
                    codes[:, column] = joint
                else:
                    residual = self.arrays[
                        f"shared__pair_residual_{level}_{pair_index}"
                    ]
                    codes[:, column] = residual[joint]
                column += 1
            if 16 in levels and pair in fine_set:
                right_map = self.arrays[f"shared__map_16_{pair[1]}"]
                card_right = int(np.max(right_map, initial=0)) + 1
                joint = states[16][:, pair[0]] * card_right + states[16][:, pair[1]]
                residual = self.arrays[
                    f"shared__pair_residual_16_{pair_index}"
                ]
                codes[:, column] = residual[joint]
                column += 1
        if column != n_columns:
            raise RuntimeError("shared fused semantic code layout mismatch")
        return codes

    def _latent_from_lookup(
        self,
        codes: np.ndarray,
        offsets: np.ndarray,
        values: np.ndarray,
        threshold_coef: np.ndarray,
    ) -> np.ndarray:
        latent = np.broadcast_to(
            threshold_coef, (len(codes), len(threshold_coef))
        ).copy()
        if codes.shape[1] != len(offsets) - 1:
            raise RuntimeError("semantic code/lookup width mismatch")
        for column in range(codes.shape[1]):
            table = values[offsets[column] : offsets[column + 1]]
            state = codes[:, column]
            valid = (state >= 0) & (state < len(table))
            latent[valid] += table[state[valid]]
        return latent

    def _head_survival(
        self, prefix: str, head_metadata: Mapping[str, Any], latent: np.ndarray
    ) -> np.ndarray:
        basis = self.arrays[f"{prefix}threshold_basis"]
        intercept = float(self.arrays[f"{prefix}intercept"][0])
        return _sigmoid(latent @ basis.T + intercept)

    def _shared_survival(self, matrix: np.ndarray) -> np.ndarray:
        codes = self._shared_codes(matrix)
        offsets = self.arrays["shared__lookup_offsets"].astype(
            np.int64, copy=False
        )
        values = self.arrays["shared__lookup_values"].astype(
            np.float64, copy=False
        )
        threshold_coef = self.arrays["shared__head__threshold_coef"]
        latent = self._latent_from_lookup(codes, offsets, values, threshold_coef)
        return self._head_survival(
            "shared__head__", self.metadata["shared"]["head"], latent
        )

    def _raw_states(self, matrix: np.ndarray, resolution: int) -> np.ndarray:
        states = np.empty(matrix.shape, dtype=np.int64)
        for feature in range(matrix.shape[1]):
            states[:, feature] = np.searchsorted(
                self.arrays[f"raw__edges_{resolution}_{feature}"],
                matrix[:, feature],
                side="right",
            )
        return states

    def _raw_survival(self, matrix: np.ndarray) -> np.ndarray:
        metadata = self.metadata["raw"]
        resolutions = tuple(int(value) for value in metadata["resolutions"])
        pair_resolution = int(metadata["pair_resolution"])
        state_cache = {
            resolution: self._raw_states(matrix, resolution)
            for resolution in dict.fromkeys((*resolutions, pair_resolution))
        }
        pairs = self.arrays["raw__pairs"].astype(np.int64, copy=False)
        code_columns = []
        for resolution in resolutions:
            states = state_cache[resolution]
            for feature in range(matrix.shape[1]):
                code_columns.append(states[:, feature])
        pair_states = state_cache[pair_resolution]
        pair_cards = self.arrays[f"raw__cards_{pair_resolution}"].astype(
            np.int64, copy=False
        )
        for left, right in pairs:
            code_columns.append(
                pair_states[:, int(left)] * int(pair_cards[int(right)])
                + pair_states[:, int(right)]
            )
        codes = (
            np.column_stack(code_columns)
            if code_columns
            else np.empty((len(matrix), 0), dtype=np.int64)
        )
        offsets = self.arrays["raw__lookup_offsets"].astype(
            np.int64, copy=False
        )
        values = self.arrays["raw__lookup_values"].astype(
            np.float64, copy=False
        )
        threshold_coef = self.arrays["raw__head__threshold_coef"]
        latent = self._latent_from_lookup(codes, offsets, values, threshold_coef)
        return self._head_survival(
            "raw__head__", self.metadata["raw"]["head"], latent
        )

    def predict_survival(self, X: Any) -> np.ndarray:
        matrix = self._matrix(X, int(self.metadata["n_features_in"]))
        if self.selected_kind == "fused":
            return self._shared_survival(matrix)
        if self.selected_kind == "raw_fused":
            return self._raw_survival(matrix)
        return np.zeros(
            (len(matrix), len(self.arrays["residual_thresholds"])),
            dtype=np.float64,
        )

    def predict(self, X: Any) -> np.ndarray:
        matrix = self._matrix(X, int(self.metadata["n_features_in"]))
        baseline = self._baseline_predict(matrix)
        if self.selected_kind == "current_mean":
            return baseline
        survival = (
            self._shared_survival(matrix)
            if self.selected_kind == "fused"
            else self._raw_survival(matrix)
        )
        branch = self.metadata[
            "shared" if self.selected_kind == "fused" else "raw"
        ]
        head = branch["head"]
        thresholds = self.arrays[
            "shared__head__thresholds"
            if self.selected_kind == "fused"
            else "raw__head__thresholds"
        ]
        correction = _integrate_survival(
            float(head["lower"]),
            float(head["upper"]),
            thresholds,
            survival,
        )
        return baseline + float(self.metadata["residual_scale"]) * correction

    decision_function = predict

    def export(self, directory: str | Path) -> Path:
        """Write deterministic JSON/NPZ IR in the existing package-v3 envelope."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_npz = directory / "model.npz"
        model_json = directory / "model.json"
        ordered_arrays = {key: self.arrays[key] for key in sorted(self.arrays)}
        np.savez_compressed(model_npz, **ordered_arrays)
        model_json.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest = {
            "format": _PACKAGE_FORMAT,
            "task_type": "regression",
            "n_outputs": 1,
            "input_columns": None,
            "adapted_feature_names": None,
            "feature_indices": None,
            "model": {
                "json": _file_record(model_json),
                "npz": _file_record(model_npz),
            },
            "adapter": None,
            "config": {},
            "metadata": {
                "engine": "fused-residual-v2",
                "selected_kind": self.selected_kind,
            },
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path

    @classmethod
    def load(cls, directory: str | Path) -> "FusedRegressionSemanticProgram":
        """Load and validate a checksum-protected semantic export."""

        directory = Path(directory)
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        from ..io import verify_export

        manifest = verify_export(manifest_path)
        if (
            manifest.get("format") != _PACKAGE_FORMAT
            or manifest.get("task_type") != "regression"
        ):
            raise ValueError("fused regression requires a regression package-v3 export")
        base = manifest_path.parent
        model = manifest.get("model")
        if not isinstance(model, dict) or "json" not in model or "npz" not in model:
            raise ValueError("fused regression export is missing model artifacts")
        metadata = json.loads(
            (base / model["json"]["file"]).read_text(encoding="utf-8")
        )
        with np.load(base / model["npz"]["file"], allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        return cls(metadata=metadata, arrays=arrays)
