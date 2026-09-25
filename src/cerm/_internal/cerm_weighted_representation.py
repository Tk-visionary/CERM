from __future__ import annotations

"""Task-neutral sample-weight kernels for nested finite-state representation.

The historical unweighted representation remains authoritative. This module
adds only weighted primitives: validation/canonicalization, frequency-weighted
quantiles, and weighted finite-state mutual-information ranking.
"""

import numpy as np


def canonical_sample_weight(
    sample_weight,
    n_rows: int,
    *,
    name: str = "sample_weight",
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(weights) != int(n_rows):
        raise ValueError(
            f"{name} length mismatch: got {len(weights)}, expected {int(n_rows)}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any(weights < 0.0):
        raise ValueError(f"{name} cannot contain negative values")
    if not np.any(weights > 0.0):
        raise ValueError(f"{name} must contain at least one positive value")
    if np.array_equal(weights, np.ones(len(weights), dtype=np.float64)):
        return None
    return weights


def _numpy_linear_lerp(lower, upper, fraction):
    difference = np.subtract(upper, lower)
    result = np.asarray(np.add(lower, difference * fraction)).copy()
    use_upper = fraction >= 0.5
    if np.any(use_upper):
        result[use_upper] = upper[use_upper] - difference[use_upper] * (
            1.0 - fraction[use_upper]
        )
    return result


def _integer_frequency_quantile(sorted_values, sorted_weights, quantiles):
    rounded = np.rint(sorted_weights)
    total_float = float(np.sum(rounded))
    if not np.isfinite(total_float) or total_float > float(np.iinfo(np.int64).max):
        return None
    counts = rounded.astype(np.int64)
    total = int(counts.sum(dtype=np.int64))
    if total <= 0:
        raise ValueError("integer frequency weights must have positive mass")
    cumulative = np.cumsum(counts, dtype=np.int64)
    virtual_index = (float(total) - 1.0) * quantiles
    lower_rank = np.floor(virtual_index).astype(np.int64)
    upper_rank = np.ceil(virtual_index).astype(np.int64)
    fraction = virtual_index - lower_rank
    lower_value = sorted_values[
        np.searchsorted(cumulative, lower_rank, side="right")
    ]
    upper_value = sorted_values[
        np.searchsorted(cumulative, upper_rank, side="right")
    ]
    return _numpy_linear_lerp(lower_value, upper_value, fraction)


def frequency_weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    is_prefiltered: bool = False,
) -> np.ndarray:
    """Weighted quantile with exact integer virtual-row semantics.

    Integer weights match ``np.quantile(np.repeat(...))`` without allocating
    repeated rows. General positive real weights use midpoint weighted-ECDF
    interpolation. ``None`` calls NumPy's historical path directly.

    ``is_prefiltered`` is an internal fast path used only when the caller has
    already removed inactive/non-finite rows and validated the quantiles.
    """

    if not is_prefiltered:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        quantiles = np.asarray(quantiles, dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot compute quantiles of an empty array")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1]")
        if sample_weight is None:
            return np.asarray(np.quantile(values, quantiles), dtype=np.float64)

        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(values):
            raise ValueError("sample_weight length mismatch")
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight contains NaN or infinity")
        if np.any(weights < 0.0):
            raise ValueError("sample_weight cannot contain negative values")
        active = weights > 0.0
        if not np.any(active):
            raise ValueError("sample_weight must contain a positive value")
        active_values = values[active]
        active_weights = weights[active]
    else:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        quantiles = np.asarray(quantiles, dtype=np.float64)
        if sample_weight is None:
            return np.asarray(np.quantile(values, quantiles), dtype=np.float64)
        active_values = values
        active_weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)

    # Do not turn an ordinary source checkout into an implicit C++ build just
    # because a compiler is installed. Packaged/loaded native cores accelerate
    # this path; otherwise the historical Python implementation remains exact.
    if (
        np.all(np.isfinite(active_values))
        and np.all(np.isfinite(active_weights))
        and np.all(active_weights > 0.0)
    ):
        try:
            from .cerm_training_core_runtime import (
                TrainingCoreUnavailable,
                _prebuilt_library_path,
                load_native_training_core,
            )

            prebuilt = _prebuilt_library_path()
            native_ready = (
                load_native_training_core.cache_info().currsize > 0
                or (prebuilt is not None and prebuilt.is_file())
            )
            if native_ready:
                try:
                    return load_native_training_core().weighted_quantile(
                        active_values,
                        active_weights,
                        quantiles,
                    )
                except TrainingCoreUnavailable:
                    pass
        except ImportError:
            pass

    order = np.argsort(active_values, kind="stable")
    sorted_values = active_values[order]
    sorted_weights = active_weights[order]

    rounded = np.rint(sorted_weights)
    if np.array_equal(sorted_weights, rounded):
        exact = _integer_frequency_quantile(
            sorted_values, sorted_weights, quantiles
        )
        if exact is not None:
            return np.asarray(exact, dtype=np.float64)

    scale = float(np.max(sorted_weights))
    stable_weights = sorted_weights / scale
    cumulative = np.cumsum(stable_weights, dtype=np.float64)
    positions = (
        cumulative - 0.5 * stable_weights
    ) / float(cumulative[-1])
    return np.asarray(
        np.interp(
            quantiles,
            positions,
            sorted_values,
            left=float(sorted_values[0]),
            right=float(sorted_values[-1]),
        ),
        dtype=np.float64,
    )


class WeightedNestedQuantileEncoder:
    """Compatibility factory returning the canonical weighted-aware encoder."""

    def __new__(cls, *args, **kwargs):
        # Lazy import avoids a cycle: cerm_hierarchical_residual imports the
        # weighted kernels above before defining its facade encoder class.
        from .cerm_hierarchical_residual import NestedQuantileEncoder

        return NestedQuantileEncoder(*args, **kwargs)


def _stable_weight_mass(weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if np.isfinite(total):
        return weights
    maximum = float(np.max(weights, initial=0.0))
    return weights if maximum <= 0.0 else weights / maximum


def _active_weighted_rows(states, y, sample_weight):
    states64 = np.asarray(states, dtype=np.int64)
    weights = canonical_sample_weight(sample_weight, len(states64))
    if weights is None:
        return states64, np.asarray(y), None
    active = weights > 0.0
    states64 = states64[active]
    labels = np.asarray(y)[active]
    active_weight = weights[active]
    return (
        states64,
        labels,
        canonical_sample_weight(active_weight, len(active_weight)),
    )


def weighted_finite_mi_validated(
    code,
    target,
    cardinality,
    n_classes,
    sample_weight,
):
    if sample_weight is None:
        from ..shared_multitask import _finite_mi_validated

        return _finite_mi_validated(code, target, cardinality, n_classes)
    weights = canonical_sample_weight(sample_weight, len(code))
    if weights is None:
        from ..shared_multitask import _finite_mi_validated

        return _finite_mi_validated(code, target, cardinality, n_classes)
    stable_weight = _stable_weight_mass(weights)
    joint = np.bincount(
        code * int(n_classes) + target,
        weights=stable_weight,
        minlength=int(cardinality) * int(n_classes),
    ).reshape(int(cardinality), int(n_classes))
    total = float(joint.sum())
    if total <= 0.0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denominator = px @ py
    mask = pxy > 0.0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / denominator[mask])))


def aggregate_feature_scores(
    states,
    y,
    task_type,
    sample_weight=None,
):
    """Weighted counterpart of shared ``_aggregate_feature_scores``."""

    if sample_weight is None:
        from ..shared_multitask import _aggregate_feature_scores

        return _aggregate_feature_scores(states, y, task_type)
    states64, labels, weights = _active_weighted_rows(states, y, sample_weight)
    if weights is None:
        from ..shared_multitask import _aggregate_feature_scores

        return _aggregate_feature_scores(states64, labels, task_type)
    if np.any(states64 < 0):
        raise ValueError("finite MI requires non-negative integer values")
    cards = states64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1

    if task_type == "multiclass":
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(target) != len(states64) or np.any(target < 0):
            raise ValueError("finite MI target is invalid")
        classes = int(target.max(initial=0)) + 1
        return np.fromiter(
            (
                weighted_finite_mi_validated(
                    states64[:, feature],
                    target,
                    int(cards[feature]),
                    classes,
                    weights,
                )
                for feature in range(states64.shape[1])
            ),
            dtype=np.float64,
            count=states64.shape[1],
        )

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or len(labels) != len(states64):
        raise ValueError("multilabel target must be an aligned 2D array")
    values = np.zeros(states64.shape[1], dtype=np.float64)
    active_outputs = 0
    for output in range(labels.shape[1]):
        target = np.asarray(labels[:, output], dtype=np.int64).reshape(-1)
        if np.unique(target).size < 2:
            continue
        if np.any((target < 0) | (target > 1)):
            raise ValueError("multilabel finite MI requires binary 0/1 columns")
        values += np.fromiter(
            (
                weighted_finite_mi_validated(
                    states64[:, feature],
                    target,
                    int(cards[feature]),
                    2,
                    weights,
                )
                for feature in range(states64.shape[1])
            ),
            dtype=np.float64,
            count=states64.shape[1],
        )
        active_outputs += 1
    return values / max(active_outputs, 1)


def rank_pairs(
    states,
    y,
    task_type,
    limit,
    feature_limit,
    aggregation,
    sample_weight=None,
):
    """Weighted counterpart of shared ``_rank_pairs`` with historical ties."""

    if sample_weight is None:
        from ..shared_multitask import _rank_pairs

        return _rank_pairs(states, y, task_type, limit, feature_limit, aggregation)
    states64, labels, weights = _active_weighted_rows(states, y, sample_weight)
    if weights is None:
        from ..shared_multitask import _rank_pairs

        return _rank_pairs(
            states64, labels, task_type, limit, feature_limit, aggregation
        )

    dimension = min(states64.shape[1], int(feature_limit))
    if dimension < 2 or int(limit) <= 0:
        return []
    states64 = np.asarray(states64[:, :dimension], dtype=np.int64, order="F")
    if np.any(states64 < 0):
        raise ValueError("pair MI requires non-negative states")
    cards = states64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    ranked = []
    joint = np.empty(len(states64), dtype=np.int64)

    if task_type == "multiclass" and aggregation == "joint":
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(target) != len(states64) or np.any(target < 0):
            raise ValueError("pair MI target is invalid")
        classes = int(target.max(initial=0)) + 1
        marginal = np.fromiter(
            (
                weighted_finite_mi_validated(
                    states64[:, feature],
                    target,
                    int(cards[feature]),
                    classes,
                    weights,
                )
                for feature in range(dimension)
            ),
            dtype=np.float64,
            count=dimension,
        )
        for left in range(dimension):
            for right in range(left + 1, dimension):
                np.multiply(states64[:, left], int(cards[right]), out=joint)
                joint += states64[:, right]
                score = weighted_finite_mi_validated(
                    joint,
                    target,
                    int(cards[left]) * int(cards[right]),
                    classes,
                    weights,
                )
                ranked.append(
                    (
                        score - max(marginal[left], marginal[right]),
                        score,
                        left,
                        right,
                    )
                )
    else:
        if task_type == "multiclass":
            multiclass = np.asarray(labels, dtype=np.int64).reshape(-1)
            outputs = [
                (multiclass == cls).astype(np.int64)
                for cls in np.unique(multiclass)
            ]
        else:
            multilabel = np.asarray(labels, dtype=np.int64)
            if multilabel.ndim != 2 or len(multilabel) != len(states64):
                raise ValueError("multilabel target must be an aligned 2D array")
            outputs = [
                multilabel[:, output]
                for output in range(multilabel.shape[1])
                if np.unique(multilabel[:, output]).size == 2
            ]
        marginal_by_output = [
            np.fromiter(
                (
                    weighted_finite_mi_validated(
                        states64[:, feature],
                        target,
                        int(cards[feature]),
                        2,
                        weights,
                    )
                    for feature in range(dimension)
                ),
                dtype=np.float64,
                count=dimension,
            )
            for target in outputs
        ]
        for left in range(dimension):
            for right in range(left + 1, dimension):
                np.multiply(states64[:, left], int(cards[right]), out=joint)
                joint += states64[:, right]
                pair_cardinality = int(cards[left]) * int(cards[right])
                gains = []
                raw_scores = []
                for target, marginal in zip(outputs, marginal_by_output):
                    score = weighted_finite_mi_validated(
                        joint, target, pair_cardinality, 2, weights
                    )
                    raw_scores.append(score)
                    gains.append(score - max(marginal[left], marginal[right]))
                if not gains:
                    aggregate = raw_score = 0.0
                elif aggregation == "max":
                    aggregate = float(np.max(gains))
                    raw_score = float(np.max(raw_scores))
                elif aggregation == "mean_max":
                    aggregate = 0.5 * float(np.mean(gains)) + 0.5 * float(
                        np.max(gains)
                    )
                    raw_score = 0.5 * float(np.mean(raw_scores)) + 0.5 * float(
                        np.max(raw_scores)
                    )
                else:
                    aggregate = float(np.mean(gains))
                    raw_score = float(np.mean(raw_scores))
                ranked.append((aggregate, raw_score, left, right))

    ranked.sort(reverse=True)
    return [(left, right) for _, _, left, right in ranked[: int(limit)]]
