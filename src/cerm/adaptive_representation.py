"""Bounded adaptive basis support for shared multiclass finite-state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualBasisDescriptor:
    """Task-neutral Program IR descriptor for one residual basis operator.

    Version 0.11 initially exposes only ``class_state_delta``.  The remaining
    fields make the node explicit and leave native lowering independent of the
    training implementation.
    """

    raw0: int
    level: int
    state: int
    target_class: int
    candidate_score: float = 0.0
    kind: str = "class_state_delta"
    raw1: int = -1
    raw2: int = -1
    threshold: float = float("nan")
    joint_values: tuple[int, ...] = ()
    hierarchy: str = "none"

    def __post_init__(self):
        if self.kind != "class_state_delta":
            raise ValueError("0.11 adaptive IR supports class_state_delta only")
        if min(int(self.raw0), int(self.level), int(self.state), int(self.target_class)) < 0:
            raise ValueError("class-state delta indices must be non-negative")


def _softmax(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    shifted = score - np.max(score, axis=1, keepdims=True)
    values = np.exp(np.clip(shifted, -50.0, 50.0))
    return values / values.sum(axis=1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    result = np.zeros((len(y), n_classes), dtype=np.float64)
    result[np.arange(len(y)), np.asarray(y, dtype=np.int64)] = 1.0
    return result


def _state_gain(
    state: np.ndarray,
    residual: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_rows: int,
    l2: float,
) -> float:
    result = 0.0
    for value, count in zip(*np.unique(state, return_counts=True)):
        if int(count) < int(minimum_rows):
            continue
        mask = state == value
        gradient = residual[mask].sum(axis=0)
        hessian = (probability[mask] * (1.0 - probability[mask])).sum(axis=0)
        result += 0.5 * float(np.sum(gradient * gradient / (hessian + float(l2))))
    return result


def generate_class_specific_deltas(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    maximum: int,
    maximum_per_class: int = 3,
    maximum_screened_features: int = 24,
    minimum_state_rows: int = 12,
    screening_l2: float = 10.0,
) -> list[ResidualBasisDescriptor]:
    """Rank class-specific basis support from training residuals only."""

    if maximum <= 0:
        return []
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    states = model.encoder_.transform(X)
    probability = _softmax(model._decision_function_from_states(states))
    residual = _one_hot(y, probability.shape[1]) - probability
    level = max(model.levels)
    feature_scores = []
    for raw in range(X.shape[1]):
        feature_scores.append((
            _state_gain(
                states[level][:, raw], residual, probability,
                minimum_rows=minimum_state_rows, l2=screening_l2,
            ),
            raw,
        ))
    feature_scores.sort(key=lambda row: (-row[0], row[1]))
    screened = [raw for _, raw in feature_scores[: int(maximum_screened_features)]]
    candidates = []
    for raw in screened:
        column = states[level][:, raw]
        for state, count in zip(*np.unique(column, return_counts=True)):
            if int(count) < int(minimum_state_rows):
                continue
            mask = column == state
            for target_class in range(probability.shape[1]):
                gradient = float(residual[mask, target_class].sum())
                hessian = float(
                    (
                        probability[mask, target_class]
                        * (1.0 - probability[mask, target_class])
                    ).sum()
                )
                score = 0.5 * gradient * gradient / (hessian + float(screening_l2))
                candidates.append(ResidualBasisDescriptor(
                    raw0=int(raw),
                    level=int(level),
                    state=int(state),
                    target_class=int(target_class),
                    candidate_score=float(score),
                ))
    candidates.sort(
        key=lambda item: (
            -item.candidate_score,
            item.target_class,
            item.raw0,
            item.state,
        )
    )
    per_class = np.zeros(probability.shape[1], dtype=np.int32)
    chosen = []
    for candidate in candidates:
        if per_class[candidate.target_class] >= int(maximum_per_class):
            continue
        chosen.append(candidate)
        per_class[candidate.target_class] += 1
        if len(chosen) >= int(maximum):
            break
    return chosen


def descriptor_code(
    descriptor: ResidualBasisDescriptor,
    states: dict[int, np.ndarray],
) -> np.ndarray:
    return (
        states[int(descriptor.level)][:, int(descriptor.raw0)]
        == int(descriptor.state)
    ).astype(np.int32)


def _fit_one_delta(
    score: np.ndarray,
    y: np.ndarray,
    code: np.ndarray,
    target_class: int,
    *,
    l2: float,
) -> np.ndarray:
    active = code == 1
    table = np.zeros((2, score.shape[1]), dtype=np.float64)
    if active.sum() < 2:
        return table
    target = (np.asarray(y) == int(target_class)).astype(np.float64)
    alpha = 0.0
    for _ in range(25):
        shifted = score.copy()
        shifted[active, target_class] += alpha
        probability = _softmax(shifted)
        gradient = float(
            (probability[active, target_class] - target[active]).sum() + l2 * alpha
        )
        hessian = float(
            (
                probability[active, target_class]
                * (1.0 - probability[active, target_class])
            ).sum()
            + l2
        )
        step = gradient / max(hessian, 1e-12)
        alpha = float(np.clip(alpha - step, -3.0, 3.0))
        if abs(step) < 1e-8:
            break
    table[1, target_class] = alpha
    return table


def fit_class_delta_tables(
    model,
    descriptors: list[ResidualBasisDescriptor] | tuple[ResidualBasisDescriptor, ...],
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 20.0,
) -> tuple[list[np.ndarray], list[dict]]:
    """Fit sparse masked corrections while leaving the base head fixed."""

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    states = model.encoder_.transform(X)
    score = np.asarray(
        model._decision_function_from_states(states), dtype=np.float64
    )
    lookups = []
    diagnostics = []
    for descriptor in descriptors:
        code = descriptor_code(descriptor, states)
        table = _fit_one_delta(
            score,
            y,
            code,
            descriptor.target_class,
            l2=float(l2),
        )
        score += table[code]
        lookups.append(table)
        diagnostics.append({
            "raw_feature": int(descriptor.raw0),
            "state": int(descriptor.state),
            "target_class": int(descriptor.target_class),
            "candidate_score": float(descriptor.candidate_score),
            "coefficient": float(table[1, descriptor.target_class]),
            "active_rows": int(np.count_nonzero(code)),
        })
    return lookups, diagnostics


def decision_with_class_deltas(
    model,
    X: np.ndarray,
    descriptors,
    lookups,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    states = model.encoder_.transform(X)
    score = np.asarray(
        model._decision_function_from_states(states), dtype=np.float64
    )
    if not descriptors:
        return score
    for descriptor, table in zip(descriptors, lookups):
        code = descriptor_code(descriptor, states)
        score += np.asarray(table, dtype=np.float64)[code]
    return score



def class_delta_prefix_probabilities(
    model,
    X: np.ndarray,
    descriptors,
    lookups,
    budgets,
) -> dict[int, np.ndarray]:
    """Evaluate all requested descriptor prefixes from one exact state pass."""
    budgets = tuple(sorted({int(value) for value in budgets if int(value) >= 0}))
    if not budgets:
        return {}
    X = np.asarray(X, dtype=np.float64)
    states = model.encoder_.transform(X)
    score = np.asarray(
        model._decision_function_from_states(states), dtype=np.float64
    ).copy()
    result: dict[int, np.ndarray] = {}
    if 0 in budgets:
        result[0] = _softmax(score)
    targets = set(budgets)
    for index, (descriptor, table) in enumerate(zip(descriptors, lookups), start=1):
        code = descriptor_code(descriptor, states)
        score += np.asarray(table, dtype=np.float64)[code]
        if index in targets:
            result[index] = _softmax(score)
    return result

def budget_prefixes(maximum: int) -> tuple[int, ...]:
    maximum = int(maximum)
    if maximum <= 0:
        return (0,)
    first = max(1, maximum // 3)
    second = max(first + 1, (2 * maximum) // 3)
    return tuple(dict.fromkeys((0, min(first, maximum), min(second, maximum), maximum)))


def extra_model_bytes(descriptors, lookups) -> int:
    return int(
        sum(64 + 8 * len(descriptor.joint_values) + table.nbytes
            for descriptor, table in zip(descriptors, lookups))
    )
