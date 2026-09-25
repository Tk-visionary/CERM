from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from ._internal.cerm_hierarchical_residual import (
    NestedQuantileEncoder,
    _pair_parent_map,
    _parent_labels_from_fine,
    _residual_state_map,
)
from ._internal.cerm_state_design import ReferenceStateEncoder
from ._internal.cerm_typed_quotient_adapters_v4 import export_typed_adapter_ir
from .adaptive_representation import (
    ResidualBasisDescriptor,
    budget_prefixes,
    class_delta_prefix_probabilities,
    extra_model_bytes,
    fit_class_delta_tables,
    generate_class_specific_deltas,
)
from .training_graph import EncodedColumnBank, fit_binary_logistic_exact
from .validation import dense_numeric_matrix, validate_dataframe_schema


@dataclass(frozen=True)
class SharedStructureConfig:
    n_pairs: int
    n_fine_pairs: int
    max_main_level: int
    C: float
    pair_aggregation: str = "joint"
    multiclass_objective: str = "ovr"
    multinomial_weight: float = 0.0


@dataclass
class _LinearMulticlassHead:
    coef_: np.ndarray
    intercept_: np.ndarray


def _finite_mi_validated(
    code: np.ndarray,
    target: np.ndarray,
    cardinality: int,
    n_classes: int,
) -> float:
    """Inner finite-MI kernel for aligned non-negative integer arrays."""
    joint = np.bincount(
        code * int(n_classes) + target,
        minlength=int(cardinality) * int(n_classes),
    ).reshape(int(cardinality), int(n_classes))
    n = float(len(code))
    if n <= 0:
        return 0.0
    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))


def _finite_mi(code: np.ndarray, target: np.ndarray, n_classes: int | None = None) -> float:
    code = np.asarray(code, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if len(code) != len(target):
        raise ValueError("code and target length mismatch")
    if np.any(code < 0) or np.any(target < 0):
        raise ValueError("finite MI requires non-negative integer values")
    card = int(code.max(initial=0)) + 1
    classes = int(n_classes or (target.max(initial=0) + 1))
    return _finite_mi_validated(code, target, card, classes)


def _aggregate_feature_scores(states: np.ndarray, y: np.ndarray, task_type: str) -> np.ndarray:
    states64 = np.asarray(states, dtype=np.int64)
    if np.any(states64 < 0):
        raise ValueError("finite MI requires non-negative integer values")
    cards = states64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    if task_type == "multiclass":
        encoded = np.asarray(y, dtype=np.int64).reshape(-1)
        if len(encoded) != len(states64) or np.any(encoded < 0):
            raise ValueError("finite MI target is invalid")
        classes = int(encoded.max(initial=0)) + 1
        return np.fromiter(
            (
                _finite_mi_validated(
                    states64[:, j], encoded, int(cards[j]), classes
                )
                for j in range(states64.shape[1])
            ),
            dtype=np.float64,
            count=states64.shape[1],
        )
    labels = np.asarray(y, dtype=np.int64)
    values = np.zeros(states64.shape[1], dtype=np.float64)
    active = 0
    for column in range(labels.shape[1]):
        target = np.asarray(labels[:, column], dtype=np.int64).reshape(-1)
        if np.unique(target).size < 2:
            continue
        values += np.fromiter(
            (
                _finite_mi_validated(
                    states64[:, j], target, int(cards[j]), 2
                )
                for j in range(states64.shape[1])
            ),
            dtype=np.float64,
            count=states64.shape[1],
        )
        active += 1
    return values / max(active, 1)


def _rank_pairs(
    states: np.ndarray,
    y: np.ndarray,
    task_type: str,
    limit: int,
    feature_limit: int,
    aggregation: str,
):
    d = min(states.shape[1], int(feature_limit))
    if d < 2 or limit <= 0:
        return []
    states64 = np.asarray(states[:, :d], dtype=np.int64, order="F")
    if np.any(states64 < 0):
        raise ValueError("pair MI requires non-negative states")
    cards = states64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    ranked = []
    joint = np.empty(len(states64), dtype=np.int64)
    if task_type == "multiclass" and aggregation == "joint":
        target = np.asarray(y, dtype=np.int64).reshape(-1)
        if len(target) != len(states64) or np.any(target < 0):
            raise ValueError("pair MI target is invalid")
        classes = int(np.max(target, initial=0)) + 1
        marginal = np.fromiter(
            (
                _finite_mi_validated(
                    states64[:, j], target, int(cards[j]), classes
                )
                for j in range(d)
            ),
            dtype=np.float64,
            count=d,
        )
        for j in range(d):
            left = states64[:, j]
            for k in range(j + 1, d):
                np.multiply(left, int(cards[k]), out=joint)
                joint += states64[:, k]
                score = _finite_mi_validated(
                    joint, target, int(cards[j]) * int(cards[k]), classes
                )
                ranked.append((score - max(marginal[j], marginal[k]), score, j, k))
    else:
        outputs = []
        if task_type == "multiclass":
            labels = np.asarray(y, dtype=np.int64).reshape(-1)
            for cls in np.unique(labels):
                outputs.append((labels == cls).astype(np.int64))
        else:
            labels = np.asarray(y, dtype=np.int64)
            outputs = [
                labels[:, output]
                for output in range(labels.shape[1])
                if np.unique(labels[:, output]).size == 2
            ]
        marginal_by_output = [
            np.fromiter(
                (
                    _finite_mi_validated(
                        states64[:, j], target, int(cards[j]), 2
                    )
                    for j in range(d)
                ),
                dtype=np.float64,
                count=d,
            )
            for target in outputs
        ]
        for j in range(d):
            left = states64[:, j]
            for k in range(j + 1, d):
                np.multiply(left, int(cards[k]), out=joint)
                joint += states64[:, k]
                values = []
                raw = []
                pair_card = int(cards[j]) * int(cards[k])
                for target, marginal in zip(outputs, marginal_by_output):
                    score = _finite_mi_validated(joint, target, pair_card, 2)
                    raw.append(score)
                    values.append(score - max(marginal[j], marginal[k]))
                if not values:
                    aggregate = score = 0.0
                elif aggregation == "max":
                    aggregate = float(np.max(values)); score = float(np.max(raw))
                elif aggregation == "mean_max":
                    aggregate = 0.5 * float(np.mean(values)) + 0.5 * float(np.max(values))
                    score = 0.5 * float(np.mean(raw)) + 0.5 * float(np.max(raw))
                else:
                    aggregate = float(np.mean(values)); score = float(np.mean(raw))
                ranked.append((aggregate, score, j, k))
    ranked.sort(reverse=True)
    return [(j, k) for _, _, j, k in ranked[: int(limit)]]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    total = exp.sum(axis=1, keepdims=True)
    return np.divide(exp, total, out=np.full_like(exp, 1.0 / exp.shape[1]), where=total > 0)


class SharedFiniteStateModel:
    """One finite-state representation with vector-valued output heads."""

    _BLEND_SELECTION_PENALTY = 0.005
    _NON_OVR_SELECTION_MARGIN = 0.005
    _SELECTION_PARSIMONY_TOLERANCE = 0.0015
    _REPRESENTATION_VALIDATION_SEED = 20260817

    def __init__(
        self,
        *,
        task_type: str,
        max_features: int = 64,
        pair_feature_limit: int = 24,
        max_bins: int = 16,
        random_state: int = 20260803,
        feature_kinds: Sequence[str] | None = None,
        feature_cardinalities: Sequence[int | None] | None = None,
        threshold: float | Sequence[float] = 0.5,
        max_pairs: int = 30,
        fixed_C: float | None = None,
        search_profile: str = "full_exact",
        selection_subsample: float = 1.0,
        n_jobs: int | None = 1,
        output_aggregation: str | None = None,
        multiclass_objective: str = "ovr",
        representation_strategy: str = "baseline",
        class_specific_budget: int = 12,
    ):
        if task_type not in {"multiclass", "multilabel"}:
            raise ValueError("task_type must be multiclass or multilabel")
        self.task_type = task_type
        self.max_features = int(max_features)
        self.pair_feature_limit = int(pair_feature_limit)
        self.max_bins = int(max_bins)
        self.levels = tuple(level for level in (4, 8, 16) if level <= self.max_bins)
        self.ranking_level = 8 if self.max_bins >= 8 else 4
        self.random_state = int(random_state)
        self.feature_kinds = feature_kinds
        self.feature_cardinalities = feature_cardinalities
        self.threshold = threshold
        self.max_pairs = max(0, int(max_pairs))
        self.fixed_C = None if fixed_C is None else float(fixed_C)
        self.search_profile = str(search_profile)
        self.selection_subsample = float(selection_subsample)
        self.n_jobs = n_jobs
        self.multiclass_objective = str(multiclass_objective)
        if self.multiclass_objective not in {"auto", "ovr", "multinomial"}:
            raise ValueError("multiclass_objective must be auto, ovr, or multinomial")
        if self.task_type != "multiclass" and self.multiclass_objective != "ovr":
            raise ValueError("multiclass_objective applies only to multiclass models")
        self.representation_strategy = str(representation_strategy)
        if self.representation_strategy not in {"baseline", "adaptive"}:
            raise ValueError("representation_strategy must be baseline or adaptive")
        self.class_specific_budget = int(class_specific_budget)
        if self.class_specific_budget < 0:
            raise ValueError("class_specific_budget must be non-negative")
        if self.task_type != "multiclass" and self.representation_strategy != "baseline":
            raise ValueError("adaptive representation applies only to multiclass models")
        self.output_aggregation = (
            ("joint" if task_type == "multiclass" else "mean")
            if output_aggregation is None else str(output_aggregation)
        )
        if self.output_aggregation not in {"auto", "joint", "mean", "max", "mean_max"}:
            raise ValueError("invalid output_aggregation")

    def _objective_candidates(self) -> tuple[str, ...]:
        if self.task_type != "multiclass":
            return ("ovr",)
        if self.multiclass_objective == "auto":
            return ("blend",)
        return (self.multiclass_objective,)

    def _aggregation_candidates(self) -> tuple[str, ...]:
        if self.output_aggregation != "auto":
            return (self.output_aggregation,)
        if self.task_type == "multiclass":
            return ("joint", "mean")
        return ("mean", "max")

    @classmethod
    def _selection_loss(cls, row) -> float:
        loss, _, config = row
        penalty = (
            cls._BLEND_SELECTION_PENALTY
            if config.multiclass_objective == "blend"
            else 0.0
        )
        return float(loss + penalty)

    @classmethod
    def _select_validation_result(cls, results):
        best_selection_loss = min(cls._selection_loss(row) for row in results)
        eligible = [
            row
            for row in results
            if cls._selection_loss(row)
            <= best_selection_loss + cls._SELECTION_PARSIMONY_TOLERANCE
        ]
        return min(
            eligible,
            key=lambda row: (row[1], cls._selection_loss(row), row[0]),
        )

    @classmethod
    def _guard_auto_validation_result(cls, results, selected_row):
        """Apply the development-frozen non-OVR validation margin.

        The guard has no access to task identity or test outcomes.  It compares
        candidates from the existing internal validation split and, when the
        selected non-OVR head has insufficient advantage, uses the same
        parsimonious selection rule restricted to OVR candidates.
        """
        ovr_results = [
            row
            for row in results
            if row[2].multiclass_objective == "ovr"
        ]
        if not ovr_results:
            raise RuntimeError("auto multiclass search produced no OVR candidate")
        best_ovr_loss = min(cls._selection_loss(row) for row in ovr_results)
        pre_guard_objective = selected_row[2].multiclass_objective
        advantage = (
            0.0
            if pre_guard_objective == "ovr"
            else best_ovr_loss - cls._selection_loss(selected_row)
        )
        applied = bool(
            pre_guard_objective != "ovr"
            and advantage < cls._NON_OVR_SELECTION_MARGIN
        )
        if applied:
            selected_row = cls._select_validation_result(ovr_results)
        return selected_row, float(best_ovr_loss), float(advantage), applied

    def _new_encoder(self):
        return NestedQuantileEncoder(
            max_bins=self.max_bins,
            levels=self.levels,
            feature_kinds=self.feature_kinds,
            feature_cardinalities=self.feature_cardinalities,
        )

    def _configs(self):
        C_values = [self.fixed_C] if self.fixed_C is not None else [0.2, 1.0]
        main_levels = [min(8, self.max_bins), self.max_bins]
        pair_prefixes = []
        if self.max_pairs > 0:
            base = (10, 30) if self.search_profile in {"practical", "aggressive"} else (10, 20, 30)
            pair_prefixes = list(dict.fromkeys(min(self.max_pairs, value) for value in base if min(self.max_pairs, value) > 0))
            if self.max_pairs not in pair_prefixes:
                pair_prefixes.append(self.max_pairs)
        aggregations = self._aggregation_candidates()
        objectives = self._objective_candidates()
        configs = []
        for objective in objectives:
            # Main-only candidates do not depend on pair ranking. Keep one copy
            # per head objective because their statistical objectives differ.
            main_aggregation = aggregations[0]
            for level in dict.fromkeys(main_levels):
                for C in C_values:
                    configs.append(SharedStructureConfig(
                        0, 0, int(level), float(C), main_aggregation, objective,
                        0.0 if objective == "ovr" else 1.0 if objective == "multinomial" else 0.5,
                    ))
            for aggregation in aggregations:
                for n_pairs in pair_prefixes:
                    for C in C_values:
                        configs.append(SharedStructureConfig(
                            int(n_pairs), 0, self.max_bins, float(C), aggregation, objective,
                            0.0 if objective == "ovr" else 1.0 if objective == "multinomial" else 0.5
                        ))
                if self.max_bins == 16 and self.max_pairs > 0 and self.search_profile == "full_exact":
                    fine = min(5, self.max_pairs)
                    for C in C_values:
                        configs.append(SharedStructureConfig(
                            self.max_pairs, fine, 16, float(C), aggregation, objective,
                            0.0 if objective == "ovr" else 1.0 if objective == "multinomial" else 0.5
                        ))
        return list(dict.fromkeys(configs))

    def _selection_indices(self, y: np.ndarray):
        if self.selection_subsample >= 1.0:
            return np.arange(len(y), dtype=np.int64)
        rng = np.random.default_rng(self.random_state + 49979687)
        if self.task_type == "multiclass":
            selected = []
            for cls in np.unique(y):
                idx = np.flatnonzero(y == cls)
                keep = min(len(idx), max(8, int(np.ceil(self.selection_subsample * len(idx)))))
                if keep < len(idx):
                    idx = rng.choice(idx, size=keep, replace=False)
                selected.append(np.asarray(idx, dtype=np.int64))
            indices = np.concatenate(selected)
        else:
            keep = min(len(y), max(16, int(np.ceil(self.selection_subsample * len(y)))))
            indices = rng.choice(len(y), size=keep, replace=False) if keep < len(y) else np.arange(len(y))
        rng.shuffle(indices)
        return np.asarray(indices, dtype=np.int64)

    def _prepare_residual_maps(self):
        self.main_residuals_ = {}
        for level, parent in zip(self.levels[1:], self.levels[:-1]):
            maps = []
            for raw_j in self.feature_idx_:
                child_map = self.encoder_.maps_[level][raw_j]
                parent_map = self.encoder_.maps_[parent][raw_j]
                child_card = int(self.encoder_.cardinalities_[level][raw_j])
                labels = _parent_labels_from_fine(child_map, parent_map, child_card)
                maps.append(_residual_state_map(labels))
            self.main_residuals_[level] = maps
        self.pair_residuals_ = {level: {} for level in self.levels[1:]}
        fine_pairs = set(self.fine_pairs_)
        for j, k in self.pairs_:
            raw_j, raw_k = int(self.feature_idx_[j]), int(self.feature_idx_[k])
            for level, parent in zip(self.levels[1:], self.levels[:-1]):
                if level > 8 and (j, k) not in fine_pairs:
                    continue
                cj = int(self.encoder_.cardinalities_[level][raw_j])
                ck = int(self.encoder_.cardinalities_[level][raw_k])
                parent_ck = int(self.encoder_.cardinalities_[parent][raw_k])
                pj = _parent_labels_from_fine(self.encoder_.maps_[level][raw_j], self.encoder_.maps_[parent][raw_j], cj)
                pk = _parent_labels_from_fine(self.encoder_.maps_[level][raw_k], self.encoder_.maps_[parent][raw_k], ck)
                parent_map = _pair_parent_map(cj, ck, pj, pk, parent_ck)
                self.pair_residuals_[level][(j, k)] = _residual_state_map(parent_map)

    def _build_codes(self, selected_states: dict[int, np.ndarray], config: SharedStructureConfig) -> np.ndarray:
        coarse = self.levels[0]
        main_levels = [level for level in self.levels if level <= config.max_main_level]
        pair_levels = [level for level in self.levels if level <= min(self.max_bins, 8)]
        n_columns = selected_states[coarse].shape[1] * len(main_levels)
        n_columns += len(self.pairs_) * len(pair_levels)
        n_columns += len(self.fine_pairs_) * int(16 in self.levels)
        codes = np.empty((len(selected_states[coarse]), max(n_columns, 1)), dtype=np.int32)
        if n_columns == 0:
            codes[:, 0] = 0
            return codes
        col = 0
        for j in range(selected_states[coarse].shape[1]):
            codes[:, col] = selected_states[coarse][:, j]
            col += 1
            for level in main_levels[1:]:
                codes[:, col] = self.main_residuals_[level][j][selected_states[level][:, j]]
                col += 1
        fine_pairs = set(self.fine_pairs_)
        joint = np.empty(len(selected_states[coarse]), dtype=np.int64)
        for j, k in self.pairs_:
            raw_j, raw_k = int(self.feature_idx_[j]), int(self.feature_idx_[k])
            for level in pair_levels:
                card_k = int(self.encoder_.cardinalities_[level][raw_k])
                np.multiply(
                    selected_states[level][:, j], card_k,
                    out=joint, casting="unsafe",
                )
                joint += selected_states[level][:, k]
                if level == coarse:
                    codes[:, col] = joint
                else:
                    codes[:, col] = self.pair_residuals_[level][(j, k)][joint]
                col += 1
            if 16 in self.levels and (j, k) in fine_pairs:
                card_k = int(self.encoder_.cardinalities_[16][raw_k])
                np.multiply(
                    selected_states[16][:, j], card_k,
                    out=joint, casting="unsafe",
                )
                joint += selected_states[16][:, k]
                codes[:, col] = self.pair_residuals_[16][(j, k)][joint]
                col += 1
        if col != n_columns:
            raise RuntimeError("shared finite-state code layout mismatch")
        return codes

    def _parallel_head_results(self, functions):
        jobs = effective_n_jobs(self.n_jobs)
        if jobs == 1 or len(functions) <= 1:
            return [function() for function in functions]
        return Parallel(n_jobs=jobs, prefer="threads")(
            delayed(function)() for function in functions
        )

    @staticmethod
    def _fused_head_scores(fitted, design: sparse.csr_matrix):
        n_outputs = len(fitted)
        n_features = int(design.shape[1])
        coef = np.zeros((n_outputs, n_features), dtype=np.float64)
        intercept = np.zeros(n_outputs, dtype=np.float64)
        constants = np.full(n_outputs, np.nan, dtype=np.float64)
        for output, model in enumerate(fitted):
            if isinstance(model, float):
                constants[output] = float(model)
            else:
                coef[output] = np.asarray(model.coef_[0], dtype=np.float64)
                intercept[output] = float(model.intercept_[0])
        scores = np.asarray(design @ coef.T, dtype=np.float64)
        scores += intercept
        return scores, constants

    def _probability_from_fitted(self, fitted, design: sparse.csr_matrix):
        if self.task_type == "multiclass" and not isinstance(fitted, (list, tuple)):
            scores = np.asarray(design @ np.asarray(fitted.coef_, dtype=np.float64).T, dtype=np.float64)
            scores += np.asarray(fitted.intercept_, dtype=np.float64)
            return _softmax(scores)
        scores, constants = self._fused_head_scores(fitted, design)
        if self.task_type == "multiclass":
            return _softmax(scores)
        probability = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
        for output, value in enumerate(constants):
            if np.isfinite(value):
                probability[:, output] = value
        return probability

    @staticmethod
    def _multiclass_linear_head(fitted) -> _LinearMulticlassHead:
        if isinstance(fitted, (list, tuple)):
            coef = np.vstack([np.asarray(model.coef_[0], dtype=np.float64) for model in fitted])
            intercept = np.asarray(
                [float(model.intercept_[0]) for model in fitted], dtype=np.float64
            )
        else:
            coef = np.asarray(fitted.coef_, dtype=np.float64)
            intercept = np.asarray(fitted.intercept_, dtype=np.float64)
        return _LinearMulticlassHead(coef_=coef, intercept_=intercept)

    @classmethod
    def _blend_multiclass_heads(cls, ovr_fitted, multinomial_fitted, weight: float):
        weight = float(np.clip(weight, 0.0, 1.0))
        if weight <= 0.0:
            return cls._multiclass_linear_head(ovr_fitted)
        if weight >= 1.0:
            return cls._multiclass_linear_head(multinomial_fitted)
        ovr = cls._multiclass_linear_head(ovr_fitted)
        multi = cls._multiclass_linear_head(multinomial_fitted)
        if ovr.coef_.shape != multi.coef_.shape:
            raise RuntimeError("multiclass head shapes do not match for blending")
        return _LinearMulticlassHead(
            coef_=(1.0 - weight) * ovr.coef_ + weight * multi.coef_,
            intercept_=(1.0 - weight) * ovr.intercept_ + weight * multi.intercept_,
        )

    def _fit_multiclass_ovr(self, design: sparse.csr_matrix, y: np.ndarray, C: float):
        n_classes = int(np.max(y, initial=0)) + 1

        def make_fit(output):
            def fit_one():
                target = (y == output).astype(np.int32)
                return fit_binary_logistic_exact(
                    design, target, C=float(C),
                    random_state=self.random_state + 104729 * output,
                    max_iter=2500,
                )
            return fit_one

        return self._parallel_head_results(
            [make_fit(output) for output in range(n_classes)]
        )

    def _fit_multiclass_multinomial(
        self, design: sparse.csr_matrix, y: np.ndarray, C: float
    ):
        return LogisticRegression(
            solver="newton-cg",
            C=float(C),
            tol=1e-7,
            max_iter=500,
            random_state=self.random_state,
        ).fit(design, y)

    def _fit_outputs(
        self,
        design: sparse.csr_matrix,
        y: np.ndarray,
        C: float,
        objective: str | None = None,
        multinomial_weight: float = 0.0,
    ):
        if self.task_type == "multiclass":
            objective = self.multiclass_objective if objective is None else str(objective)
            if objective == "auto":
                raise RuntimeError("auto objective must be resolved before fitting outputs")
            if objective == "multinomial":
                model = self._fit_multiclass_multinomial(design, y, C)
                return model, self._probability_from_fitted(model, design)
            if objective == "blend":
                ovr = self._fit_multiclass_ovr(design, y, C)
                multinomial = self._fit_multiclass_multinomial(design, y, C)
                model = self._blend_multiclass_heads(
                    ovr, multinomial, multinomial_weight
                )
                return model, self._probability_from_fitted(model, design)
            models = self._fit_multiclass_ovr(design, y, C)
            return models, self._probability_from_fitted(models, design)

        def make_fit(output):
            def fit_one():
                target = y[:, output]
                unique = np.unique(target)
                if len(unique) == 1:
                    return float(unique[0])
                return fit_binary_logistic_exact(
                    design, target, C=float(C),
                    random_state=self.random_state + 104729 * output,
                    max_iter=2500,
                )
            return fit_one

        models = self._parallel_head_results(
            [make_fit(output) for output in range(y.shape[1])]
        )
        return models, self._probability_from_fitted(models, design)

    def _validation_probability(self, fitted, valid_design):
        return self._probability_from_fitted(fitted, valid_design)

    def _loss(self, y, probability):
        if self.task_type == "multiclass":
            labels = np.arange(probability.shape[1])
            return float(log_loss(y, probability, labels=labels))
        losses = []
        for output in range(y.shape[1]):
            if np.unique(y[:, output]).size < 2:
                continue
            p = np.clip(probability[:, output], 1e-10, 1 - 1e-10)
            target = y[:, output]
            losses.append(float(np.mean(-(target * np.log(p) + (1 - target) * np.log(1 - p)))))
        return float(np.mean(losses)) if losses else 0.0

    def _maximal_code_metadata(self, n_features: int, n_pairs: int, n_fine_pairs: int):
        metadata = []
        for feature in range(int(n_features)):
            for level in self.levels:
                metadata.append(("main", int(level), feature))
        for pair_index in range(int(n_pairs)):
            for level in self.levels:
                if level <= 8 or pair_index < int(n_fine_pairs):
                    metadata.append(("pair", int(level), pair_index))
        return metadata

    @staticmethod
    def _candidate_code_columns(metadata, config: SharedStructureConfig):
        columns = []
        for column, (kind, level, index) in enumerate(metadata):
            if kind == "main" and level <= int(config.max_main_level):
                columns.append(column)
            elif kind == "pair":
                if level <= 8 and index < int(config.n_pairs):
                    columns.append(column)
                elif level > 8 and index < int(config.n_fine_pairs):
                    columns.append(column)
        return columns

    def _fit_structure(self, X: np.ndarray, y: np.ndarray, config: SharedStructureConfig):
        self.extra_descriptors_ = ()
        self.extra_lookup_ = ()
        self.representation_strategy_ = "baseline"
        self.class_delta_selected_budget_ = 0
        self.class_delta_validation_scores_ = []
        self.class_delta_validation_improvement_ = 0.0
        self.class_delta_fit_diagnostics_ = []
        self.encoder_ = self._new_encoder()
        all_states = self.encoder_.fit_transform(X)
        scores = _aggregate_feature_scores(all_states[self.ranking_level], y, self.task_type)
        self.feature_idx_ = np.argsort(-scores, kind="stable")[: min(self.max_features, X.shape[1])]
        states = {level: values[:, self.feature_idx_] for level, values in all_states.items()}
        ranked = (
            _rank_pairs(
                states[self.ranking_level], y, self.task_type,
                config.n_pairs, self.pair_feature_limit,
                config.pair_aggregation,
            )
            if config.n_pairs > 0
            else []
        )
        self.pairs_ = ranked[: config.n_pairs]
        self.fine_pairs_ = self.pairs_[: min(config.n_fine_pairs, len(self.pairs_))]
        self.config_ = SharedStructureConfig(
            len(self.pairs_), len(self.fine_pairs_), config.max_main_level,
            config.C, config.pair_aggregation, config.multiclass_objective,
            config.multinomial_weight,
        )
        self.multiclass_objective_ = str(config.multiclass_objective)
        self.multinomial_weight_ = float(config.multinomial_weight)
        self._prepare_residual_maps()
        codes = self._build_codes(states, self.config_)
        self.oh_ = ReferenceStateEncoder()
        design = self.oh_.fit_transform(codes)
        fitted, _ = self._fit_outputs(
            design, y, config.C, config.multiclass_objective, config.multinomial_weight
        )
        self._compile_lookup(fitted)
        self.design_dim_ = int(design.shape[1])
        self.operator_columns_ = int(codes.shape[1])
        self.model_bytes_estimate_ = int(
            sum(table.nbytes for table in self.lookup_)
            + self.intercept_.nbytes
            + self.encoder_.threshold_bytes_
            + self.feature_idx_.nbytes
            + np.asarray(self.pairs_, dtype=np.int32).nbytes
        )
        return self

    def _fit_adaptive_class_delta(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train: np.ndarray,
        valid: np.ndarray,
        selected: SharedStructureConfig,
    ):
        """Select a bounded class-specific delta using the existing split.

        The already-selected base structure is fitted once on the existing
        internal training rows.  This avoids a nested CERM search and reuses the
        state bank contract established by the base TrainingGraph.
        """

        temporary = object.__new__(SharedFiniteStateModel)
        temporary.__dict__.update(self.__dict__)
        temporary.representation_strategy = "baseline"
        temporary._fit_structure(X[train], y[train], selected)
        maximum = int(self.class_specific_budget)
        bank = generate_class_specific_deltas(
            temporary,
            X[train],
            y[train],
            maximum=maximum,
        )
        requested_prefixes = budget_prefixes(maximum)[1:]
        prefix_rows: list[tuple[int, int]] = []
        seen_budgets = {0}
        for requested in requested_prefixes:
            budget = min(int(requested), len(bank))
            if budget in seen_budgets:
                continue
            seen_budgets.add(budget)
            prefix_rows.append((int(requested), int(budget)))
        maximum_budget = max((budget for _, budget in prefix_rows), default=0)
        fitted_descriptors = bank[:maximum_budget]
        if maximum_budget > 0:
            fitted_lookups, _ = fit_class_delta_tables(
                temporary, fitted_descriptors, X[train], y[train]
            )
        else:
            fitted_lookups = []
        probabilities = class_delta_prefix_probabilities(
            temporary,
            X[valid],
            fitted_descriptors,
            fitted_lookups,
            [0, *(budget for _, budget in prefix_rows)],
        )
        baseline_probability = probabilities[0]
        baseline_loss = self._loss(y[valid], baseline_probability)
        rows = [{
            "budget": 0,
            "available_descriptors": len(bank),
            "validation_log_loss": float(baseline_loss),
            "model_bytes_estimate": int(temporary.model_bytes_estimate_),
            "model_ratio": 1.0,
            "resource_eligible": True,
        }]
        for requested, budget in prefix_rows:
            descriptors = fitted_descriptors[:budget]
            lookups = fitted_lookups[:budget]
            probability = probabilities[budget]
            bytes_estimate = int(
                temporary.model_bytes_estimate_
                + extra_model_bytes(descriptors, lookups)
            )
            ratio = float(bytes_estimate / max(temporary.model_bytes_estimate_, 1))
            rows.append({
                "budget": budget,
                "requested_budget": int(requested),
                "available_descriptors": len(bank),
                "validation_log_loss": self._loss(y[valid], probability),
                "model_bytes_estimate": bytes_estimate,
                "model_ratio": ratio,
                "resource_eligible": bool(ratio <= 2.0),
            })
        eligible = [
            row for row in rows
            if row["budget"] > 0
            and row["resource_eligible"]
            and row["validation_log_loss"] <= baseline_loss - 0.001
        ]
        if eligible:
            best_loss = min(row["validation_log_loss"] for row in eligible)
            near = [
                row for row in eligible
                if row["validation_log_loss"] <= best_loss + 0.0015
            ]
            chosen = min(
                near,
                key=lambda row: (row["budget"], row["validation_log_loss"]),
            )
        else:
            chosen = rows[0]
        selected_budget = int(chosen["budget"])

        self._fit_structure(X, y, selected)
        final_descriptors = generate_class_specific_deltas(
            self,
            X,
            y,
            maximum=selected_budget,
        )
        final_descriptors = tuple(final_descriptors[:selected_budget])
        final_lookups, fit_diagnostics = fit_class_delta_tables(
            self, final_descriptors, X, y
        )
        self.extra_descriptors_ = final_descriptors
        self.extra_lookup_ = tuple(final_lookups)
        self.representation_strategy_ = "adaptive"
        self.class_delta_selected_budget_ = len(final_descriptors)
        self.class_delta_validation_scores_ = sorted(
            rows, key=lambda row: row["budget"]
        )
        self.class_delta_baseline_validation_loss_ = float(baseline_loss)
        self.class_delta_selected_validation_loss_ = float(
            chosen["validation_log_loss"]
        )
        self.class_delta_validation_improvement_ = float(
            baseline_loss - chosen["validation_log_loss"]
        )
        self.class_delta_fit_diagnostics_ = fit_diagnostics
        added_bytes = extra_model_bytes(final_descriptors, final_lookups)
        self.model_bytes_estimate_ = int(self.model_bytes_estimate_ + added_bytes)
        self.operator_columns_ = int(
            self.operator_columns_ + len(final_descriptors)
        )
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        selection = self._selection_indices(y)
        if self.task_type == "multiclass":
            train, valid = train_test_split(
                selection, test_size=0.22, random_state=self.random_state, stratify=y[selection]
            )
        else:
            train, valid = train_test_split(
                selection, test_size=0.22, random_state=self.random_state
            )
        self.selection_rows_ = int(len(selection))
        encoder = self._new_encoder()
        states_train_all = encoder.fit_transform(X[train])
        states_valid_all = encoder.transform(X[valid])
        scores = _aggregate_feature_scores(states_train_all[self.ranking_level], y[train], self.task_type)
        idx = np.argsort(-scores, kind="stable")[: min(self.max_features, X.shape[1])]
        st = {level: values[:, idx] for level, values in states_train_all.items()}
        sv = {level: values[:, idx] for level, values in states_valid_all.items()}
        configs = self._configs()
        results = []
        design_cache = {}
        graph_columns = 0
        graph_design_dim = 0

        # Each aggregation has its own ranked pair bank. Candidate designs are
        # still exact prefix/views within that bank, so no candidate rebuilds
        # state columns or one-hot encodings.
        configs_by_aggregation: dict[str, list[SharedStructureConfig]] = {}
        for config in configs:
            configs_by_aggregation.setdefault(config.pair_aggregation, []).append(config)

        for aggregation, aggregation_configs in configs_by_aggregation.items():
            max_pairs = max(config.n_pairs for config in aggregation_configs)
            max_fine = max(config.n_fine_pairs for config in aggregation_configs)
            ranked = _rank_pairs(
                st[self.ranking_level], y[train], self.task_type,
                max_pairs, self.pair_feature_limit, aggregation,
            )

            cache = object.__new__(SharedFiniteStateModel)
            cache.__dict__.update(self.__dict__)
            cache.encoder_ = encoder
            cache.feature_idx_ = idx
            cache.pairs_ = ranked[:max_pairs]
            cache.fine_pairs_ = cache.pairs_[: min(max_fine, len(cache.pairs_))]
            cache.config_ = SharedStructureConfig(
                len(cache.pairs_), len(cache.fine_pairs_), self.max_bins,
                0.2, aggregation, aggregation_configs[0].multiclass_objective,
                aggregation_configs[0].multinomial_weight,
            )
            cache._prepare_residual_maps()
            max_train_codes = cache._build_codes(st, cache.config_)
            max_valid_codes = cache._build_codes(sv, cache.config_)
            bank = EncodedColumnBank.build(
                max_train_codes, max_valid_codes, ReferenceStateEncoder
            )
            metadata = self._maximal_code_metadata(
                len(idx), len(cache.pairs_), len(cache.fine_pairs_)
            )
            if len(metadata) != max_train_codes.shape[1]:
                raise RuntimeError("shared maximal code metadata mismatch")
            graph_columns = max(graph_columns, int(max_train_codes.shape[1]))
            graph_design_dim = max(graph_design_dim, int(bank.train.shape[1]))

            for config in aggregation_configs:
                actual = SharedStructureConfig(
                    min(config.n_pairs, len(cache.pairs_)),
                    min(config.n_fine_pairs, config.n_pairs, len(cache.fine_pairs_)),
                    config.max_main_level, config.C, aggregation,
                    config.multiclass_objective, config.multinomial_weight,
                )
                key = (aggregation, actual.n_pairs, actual.n_fine_pairs, actual.max_main_level)
                if key not in design_cache:
                    columns = self._candidate_code_columns(metadata, actual)
                    design_cache[key] = bank.view(columns)
                z_train, z_valid = design_cache[key]
                if self.task_type == "multiclass" and actual.multiclass_objective == "blend":
                    ovr = self._fit_multiclass_ovr(z_train, y[train], actual.C)
                    multinomial = self._fit_multiclass_multinomial(z_train, y[train], actual.C)
                    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                        if weight <= 0.0:
                            objective = "ovr"
                        elif weight >= 1.0:
                            objective = "multinomial"
                        else:
                            objective = "blend"
                        fitted = self._blend_multiclass_heads(ovr, multinomial, weight)
                        probability = self._validation_probability(fitted, z_valid)
                        loss = self._loss(y[valid], probability)
                        bytes_estimate = int(
                            z_train.shape[1] * max(1, probability.shape[1]) * 8
                        )
                        candidate = SharedStructureConfig(
                            actual.n_pairs, actual.n_fine_pairs,
                            actual.max_main_level, actual.C,
                            actual.pair_aggregation, objective, float(weight),
                        )
                        results.append((loss, bytes_estimate, candidate))
                else:
                    fitted, _ = self._fit_outputs(
                        z_train, y[train], actual.C,
                        actual.multiclass_objective, actual.multinomial_weight,
                    )
                    probability = self._validation_probability(fitted, z_valid)
                    loss = self._loss(y[valid], probability)
                    bytes_estimate = int(
                        z_train.shape[1] * max(1, probability.shape[1]) * 8
                    )
                    results.append((loss, bytes_estimate, actual))
        results.sort(key=lambda row: (row[0], row[1]))

        selected_row = self._select_validation_result(results)
        pre_guard = selected_row[2]
        if self.multiclass_objective == "auto":
            (
                selected_row,
                best_ovr_validation_loss,
                non_ovr_validation_advantage,
                objective_guard_applied,
            ) = self._guard_auto_validation_result(results, selected_row)
        else:
            ovr_losses = [
                self._selection_loss(row)
                for row in results
                if row[2].multiclass_objective == "ovr"
            ]
            best_ovr_validation_loss = (
                min(ovr_losses) if ovr_losses else float("nan")
            )
            non_ovr_validation_advantage = float("nan")
            objective_guard_applied = False
        selected = selected_row[2]
        self.validation_scores_ = results
        self.selected_validation_loss_ = float(selected_row[0])
        self.best_unpenalized_validation_loss_ = float(results[0][0])
        self.best_ovr_validation_loss_ = float(best_ovr_validation_loss)
        self.pre_guard_multiclass_objective_ = str(
            pre_guard.multiclass_objective
        )
        self.pre_guard_multinomial_weight_ = float(
            pre_guard.multinomial_weight
        )
        self.non_ovr_validation_advantage_ = float(
            non_ovr_validation_advantage
        )
        self.objective_guard_applied_ = bool(objective_guard_applied)
        self.objective_guard_margin_ = float(
            self._NON_OVR_SELECTION_MARGIN
            if self.multiclass_objective == "auto"
            else 0.0
        )
        self.objective_selection_penalty_ = float(
            self._BLEND_SELECTION_PENALTY
            if self.multiclass_objective == "auto" else 0.0
        )
        self.training_graph_columns_ = int(graph_columns)
        self.training_graph_design_dim_ = int(graph_design_dim)
        self.training_graph_design_count_ = int(len(design_cache))
        if self.task_type == "multiclass" and self.representation_strategy == "adaptive":
            if self.class_specific_budget == 0:
                self._fit_structure(X, y, selected)
                self.representation_strategy_ = "adaptive"
                self.class_delta_baseline_validation_loss_ = float(
                    self.selected_validation_loss_
                )
                self.class_delta_selected_validation_loss_ = float(
                    self.selected_validation_loss_
                )
                return self
            representation_rows = np.arange(len(y), dtype=np.int64)
            representation_train, representation_valid = train_test_split(
                representation_rows,
                test_size=0.22,
                random_state=self._REPRESENTATION_VALIDATION_SEED,
                stratify=y,
            )
            return self._fit_adaptive_class_delta(
                X,
                y,
                representation_train,
                representation_valid,
                selected,
            )
        return self._fit_structure(X, y, selected)

    def _compile_lookup(self, fitted):
        if self.task_type == "multiclass":
            if isinstance(fitted, (list, tuple)):
                coef = np.vstack([np.asarray(model.coef_[0], dtype=np.float64) for model in fitted])
                intercept = np.asarray([float(model.intercept_[0]) for model in fitted], dtype=np.float64)
                n_classes = len(fitted)
            else:
                coef = np.asarray(fitted.coef_, dtype=np.float64)
                intercept = np.asarray(fitted.intercept_, dtype=np.float64)
                n_classes = int(coef.shape[0])
            self.classes_ = np.arange(n_classes, dtype=np.int32)
            self.constant_outputs_ = None
        else:
            n_outputs = len(fitted)
            coef = np.zeros((n_outputs, self.oh_.n_features_out_), dtype=np.float64)
            intercept = np.zeros(n_outputs, dtype=np.float64)
            constants = np.full(n_outputs, np.nan, dtype=np.float64)
            for output, model in enumerate(fitted):
                if isinstance(model, float):
                    constants[output] = model
                else:
                    coef[output] = model.coef_[0]
                    intercept[output] = model.intercept_[0]
            self.classes_ = None
            self.constant_outputs_ = constants
        self.intercept_ = intercept
        self.lookup_ = []
        offset = 0
        for cardinality in self.oh_.cardinalities_:
            card = int(cardinality)
            table = np.zeros((card, coef.shape[0]), dtype=np.float64)
            width = max(card - 1, 0)
            if width:
                table[1:, :] = coef[:, offset : offset + width].T
            offset += width
            self.lookup_.append(table)
        if offset != coef.shape[1]:
            raise RuntimeError("shared coefficient layout mismatch")
        self.lookup_active_rows_ = tuple(
            np.any(table != 0.0, axis=1) for table in self.lookup_
        )
        self._prepare_execution_maps()

    def _prepare_execution_maps(self) -> None:
        """Compose nested quotient maps into one exact execution level.

        Every emitted code is the same code produced by ``_build_codes``.  The
        pass only replaces repeated level mapping and pair-joint construction
        with small precomputed integer maps; lookup additions keep their
        original order, so decision scores remain bitwise identical.
        """
        coarse = self.levels[0]
        main_levels = tuple(
            level for level in self.levels
            if level <= int(self.config_.max_main_level)
        )
        pair_levels = tuple(
            level for level in self.levels
            if level <= min(int(self.max_bins), 8)
        )
        pair_top = (
            16
            if 16 in self.levels and len(self.fine_pairs_) > 0
            else (
                max(pair_levels)
                if pair_levels and len(self.pairs_) > 0
                else coarse
            )
        )
        self.execution_level_ = int(max(max(main_levels), pair_top))
        execution_cards = np.asarray(
            self.encoder_.cardinalities_[self.execution_level_][self.feature_idx_],
            dtype=np.int64,
        )
        self.execution_cardinalities_ = execution_cards

        parent_cache: dict[tuple[int, int], np.ndarray] = {}

        def labels(raw_feature: int, level: int) -> np.ndarray:
            key = (int(raw_feature), int(level))
            cached = parent_cache.get(key)
            if cached is not None:
                return cached
            raw_feature = int(raw_feature)
            child_card = int(
                self.encoder_.cardinalities_[self.execution_level_][raw_feature]
            )
            if int(level) == self.execution_level_:
                value = np.arange(child_card, dtype=np.int32)
            else:
                value = _parent_labels_from_fine(
                    self.encoder_.maps_[self.execution_level_][raw_feature],
                    self.encoder_.maps_[int(level)][raw_feature],
                    child_card,
                ).astype(np.int32, copy=False)
            parent_cache[key] = value
            return value

        main_maps = []
        for position, raw_feature in enumerate(self.feature_idx_):
            feature_maps = []
            for index, level in enumerate(main_levels):
                level_state = labels(int(raw_feature), int(level))
                if index == 0:
                    code_map = level_state
                else:
                    code_map = self.main_residuals_[level][position][level_state]
                feature_maps.append(np.asarray(code_map, dtype=np.int32))
            main_maps.append(tuple(feature_maps))
        self.execution_main_code_maps_ = tuple(main_maps)

        fine_pairs = set(self.fine_pairs_)
        pair_plans = []
        for j, k in self.pairs_:
            raw_j = int(self.feature_idx_[j])
            raw_k = int(self.feature_idx_[k])
            card_j = int(execution_cards[j])
            card_k = int(execution_cards[k])
            exec_j = np.arange(card_j, dtype=np.int64)[:, None]
            exec_k = np.arange(card_k, dtype=np.int64)[None, :]
            code_maps = []
            for level in pair_levels:
                state_j = labels(raw_j, level)[exec_j]
                state_k = labels(raw_k, level)[exec_k]
                level_card_k = int(self.encoder_.cardinalities_[level][raw_k])
                joint = state_j * level_card_k + state_k
                code = (
                    joint
                    if level == coarse
                    else self.pair_residuals_[level][(j, k)][joint]
                )
                code_maps.append(np.asarray(code, dtype=np.int32).reshape(-1))
            if 16 in self.levels and (j, k) in fine_pairs:
                state_j = labels(raw_j, 16)[exec_j]
                state_k = labels(raw_k, 16)[exec_k]
                level_card_k = int(self.encoder_.cardinalities_[16][raw_k])
                joint = state_j * level_card_k + state_k
                code_maps.append(
                    np.asarray(
                        self.pair_residuals_[16][(j, k)][joint],
                        dtype=np.int32,
                    ).reshape(-1)
                )
            pair_plans.append((int(j), int(k), card_k, tuple(code_maps)))
        self.execution_pair_plans_ = tuple(pair_plans)

    def _states_and_codes(self, X):
        all_states = self.encoder_.transform(np.asarray(X, dtype=np.float64))
        states = {level: values[:, self.feature_idx_] for level, values in all_states.items()}
        return all_states, self._build_codes(states, self.config_)

    def _codes(self, X):
        return self._states_and_codes(X)[1]

    def _selected_states(self, X):
        return self.encoder_.transform_columns(
            np.asarray(X, dtype=np.float64), self.feature_idx_
        )

    @staticmethod
    def _add_lookup_rows(
        score: np.ndarray,
        state: np.ndarray,
        table: np.ndarray,
        active_rows: np.ndarray | None = None,
    ) -> None:
        state = np.asarray(state)
        valid = (state >= 0) & (state < len(table))
        if active_rows is not None and np.any(valid):
            active = np.zeros_like(valid)
            active[valid] = active_rows[state[valid]]
            valid &= active
        if np.any(valid):
            score[valid] += table[state[valid]]

    def _decision_from_selected_states(
        self, selected_states: dict[int, np.ndarray]
    ) -> np.ndarray:
        coarse = self.levels[0]
        n_rows = len(selected_states[coarse])
        score = np.tile(self.intercept_, (n_rows, 1))
        active_rows = getattr(self, "lookup_active_rows_", None)
        lookup_col = 0
        main_levels = [
            level for level in self.levels
            if level <= self.config_.max_main_level
        ]
        for j in range(selected_states[coarse].shape[1]):
            table = self.lookup_[lookup_col]
            self._add_lookup_rows(
                score, selected_states[coarse][:, j], table,
                None if active_rows is None else active_rows[lookup_col],
            )
            lookup_col += 1
            for level in main_levels[1:]:
                state = self.main_residuals_[level][j][
                    selected_states[level][:, j]
                ]
                table = self.lookup_[lookup_col]
                self._add_lookup_rows(
                    score, state, table,
                    None if active_rows is None else active_rows[lookup_col],
                )
                lookup_col += 1

        pair_levels = [
            level for level in self.levels
            if level <= min(self.max_bins, 8)
        ]
        fine_pairs = set(self.fine_pairs_)
        joint = np.empty(n_rows, dtype=np.int64)
        for j, k in self.pairs_:
            raw_k = int(self.feature_idx_[k])
            for level in pair_levels:
                card_k = int(self.encoder_.cardinalities_[level][raw_k])
                np.multiply(
                    selected_states[level][:, j], card_k,
                    out=joint, casting="unsafe",
                )
                joint += selected_states[level][:, k]
                state = (
                    joint
                    if level == coarse
                    else self.pair_residuals_[level][(j, k)][joint]
                )
                table = self.lookup_[lookup_col]
                self._add_lookup_rows(
                    score, state, table,
                    None if active_rows is None else active_rows[lookup_col],
                )
                lookup_col += 1
            if 16 in self.levels and (j, k) in fine_pairs:
                card_k = int(self.encoder_.cardinalities_[16][raw_k])
                np.multiply(
                    selected_states[16][:, j], card_k,
                    out=joint, casting="unsafe",
                )
                joint += selected_states[16][:, k]
                state = self.pair_residuals_[16][(j, k)][joint]
                table = self.lookup_[lookup_col]
                self._add_lookup_rows(
                    score, state, table,
                    None if active_rows is None else active_rows[lookup_col],
                )
                lookup_col += 1
        if lookup_col != len(self.lookup_):
            raise RuntimeError("shared lookup execution layout mismatch")
        return score

    def _decision_from_execution_states(
        self, execution_states: np.ndarray
    ) -> np.ndarray:
        """Execute the compiled lookup program from one quotient level.

        The lookup-table additions deliberately follow the original emitted
        column order, preserving bitwise score equality while eliminating
        repeated hierarchy transforms and pair-joint construction.
        """
        execution_states = np.asarray(execution_states, dtype=np.int32)
        n_rows = len(execution_states)
        score = np.tile(self.intercept_, (n_rows, 1))
        active_rows = getattr(self, "lookup_active_rows_", None)
        lookup_col = 0

        for position, code_maps in enumerate(self.execution_main_code_maps_):
            fine_state = execution_states[:, position]
            for code_map in code_maps:
                state = code_map[fine_state]
                self._add_lookup_rows(
                    score, state, self.lookup_[lookup_col],
                    None if active_rows is None else active_rows[lookup_col],
                )
                lookup_col += 1

        joint = np.empty(n_rows, dtype=np.int64)
        for left, right, right_cardinality, code_maps in self.execution_pair_plans_:
            np.multiply(
                execution_states[:, left], right_cardinality,
                out=joint, casting="unsafe",
            )
            joint += execution_states[:, right]
            for code_map in code_maps:
                state = code_map[joint]
                self._add_lookup_rows(
                    score, state, self.lookup_[lookup_col],
                    None if active_rows is None else active_rows[lookup_col],
                )
                lookup_col += 1

        if lookup_col != len(self.lookup_):
            raise RuntimeError("shared compiled execution layout mismatch")
        return score

    def _decision_function_from_states(
        self, all_states: dict[int, np.ndarray]
    ) -> np.ndarray:
        execution_states = all_states[self.execution_level_][:, self.feature_idx_]
        score = self._decision_from_execution_states(execution_states)
        for descriptor, table in zip(
            getattr(self, "extra_descriptors_", ()),
            getattr(self, "extra_lookup_", ()),
        ):
            state = (
                all_states[int(descriptor.level)][:, int(descriptor.raw0)]
                == int(descriptor.state)
            ).astype(np.int32)
            score += np.asarray(table, dtype=np.float64)[state]
        if self.task_type == "multilabel" and self.constant_outputs_ is not None:
            for output, value in enumerate(self.constant_outputs_):
                if np.isfinite(value):
                    score[:, output] = 40.0 if value >= 0.5 else -40.0
        return score

    def decision_function(self, X):
        matrix = np.asarray(X, dtype=np.float64)
        execution_states = self.encoder_.transform_level_columns(
            matrix, self.execution_level_, self.feature_idx_
        )
        score = self._decision_from_execution_states(execution_states)
        descriptors = tuple(getattr(self, "extra_descriptors_", ()))
        lookups = tuple(getattr(self, "extra_lookup_", ()))
        if descriptors:
            raw_columns = tuple(dict.fromkeys(int(item.raw0) for item in descriptors))
            raw_positions = {raw: pos for pos, raw in enumerate(raw_columns)}
            extra_states = self.encoder_.transform_columns(matrix, raw_columns)
            for descriptor, table in zip(descriptors, lookups):
                state = (
                    extra_states[int(descriptor.level)][:, raw_positions[int(descriptor.raw0)]]
                    == int(descriptor.state)
                ).astype(np.int32)
                score += np.asarray(table, dtype=np.float64)[state]
        if self.task_type == "multilabel" and self.constant_outputs_ is not None:
            for output, value in enumerate(self.constant_outputs_):
                if np.isfinite(value):
                    score[:, output] = 40.0 if value >= 0.5 else -40.0
        return score

    def predict_proba(self, X):
        score = self.decision_function(X)
        if self.task_type == "multiclass":
            return _softmax(score)
        probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
        if self.constant_outputs_ is not None:
            for output, value in enumerate(self.constant_outputs_):
                if np.isfinite(value):
                    probability[:, output] = value
        return probability

    def predict(self, X):
        probability = self.predict_proba(X)
        if self.task_type == "multiclass":
            return self.classes_[np.argmax(probability, axis=1)]
        threshold = np.asarray(self.threshold, dtype=np.float64)
        return (probability >= threshold).astype(np.int32)

    @staticmethod
    def _pack_variable(arrays, dtype):
        arrays = [np.asarray(array, dtype=dtype).ravel() for array in arrays]
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        if arrays:
            np.cumsum([len(array) for array in arrays], out=offsets[1:])
            flat = np.concatenate(arrays) if offsets[-1] else np.empty(0, dtype=dtype)
        else:
            flat = np.empty(0, dtype=dtype)
        return flat, offsets

    @staticmethod
    def _unpack_variable(flat, offsets):
        flat = np.asarray(flat)
        offsets = np.asarray(offsets, dtype=np.int64)
        return [flat[offsets[index] : offsets[index + 1]].copy() for index in range(len(offsets) - 1)]

    def export_ir(self, prefix: str | Path):
        prefix = Path(prefix)
        npz = prefix.with_suffix(".npz")
        js = prefix.with_suffix(".json")
        arrays = {
            "feature_idx": np.asarray(self.feature_idx_, dtype=np.int32),
            "pairs": np.asarray(self.pairs_, dtype=np.int32).reshape(-1, 2),
            "fine_pairs": np.asarray(self.fine_pairs_, dtype=np.int32).reshape(-1, 2),
            "direct_state_mask": np.asarray(self.encoder_.direct_state_mask_, dtype=np.uint8),
            "direct_state_cardinalities": np.asarray(self.encoder_.direct_state_cardinalities_, dtype=np.int32),
            "intercept": np.asarray(self.intercept_, dtype=np.float64),
            "levels": np.asarray(self.levels, dtype=np.int16),
            "classes": np.asarray([] if self.classes_ is None else self.classes_, dtype=np.int64),
            "constant_outputs": np.asarray([] if self.constant_outputs_ is None else self.constant_outputs_, dtype=np.float64),
        }
        threshold_flat, threshold_offsets = self._pack_variable(
            self.encoder_.thresholds_, np.float64
        )
        arrays["threshold_values"] = threshold_flat
        arrays["threshold_offsets"] = threshold_offsets
        for level in self.levels:
            values, offsets = self._pack_variable(self.encoder_.maps_[level], np.int16)
            arrays[f"map{level}_values"] = values
            arrays[f"map{level}_offsets"] = offsets
            if level > self.levels[0]:
                values, offsets = self._pack_variable(self.main_residuals_[level], np.int32)
                arrays[f"main_res{level}_values"] = values
                arrays[f"main_res{level}_offsets"] = offsets
                pairs = self.pairs_ if level <= 8 else self.fine_pairs_
                values, offsets = self._pack_variable(
                    [self.pair_residuals_[level][pair] for pair in pairs], np.int32
                )
                arrays[f"pair_res{level}_values"] = values
                arrays[f"pair_res{level}_offsets"] = offsets
        lookup_values, lookup_offsets = self._pack_variable(
            [np.asarray(table, dtype=np.float64).ravel() for table in self.lookup_],
            np.float64,
        )
        arrays["lookup_values"] = lookup_values
        arrays["lookup_offsets"] = lookup_offsets
        arrays["lookup_rows"] = np.asarray([len(table) for table in self.lookup_], dtype=np.int32)
        extra_descriptors = tuple(getattr(self, "extra_descriptors_", ()))
        extra_lookup = tuple(getattr(self, "extra_lookup_", ()))
        if len(extra_descriptors) != len(extra_lookup):
            raise RuntimeError("adaptive descriptor/lookup layout mismatch")
        adaptive_format = bool(
            extra_descriptors
            or getattr(self, "representation_strategy_", "baseline") == "adaptive"
        )
        if adaptive_format:
            arrays["extra_kind"] = np.ones(len(extra_descriptors), dtype=np.int8)
            arrays["extra_raw0"] = np.asarray(
                [descriptor.raw0 for descriptor in extra_descriptors], dtype=np.int32
            )
            arrays["extra_level"] = np.asarray(
                [descriptor.level for descriptor in extra_descriptors], dtype=np.int16
            )
            arrays["extra_state"] = np.asarray(
                [descriptor.state for descriptor in extra_descriptors], dtype=np.int32
            )
            arrays["extra_target_class"] = np.asarray(
                [descriptor.target_class for descriptor in extra_descriptors],
                dtype=np.int32,
            )
            arrays["extra_candidate_score"] = np.asarray(
                [descriptor.candidate_score for descriptor in extra_descriptors],
                dtype=np.float64,
            )
            values, offsets = self._pack_variable(
                [np.asarray(table, dtype=np.float64).ravel() for table in extra_lookup],
                np.float64,
            )
            arrays["extra_lookup_values"] = values
            arrays["extra_lookup_offsets"] = offsets
            arrays["extra_lookup_rows"] = np.asarray(
                [table.shape[0] for table in extra_lookup], dtype=np.int32
            )
            arrays["extra_lookup_outputs"] = np.asarray(
                [table.shape[1] for table in extra_lookup], dtype=np.int32
            )
        np.savez_compressed(npz, **arrays)
        manifest = {
            "format": (
                "cerm-shared-finite-state-v3"
                if adaptive_format else "cerm-shared-finite-state-v2"
            ),
            "task_type": self.task_type,
            "n_outputs": int(self.intercept_.size),
            "n_input_features": int(self.encoder_.n_features_in_),
            "max_bins": self.max_bins,
            "levels": list(self.levels),
            "feature_kinds": [str(value) for value in self.encoder_.feature_kinds_],
            "feature_cardinalities": [
                None if value is None else int(value)
                for value in self.encoder_.feature_cardinalities_
            ],
            "config": self.config_.__dict__,
            "model_bytes_estimate": self.model_bytes_estimate_,
            "array_file": npz.name,
            "pickle_free": True,
            "multiclass_objective": self.multiclass_objective_,
            "multinomial_weight": float(self.config_.multinomial_weight),
        }
        if adaptive_format:
            manifest.update({
                "representation_strategy": "adaptive",
                "class_specific_budget": int(self.class_specific_budget),
                "class_delta_selected_budget": int(
                    getattr(self, "class_delta_selected_budget_", len(extra_descriptors))
                ),
                "class_delta_validation_improvement": float(
                    getattr(self, "class_delta_validation_improvement_", 0.0)
                ),
                "class_delta_validation_scores": list(
                    getattr(self, "class_delta_validation_scores_", ())
                ),
                "representation_validation_seed": int(
                    self._REPRESENTATION_VALIDATION_SEED
                ),
            })
        js.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        return npz, js

    @classmethod
    def load_ir(cls, json_path: str | Path, npz_path: str | Path):
        metadata = json.loads(Path(json_path).read_text(encoding="utf-8"))
        format_name = metadata.get("format")
        if format_name not in {
            "cerm-shared-finite-state-v2",
            "cerm-shared-finite-state-v3",
        }:
            raise ValueError(f"unsupported shared finite-state format: {metadata.get('format')!r}")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        model = cls(
            task_type=metadata["task_type"],
            max_features=int(metadata.get("n_input_features", 1)),
            pair_feature_limit=max(1, int(len(arrays["feature_idx"]))),
            max_bins=int(metadata["max_bins"]),
            feature_kinds=tuple(metadata["feature_kinds"]),
            feature_cardinalities=tuple(metadata["feature_cardinalities"]),
            multiclass_objective=(
                str(metadata.get("multiclass_objective", "ovr"))
                if str(metadata.get("multiclass_objective", "ovr")) in {"ovr", "multinomial"}
                else "auto"
            ),
            representation_strategy=(
                str(metadata.get("representation_strategy", "baseline"))
                if format_name == "cerm-shared-finite-state-v3"
                else "baseline"
            ),
            class_specific_budget=int(metadata.get("class_specific_budget", 0)),
        )
        model.levels = tuple(int(level) for level in arrays["levels"])
        model.ranking_level = 8 if model.max_bins >= 8 else 4
        encoder = NestedQuantileEncoder(
            max_bins=model.max_bins,
            levels=model.levels,
            feature_kinds=tuple(metadata["feature_kinds"]),
            feature_cardinalities=tuple(metadata["feature_cardinalities"]),
        )
        encoder.feature_kinds_ = list(metadata["feature_kinds"])
        encoder.feature_cardinalities_ = list(metadata["feature_cardinalities"])
        encoder.thresholds_ = cls._unpack_variable(
            arrays["threshold_values"], arrays["threshold_offsets"]
        )
        encoder.maps_ = {}
        encoder.cardinalities_ = {}
        for level in model.levels:
            maps = cls._unpack_variable(
                arrays[f"map{level}_values"], arrays[f"map{level}_offsets"]
            )
            encoder.maps_[level] = [np.asarray(value, dtype=np.int16) for value in maps]
            encoder.cardinalities_[level] = np.asarray(
                [int(value.max(initial=0)) + 1 for value in maps], dtype=np.int16
            )
        encoder.direct_state_mask_ = arrays["direct_state_mask"].astype(bool)
        encoder.direct_state_cardinalities_ = arrays["direct_state_cardinalities"].astype(np.int32)
        encoder.n_features_in_ = int(metadata["n_input_features"])
        model.encoder_ = encoder
        model.feature_idx_ = arrays["feature_idx"].astype(np.int32)
        model.pairs_ = [tuple(map(int, pair)) for pair in arrays["pairs"].reshape(-1, 2)]
        model.fine_pairs_ = [tuple(map(int, pair)) for pair in arrays["fine_pairs"].reshape(-1, 2)]
        cfg = metadata["config"]
        model.config_ = SharedStructureConfig(
            int(cfg["n_pairs"]), int(cfg["n_fine_pairs"]),
            int(cfg["max_main_level"]), float(cfg["C"]),
            str(cfg.get("pair_aggregation", "joint")),
            str(cfg.get("multiclass_objective", metadata.get("multiclass_objective", "ovr"))),
            float(cfg.get("multinomial_weight", 0.0)),
        )
        model.multiclass_objective_ = str(model.config_.multiclass_objective)
        model.multinomial_weight_ = float(model.config_.multinomial_weight)
        model.main_residuals_ = {}
        model.pair_residuals_ = {level: {} for level in model.levels[1:]}
        for level in model.levels[1:]:
            model.main_residuals_[level] = [
                np.asarray(value, dtype=np.int32)
                for value in cls._unpack_variable(
                    arrays[f"main_res{level}_values"], arrays[f"main_res{level}_offsets"]
                )
            ]
            pairs = model.pairs_ if level <= 8 else model.fine_pairs_
            values = cls._unpack_variable(
                arrays[f"pair_res{level}_values"], arrays[f"pair_res{level}_offsets"]
            )
            model.pair_residuals_[level] = {
                pair: np.asarray(value, dtype=np.int32)
                for pair, value in zip(pairs, values)
            }
        n_outputs = int(metadata["n_outputs"])
        lookup_flat = arrays["lookup_values"]
        lookup_offsets = arrays["lookup_offsets"]
        rows = arrays["lookup_rows"]
        model.lookup_ = []
        for index, row_count in enumerate(rows):
            values = lookup_flat[lookup_offsets[index] : lookup_offsets[index + 1]]
            model.lookup_.append(values.reshape(int(row_count), n_outputs).copy())
        model.lookup_active_rows_ = tuple(
            np.any(table != 0.0, axis=1) for table in model.lookup_
        )
        model._prepare_execution_maps()
        model.intercept_ = arrays["intercept"].astype(np.float64)
        model.classes_ = arrays["classes"].astype(np.int64) if arrays["classes"].size else None
        model.constant_outputs_ = arrays["constant_outputs"].astype(np.float64) if arrays["constant_outputs"].size else None
        model.extra_descriptors_ = ()
        model.extra_lookup_ = ()
        model.representation_strategy_ = "baseline"
        model.class_delta_selected_budget_ = 0
        model.class_delta_validation_scores_ = []
        model.class_delta_validation_improvement_ = 0.0
        model.class_delta_fit_diagnostics_ = []
        if format_name == "cerm-shared-finite-state-v3":
            required = {
                "extra_kind", "extra_raw0", "extra_level", "extra_state",
                "extra_target_class", "extra_candidate_score",
                "extra_lookup_values", "extra_lookup_offsets",
                "extra_lookup_rows", "extra_lookup_outputs",
            }
            missing = sorted(required.difference(arrays))
            if missing:
                raise ValueError(
                    "adaptive shared IR is missing arrays: " + ", ".join(missing)
                )
            count = int(len(arrays["extra_kind"]))
            vector_names = (
                "extra_raw0", "extra_level", "extra_state",
                "extra_target_class", "extra_candidate_score",
                "extra_lookup_rows", "extra_lookup_outputs",
            )
            if any(len(arrays[name]) != count for name in vector_names):
                raise ValueError("adaptive shared IR descriptor shape mismatch")
            offsets = np.asarray(arrays["extra_lookup_offsets"], dtype=np.int64)
            values = np.asarray(arrays["extra_lookup_values"], dtype=np.float64)
            if (
                len(offsets) != count + 1
                or offsets[0] != 0
                or offsets[-1] != len(values)
                or np.any(np.diff(offsets) < 0)
            ):
                raise ValueError("adaptive shared IR lookup offsets are invalid")
            descriptors = []
            lookups = []
            for index in range(count):
                if int(arrays["extra_kind"][index]) != 1:
                    raise ValueError("unsupported adaptive shared IR operator")
                raw = int(arrays["extra_raw0"][index])
                level = int(arrays["extra_level"][index])
                state = int(arrays["extra_state"][index])
                target_class = int(arrays["extra_target_class"][index])
                if not 0 <= raw < encoder.n_features_in_:
                    raise ValueError("adaptive shared IR raw feature is out of range")
                if level not in model.levels:
                    raise ValueError("adaptive shared IR level is unavailable")
                if not 0 <= state < int(encoder.cardinalities_[level][raw]):
                    raise ValueError("adaptive shared IR state is out of range")
                if not 0 <= target_class < n_outputs:
                    raise ValueError("adaptive shared IR class is out of range")
                start, stop = map(int, offsets[index : index + 2])
                row_count = int(arrays["extra_lookup_rows"][index])
                output_count = int(arrays["extra_lookup_outputs"][index])
                if row_count != 2 or output_count != n_outputs:
                    raise ValueError("adaptive shared IR lookup shape is invalid")
                if stop - start != row_count * output_count:
                    raise ValueError("adaptive shared IR lookup extent is invalid")
                table = values[start:stop].reshape(row_count, output_count).copy()
                if not np.all(np.isfinite(table)):
                    raise ValueError("adaptive shared IR lookup must be finite")
                other_outputs = np.delete(table, target_class, axis=1)
                if np.any(other_outputs != 0.0):
                    raise ValueError("adaptive shared IR class mask is invalid")
                descriptors.append(ResidualBasisDescriptor(
                    raw0=raw,
                    level=level,
                    state=state,
                    target_class=target_class,
                    candidate_score=float(arrays["extra_candidate_score"][index]),
                ))
                lookups.append(table)
            selected_budget = int(metadata.get("class_delta_selected_budget", count))
            if selected_budget != count:
                raise ValueError("adaptive shared IR selected budget mismatch")
            model.extra_descriptors_ = tuple(descriptors)
            model.extra_lookup_ = tuple(lookups)
            model.representation_strategy_ = "adaptive"
            model.class_delta_selected_budget_ = count
            model.class_delta_validation_scores_ = list(
                metadata.get("class_delta_validation_scores") or ()
            )
            model.class_delta_validation_improvement_ = float(
                metadata.get("class_delta_validation_improvement", 0.0)
            )
        model.model_bytes_estimate_ = int(metadata.get("model_bytes_estimate", 0))
        model.operator_columns_ = int(
            len(model.lookup_) + len(model.extra_descriptors_)
        )
        return model


@dataclass
class SharedFiniteStateProgram:
    model: SharedFiniteStateModel
    adapter: Any
    input_columns: tuple[str, ...] | None
    adapted_feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...] | None
    task_type: str
    classes: tuple[Any, ...] | None = None
    output_names: tuple[str, ...] | None = None
    threshold: float | tuple[float, ...] = 0.5
    library_version: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def _prepare_input(self, X):
        if self.adapter is None:
            matrix = dense_numeric_matrix(X)
        else:
            frame = validate_dataframe_schema(X, self.input_columns or ())
            matrix = self.adapter.transform(frame).matrix
        if self.feature_indices is not None:
            matrix = matrix[:, np.asarray(self.feature_indices, dtype=np.int64)]
        return np.asarray(matrix, dtype=np.float64)

    def decision_function(self, X):
        return self.model.decision_function(self._prepare_input(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._prepare_input(X))

    def predict(self, X):
        probability = self.predict_proba(X)
        if self.task_type == "multiclass":
            classes = np.asarray(self.classes)
            return classes[np.argmax(probability, axis=1)]
        threshold = np.asarray(self.threshold, dtype=np.float64)
        return (probability >= threshold).astype(np.int32)

    @property
    def model_bytes_estimate(self):
        adapter_bytes = 0
        if self.adapter is not None:
            from .adapters import estimate_typed_adapter_bytes
            adapter_bytes = estimate_typed_adapter_bytes(self.adapter)
        return int(self.model.model_bytes_estimate_ + adapter_bytes)

    def optimize(self, target: str = "balanced"):
        if target not in {"memory", "balanced", "latency"}:
            raise ValueError("target must be memory, balanced, or latency")
        return self

    def autotune(
        self,
        calibration_X,
        prefix: str | Path,
        *,
        target: str = "latency",
        benchmark_rows: int = 50_000,
    ):
        # The shared runtime currently has one exact native lowering.  Preserve
        # the common compiler API while recording the requested objective.
        compiled = self.compile_native(prefix)
        compiled.metadata.update({
            "autotune_target": target,
            "benchmark_rows": int(benchmark_rows),
            "candidate_count": 1,
        })
        return compiled

    def compile_native(self, prefix: str | Path):
        from ._internal.cerm_shared_native_codegen import compile_shared_native
        predict, source, library = compile_shared_native(self.model, prefix)
        adaptive = bool(getattr(self.model, "extra_descriptors_", ()))
        return CompiledSharedFiniteStateProgram(
            predict_matrix=predict,
            adapter=self.adapter,
            input_columns=self.input_columns,
            feature_indices=self.feature_indices,
            task_type=self.task_type,
            classes=self.classes,
            output_names=self.output_names,
            threshold=self.threshold,
            source_path=Path(source),
            library_path=Path(library),
            n_outputs=int(len(self.model.intercept_)),
            metadata={
                **self.metadata,
                "backend": "shared-native-v2" if adaptive else "shared-native-v1",
                "representation_strategy": getattr(
                    self.model, "representation_strategy_", "baseline"
                ),
            },
        )

    def export(self, directory: str | Path, *, config: dict[str, Any] | None = None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_npz, model_json = self.model.export_ir(directory / "model")
        model_format = json.loads(model_json.read_text(encoding="utf-8"))["format"]
        adapter_record = None
        if self.adapter is not None:
            adapter_npz, adapter_json = export_typed_adapter_ir(self.adapter, directory / "adapter")
            def rec(path: Path):
                payload = path.read_bytes()
                return {"file": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            adapter_record = {"npz": rec(adapter_npz), "json": rec(adapter_json)}
        def rec(path: Path):
            payload = path.read_bytes()
            return {"file": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        manifest = {
            "format": (
                "cerm-shared-program-v2"
                if model_format == "cerm-shared-finite-state-v3"
                else "cerm-shared-program-v1"
            ),
            "library_version": self.library_version,
            "task_type": self.task_type,
            "classes": self.classes,
            "output_names": self.output_names,
            "threshold": self.threshold,
            "input_columns": self.input_columns,
            "adapted_feature_names": self.adapted_feature_names,
            "feature_indices": self.feature_indices,
            "model": {"npz": rec(model_npz), "json": rec(model_json)},
            "adapter": adapter_record,
            "metadata": self.metadata,
            "config": config or {},
            "model_bytes_estimate": self.model_bytes_estimate,
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path):
        from .portable import PortableTypedAdapter

        directory = Path(directory)
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") not in {
            "cerm-shared-program-v1",
            "cerm-shared-program-v2",
        }:
            raise ValueError("unsupported shared program format")
        def verify(record):
            path = manifest_path.parent / record["file"]
            payload = path.read_bytes()
            if len(payload) != int(record["bytes"]):
                raise ValueError(f"byte-count mismatch for {path.name}")
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {path.name}")
            return path
        model = SharedFiniteStateModel.load_ir(
            verify(manifest["model"]["json"]),
            verify(manifest["model"]["npz"]),
        )
        adapter = None
        if manifest.get("adapter") is not None:
            adapter = PortableTypedAdapter.load(
                verify(manifest["adapter"]["json"]),
                verify(manifest["adapter"]["npz"]),
            )
        threshold = manifest.get("threshold", 0.5)
        if isinstance(threshold, list):
            threshold = tuple(float(value) for value in threshold)
        classes = manifest.get("classes")
        output_names = manifest.get("output_names")
        feature_indices = manifest.get("feature_indices")
        input_columns = manifest.get("input_columns")
        return cls(
            model=model,
            adapter=adapter,
            input_columns=tuple(input_columns) if input_columns is not None else None,
            adapted_feature_names=tuple(manifest.get("adapted_feature_names") or ()),
            feature_indices=tuple(feature_indices) if feature_indices is not None else None,
            task_type=manifest["task_type"],
            classes=tuple(classes) if classes is not None else None,
            output_names=tuple(output_names) if output_names is not None else None,
            threshold=threshold,
            library_version=manifest.get("library_version", "unknown"),
            metadata=dict(manifest.get("metadata") or {}),
        )

@dataclass
class CompiledSharedFiniteStateProgram:
    predict_matrix: Any
    adapter: Any
    input_columns: tuple[str, ...] | None
    feature_indices: tuple[int, ...] | None
    task_type: str
    classes: tuple[Any, ...] | None
    output_names: tuple[str, ...] | None
    threshold: float | tuple[float, ...]
    source_path: Path
    library_path: Path
    n_outputs: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def _prepare_input(self, X):
        if self.adapter is None:
            matrix = dense_numeric_matrix(X)
        else:
            frame = validate_dataframe_schema(X, self.input_columns or ())
            matrix = self.adapter.transform(frame).matrix
        if self.feature_indices is not None:
            matrix = matrix[:, np.asarray(self.feature_indices, dtype=np.int64)]
        return np.asarray(matrix, dtype=np.float64)

    def predict_proba(self, X):
        return np.asarray(self.predict_matrix(self._prepare_input(X)), dtype=np.float64)

    def predict(self, X):
        probability = self.predict_proba(X)
        if self.task_type == "multiclass":
            return np.asarray(self.classes)[np.argmax(probability, axis=1)]
        return (probability >= np.asarray(self.threshold, dtype=np.float64)).astype(np.int32)

    @staticmethod
    def _record(path: Path):
        payload = path.read_bytes()
        return {"file": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def save(self, directory: str | Path, *, include_source: bool = False):
        import platform
        import shutil

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / self.library_path.name
        shutil.copy2(self.library_path, library)
        source_record = None
        if include_source and self.source_path.is_file():
            source = directory / self.source_path.name
            shutil.copy2(self.source_path, source)
            source_record = self._record(source)
        adapter_record = None
        if self.adapter is not None:
            if hasattr(self.adapter, "export") and self.adapter.__class__.__name__ == "PortableTypedAdapter":
                adapter_npz, adapter_json = self.adapter.export(directory / "adapter")
                portable = True
            else:
                adapter_npz, adapter_json = export_typed_adapter_ir(self.adapter, directory / "adapter")
                portable = bool(json.loads(adapter_json.read_text(encoding="utf-8")).get("portable", False))
            if not portable:
                adapter_npz.unlink(missing_ok=True)
                adapter_json.unlink(missing_ok=True)
                raise ValueError("shared compiled program requires a portable adapter")
            adapter_record = {"npz": self._record(adapter_npz), "json": self._record(adapter_json)}
        runtime = directory / "runtime.json"
        runtime.write_text(json.dumps({
            "input_columns": self.input_columns,
            "feature_indices": self.feature_indices,
            "task_type": self.task_type,
            "classes": self.classes,
            "output_names": self.output_names,
            "threshold": self.threshold,
            "n_outputs": self.n_outputs,
            "metadata": self.metadata,
        }, indent=2, default=str), encoding="utf-8")
        manifest = {
            "format": "cerm-compiled-shared-program-v1",
            "library": self._record(library),
            "source": source_record,
            "runtime": self._record(runtime),
            "adapter": adapter_record,
            "symbol": "cerm_shared_predict",
            "machine": platform.machine(),
            "system": platform.system(),
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path):
        import platform
        from .native_runtime import load_native_matrix_predictor
        from .portable import PortableTypedAdapter

        directory = Path(directory)
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "cerm-compiled-shared-program-v1":
            raise ValueError("unsupported compiled shared program format")
        if manifest.get("machine") != platform.machine() or manifest.get("system") != platform.system():
            raise RuntimeError("compiled shared program targets a different platform")
        def verify(record):
            path = manifest_path.parent / record["file"]
            payload = path.read_bytes()
            if len(payload) != int(record["bytes"]) or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"artifact verification failed: {path.name}")
            return path
        library = verify(manifest["library"])
        runtime = json.loads(verify(manifest["runtime"]).read_text(encoding="utf-8"))
        adapter_record = manifest.get("adapter")
        adapter = None
        if adapter_record is not None:
            adapter = PortableTypedAdapter.load(verify(adapter_record["json"]), verify(adapter_record["npz"]))
        predictor = load_native_matrix_predictor(
            library, n_outputs=int(runtime["n_outputs"]), symbol=manifest.get("symbol", "cerm_shared_predict")
        )
        source_record = manifest.get("source")
        source = verify(source_record) if source_record else manifest_path.parent / "unavailable.cpp"
        return cls(
            predict_matrix=predictor,
            adapter=adapter,
            input_columns=tuple(runtime["input_columns"]) if runtime.get("input_columns") is not None else None,
            feature_indices=tuple(runtime["feature_indices"]) if runtime.get("feature_indices") is not None else None,
            task_type=runtime["task_type"],
            classes=tuple(runtime["classes"]) if runtime.get("classes") is not None else None,
            output_names=tuple(runtime["output_names"]) if runtime.get("output_names") is not None else None,
            threshold=tuple(runtime["threshold"]) if isinstance(runtime.get("threshold"), list) else runtime.get("threshold", 0.5),
            source_path=source,
            library_path=library,
            n_outputs=int(runtime["n_outputs"]),
            metadata=dict(runtime.get("metadata") or {}),
        )
