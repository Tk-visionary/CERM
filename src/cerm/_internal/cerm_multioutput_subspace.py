from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from .._compat.sklearn import fit_multi_target_ridge_lsqr_exact

from ..regression import (
    FiniteStateRidgeRegressor,
    NestedQuantileEncoder,
    RegressionConfig,
)
from ..training_cache import FiniteStateValueHistogramCache, ValueHistogramStats
from .cerm_state_design import ReferenceStateEncoder


def _fit_multi_target_lsqr_exact(
    design,
    target: np.ndarray,
    *,
    alpha: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delegate version-sensitive sparse Ridge details to the sklearn compat layer."""
    return fit_multi_target_ridge_lsqr_exact(
        design,
        target,
        alpha=float(alpha),
        tol=float(tol),
        max_iter=int(max_iter),
    )


@dataclass(frozen=True)
class OutputSubspace:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    mp_edge: float
    rank: int
    explained_energy: float


def output_subspace(target: np.ndarray, *, rank_cap: int = 6) -> OutputSubspace:
    target = np.asarray(target, dtype=np.float64)
    mean = target.mean(axis=0)
    variance = np.mean((target - mean) ** 2, axis=0)
    scale = np.sqrt(np.maximum(variance, np.finfo(np.float64).eps))
    standardized = (target - mean) / scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt.astype(np.float64, copy=True)
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    eigenvalues = (singular * singular) / max(len(target), 1)
    mp_edge = float((1.0 + np.sqrt(target.shape[1] / max(len(target), 1))) ** 2)
    rank = int(min(rank_cap, np.sum(eigenvalues > mp_edge)))
    total = float(eigenvalues.sum())
    explained = 0.0 if total <= 0.0 else float(eigenvalues[:rank].sum() / total)
    return OutputSubspace(
        mean=mean,
        scale=scale,
        components=components,
        eigenvalues=eigenvalues,
        mp_edge=mp_edge,
        rank=rank,
        explained_energy=explained,
    )


def _quota_features(scores: np.ndarray, budget: int) -> np.ndarray:
    ranks = np.argsort(-scores, axis=1, kind="stable")
    chosen: list[int] = []
    seen: set[int] = set()
    depth = 0
    while len(chosen) < budget and depth < scores.shape[1]:
        for direction in range(scores.shape[0]):
            feature = int(ranks[direction, depth])
            if feature not in seen:
                chosen.append(feature)
                seen.add(feature)
                if len(chosen) >= budget:
                    break
        depth += 1
    return np.asarray(chosen, dtype=np.int64)


class MultiOutputSubspaceFiniteState:
    """Experimental one-representation finite-state selector for vector targets."""

    def __init__(
        self,
        *,
        rank: int,
        components: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
        eigenvalues: np.ndarray,
        max_bins: int,
        linear_feature_budget: int,
        state_feature_budget: int = 8,
        interaction_feature_budget: int = 24,
        max_pairs: int = 4,
        head_alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int = 20260803,
        n_jobs: int | None = 1,
    ):
        self.rank = int(rank)
        self.components = np.asarray(components, dtype=np.float64)
        self.target_mean = np.asarray(target_mean, dtype=np.float64)
        self.target_scale = np.asarray(target_scale, dtype=np.float64)
        self.eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self.max_bins = int(max_bins)
        self.linear_feature_budget = int(linear_feature_budget)
        self.state_feature_budget = int(state_feature_budget)
        self.interaction_feature_budget = int(interaction_feature_budget)
        self.max_pairs = int(max_pairs)
        self.head_alpha = float(head_alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self.levels = tuple(level for level in (4, 8, 16) if level <= self.max_bins)

    def _subspace_targets(self, target: np.ndarray) -> np.ndarray:
        standardized = (target - self.target_mean) / self.target_scale
        return standardized @ self.components[: self.rank].T

    @staticmethod
    def _between_group_histogram_scores(
        stats: ValueHistogramStats,
        overall: np.ndarray,
        total: np.ndarray,
    ) -> np.ndarray:
        mass = np.asarray(stats.mass, dtype=np.float64)
        sums = np.asarray(stats.sums, dtype=np.float64)
        means = np.divide(
            sums,
            mass[:, None],
            out=np.zeros_like(sums),
            where=mass[:, None] > 0.0,
        )
        between = np.sum(
            mass[:, None] * (means - overall[None, :]) ** 2,
            axis=0,
        )
        return between / np.maximum(total, 1e-15)

    def _direction_scores(self, states: np.ndarray, target_pc: np.ndarray):
        direction_variance = np.var(target_pc, axis=0)
        overall = np.mean(target_pc, axis=0)
        total = np.sum((target_pc - overall) ** 2, axis=0)
        cache = FiniteStateValueHistogramCache(
            states,
            target_pc,
            pairs=(),
            backend="auto",
            n_jobs=self.n_jobs,
        )
        raw = np.column_stack(
            [
                self._between_group_histogram_scores(stats, overall, total)
                for stats in cache.main
            ]
        )
        normalized = raw / np.maximum(direction_variance[:, None], 1e-15)
        return (
            raw,
            normalized,
            direction_variance,
            overall,
            total,
            cache.backend_,
        )

    def fit(self, matrix: np.ndarray, target: np.ndarray):
        X = np.asarray(matrix, dtype=np.float64)
        Y = np.asarray(target, dtype=np.float64)
        if self.rank < 1:
            raise ValueError("shared subspace rank must be positive")
        if len(self.levels) == 0:
            raise ValueError("max_bins must allow at least the 4-state level")
        pc = self._subspace_targets(Y)
        indices = np.arange(len(X))
        train, valid = train_test_split(
            indices,
            test_size=0.22,
            random_state=self.random_state,
        )
        Xa, Xv = X[train], X[valid]
        Ta, Tv = pc[train], pc[valid]

        encoder = NestedQuantileEncoder(
            max_bins=self.max_bins,
            levels=self.levels,
        )
        Sa_all = encoder.fit_transform(Xa)
        Sv_all = encoder.transform(Xv)
        ranking_level = max(self.levels)
        rank_states = Sa_all[ranking_level]
        (
            raw_scores,
            normalized_scores,
            direction_variance,
            target_overall,
            target_total,
            main_histogram_backend,
        ) = self._direction_scores(rank_states, Ta)

        state_budget = min(self.state_feature_budget, X.shape[1])
        linear_budget = min(self.linear_feature_budget, X.shape[1])
        state_main = _quota_features(normalized_scores, state_budget)
        linear_features = _quota_features(normalized_scores, linear_budget)

        interaction_budget = min(
            self.interaction_feature_budget,
            X.shape[1],
        )
        if interaction_budget >= X.shape[1]:
            interaction_features = np.arange(X.shape[1], dtype=np.int64)
        else:
            interaction_features = _quota_features(
                normalized_scores,
                interaction_budget,
            )
        pair_states = rank_states[:, interaction_features]
        local_pairs = [
            (left, right)
            for left in range(len(interaction_features))
            for right in range(left + 1, len(interaction_features))
        ]
        pair_cache = FiniteStateValueHistogramCache(
            pair_states,
            Ta,
            pairs=local_pairs,
            backend="auto",
            n_jobs=self.n_jobs,
        )
        scored_pairs: list[tuple[float, int, int]] = []
        for left, right in local_pairs:
            raw_left = int(interaction_features[left])
            raw_right = int(interaction_features[right])
            pair_score = self._between_group_histogram_scores(
                pair_cache.pairs[(left, right)],
                target_overall,
                target_total,
            )
            gains = (
                pair_score
                - raw_scores[:, raw_left]
                - raw_scores[:, raw_right]
            )
            normalized_gain = gains / np.maximum(
                direction_variance,
                1e-15,
            )
            score = float(np.max(np.maximum(normalized_gain, 0.0)))
            scored_pairs.append((score, raw_left, raw_right))
        self.histogram_backend_ = (
            "native"
            if main_histogram_backend == "native"
            and pair_cache.backend_ == "native"
            else "python"
        )
        scored_pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
        raw_pairs = [
            (left, right)
            for _, left, right in scored_pairs[: max(self.max_pairs, 0)]
        ]

        helper = FiniteStateRidgeRegressor(
            max_features=max(len(state_main), 1),
            max_bins=self.max_bins,
            max_interaction_features=max(len(interaction_features), 1),
            max_interactions=max(self.max_pairs, 0),
            interaction_order=2,
            reg_lambda=1.0,
            preset="balanced",
            random_state=self.random_state,
        )
        linear_positions = np.arange(len(linear_features), dtype=np.int64)
        train_linear, linear_mean, linear_scale = (
            helper._fit_linear_transform(
                Xa[:, linear_features],
                linear_positions,
                None,
            )
        )
        valid_linear = helper._apply_linear_transform(
            Xv[:, linear_features],
            linear_positions,
            linear_mean,
            linear_scale,
        )

        pair_counts = [0]
        for count in (1, 2, self.max_pairs):
            count = min(int(count), len(raw_pairs))
            if count > 0 and count not in pair_counts:
                pair_counts.append(count)
        levels = tuple(
            dict.fromkeys((min(8, self.max_bins), self.max_bins))
        )
        configs = [RegressionConfig(0, 0, self.head_alpha)]
        configs.extend(
            RegressionConfig(level, count, self.head_alpha)
            for level in levels
            for count in pair_counts
        )
        configs = list(dict.fromkeys(configs))
        validation_rows = []
        pc_scale = np.maximum(np.std(Ta, axis=0), 1e-12)

        for config in configs:
            active_raw_pairs = raw_pairs[: config.n_pairs]
            state_features = [int(value) for value in state_main]
            state_set = set(state_features)
            for left, right in active_raw_pairs:
                if left not in state_set:
                    state_features.append(left)
                    state_set.add(left)
                if right not in state_set:
                    state_features.append(right)
                    state_set.add(right)
            state_features_arr = np.asarray(
                state_features,
                dtype=np.int64,
            )
            local = {
                int(raw): index
                for index, raw in enumerate(state_features_arr)
            }
            local_pairs = [
                (local[left], local[right])
                for left, right in active_raw_pairs
            ]
            train_states = {
                level: values[:, state_features_arr]
                for level, values in Sa_all.items()
            }
            valid_states = {
                level: values[:, state_features_arr]
                for level, values in Sv_all.items()
            }
            if config.max_main_level == 0:
                train_design = train_linear
                valid_design = valid_linear
            else:
                pair_level = max(
                    level
                    for level in self.levels
                    if level <= min(config.max_main_level, 8)
                )
                pair_cards = np.asarray(
                    [
                        encoder.cardinalities_[pair_level][
                            int(state_features_arr[right])
                        ]
                        for _, right in local_pairs
                    ],
                    dtype=np.int32,
                )
                train_codes = helper._codes(
                    train_states,
                    local_pairs,
                    config,
                    pair_cards,
                )
                valid_codes = helper._codes(
                    valid_states,
                    local_pairs,
                    config,
                    pair_cards,
                )
                ref = ReferenceStateEncoder()
                train_state_design = ref.fit_transform(train_codes)
                valid_state_design = ref.transform(valid_codes)
                train_design = sparse.hstack(
                    [train_state_design, train_linear],
                    format="csr",
                )
                valid_design = sparse.hstack(
                    [valid_state_design, valid_linear],
                    format="csr",
                )
            coefficients, intercepts, _ = _fit_multi_target_lsqr_exact(
                train_design,
                Ta,
                alpha=float(config.alpha),
                tol=self.tol,
                max_iter=self.max_iter,
            )
            prediction_pc = valid_design @ coefficients.T + intercepts
            row_error = np.mean(
                ((prediction_pc - Tv) / pc_scale) ** 2,
                axis=1,
            )
            mse = float(row_error.mean())
            se = (
                float(
                    row_error.std(ddof=1)
                    / np.sqrt(len(row_error))
                )
                if len(row_error) > 1
                else 0.0
            )
            validation_rows.append(
                (
                    mse,
                    se,
                    int(train_design.shape[1]),
                    config,
                    state_features_arr,
                    tuple(local_pairs),
                )
            )

        validation_rows.sort(
            key=lambda row: (row[0], row[2], row[3].alpha)
        )
        tolerance = validation_rows[0][1]
        eligible = [
            row
            for row in validation_rows
            if row[0] <= validation_rows[0][0] + tolerance
        ]
        selected = min(
            eligible,
            key=lambda row: (row[2], row[0], row[3].alpha),
        )
        config = selected[3]
        state_features = np.asarray(selected[4], dtype=np.int64)
        pairs = list(selected[5])

        self.encoder_ = NestedQuantileEncoder(
            max_bins=self.max_bins,
            levels=self.levels,
        )
        full_states_all = self.encoder_.fit_transform(X)
        full_states = {
            level: values[:, state_features]
            for level, values in full_states_all.items()
        }
        self.state_feature_idx_ = state_features
        self.linear_feature_idx_ = linear_features
        self.pairs_ = pairs
        self.config_ = config
        self.linear_positions_ = linear_positions
        full_linear, self.linear_mean_, self.linear_scale_ = (
            helper._fit_linear_transform(
                X[:, linear_features],
                linear_positions,
                None,
            )
        )
        if config.max_main_level == 0:
            self.state_encoder_ = None
            self.pair_cardinalities_ = np.empty(0, dtype=np.int32)
            design = full_linear
        else:
            pair_level = max(
                level
                for level in self.levels
                if level <= min(config.max_main_level, 8)
            )
            self.pair_cardinalities_ = np.asarray(
                [
                    self.encoder_.cardinalities_[pair_level][
                        int(state_features[right])
                    ]
                    for _, right in pairs
                ],
                dtype=np.int32,
            )
            codes = helper._codes(
                full_states,
                pairs,
                config,
                self.pair_cardinalities_,
            )
            self.state_encoder_ = ReferenceStateEncoder()
            state_design = self.state_encoder_.fit_transform(codes)
            design = sparse.hstack(
                [state_design, full_linear],
                format="csr",
            )
        self.helper_ = helper
        self.head_ = Ridge(
            alpha=self.head_alpha,
            fit_intercept=True,
            solver="lsqr",
            max_iter=self.max_iter,
            tol=self.tol,
        ).fit(design, Y)
        self.design_dim_ = int(design.shape[1])
        self.validation_scores_ = validation_rows
        self.selection_tolerance_ = float(tolerance)
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        X = np.asarray(matrix, dtype=np.float64)
        linear = self.helper_._apply_linear_transform(
            X[:, self.linear_feature_idx_],
            self.linear_positions_,
            self.linear_mean_,
            self.linear_scale_,
        )
        if self.config_.max_main_level == 0:
            design = linear
        else:
            states = self.encoder_.transform_columns(
                X,
                self.state_feature_idx_,
            )
            codes = self.helper_._codes(
                states,
                self.pairs_,
                self.config_,
                self.pair_cardinalities_,
            )
            state_design = self.state_encoder_.transform(codes)
            design = sparse.hstack(
                [state_design, linear],
                format="csr",
            )
        return self.head_.predict(design)

    @property
    def model_bytes_estimate(self) -> int:
        size = (
            self.encoder_.threshold_bytes_
            + self.state_feature_idx_.nbytes
            + self.linear_feature_idx_.nbytes
            + np.asarray(self.pairs_, dtype=np.int32).nbytes
            + self.pair_cardinalities_.nbytes
            + self.linear_mean_.nbytes
            + self.linear_scale_.nbytes
            + np.asarray(self.head_.coef_, dtype=np.float64).nbytes
            + np.asarray(self.head_.intercept_, dtype=np.float64).nbytes
        )
        if self.state_encoder_ is not None:
            size += self.state_encoder_.offsets_.nbytes
            size += self.state_encoder_.cardinalities_.nbytes
        return int(size)
