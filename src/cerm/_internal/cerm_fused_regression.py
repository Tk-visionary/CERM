from __future__ import annotations

"""Internal fused residual regression built on CERM finite-state programs.

The public ``CERMRegressor`` keeps its historical ridge-head default. This
module contains the production-shaped implementation of the research V2
residual distribution head so it can be integrated behind an opt-in public
strategy without depending on the separate research repository.
"""

from dataclasses import dataclass
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d

from .cerm_nested_representation import (
    NestedResidualMaps,
    build_nested_codes,
    candidate_code_columns,
    maximal_code_metadata,
    prepare_nested_residual_maps,
)
from .cerm_state_design import ReferenceStateEncoder
from .cerm_weighted_representation import (
    WeightedNestedQuantileEncoder,
    aggregate_feature_scores,
    frequency_weighted_quantile,
    rank_pairs,
)
from .._compat import num_samples
from ..training_graph import EncodedColumnBank
from ..validation import dense_numeric_matrix, validate_dataframe_schema


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    return frequency_weighted_quantile(values, quantiles, sample_weight)


def _weighted_rmse(
    residual: np.ndarray, sample_weight: np.ndarray | None
) -> float:
    squared = np.asarray(residual, dtype=np.float64) ** 2
    if sample_weight is None:
        return float(np.mean(squared) ** 0.5)
    return float(np.average(squared, weights=sample_weight) ** 0.5)


def integrate_survival(
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


class FusedThresholdHead:
    """One convex low-rank threshold-axis logistic head."""

    def __init__(
        self,
        *,
        spline_knots: int = 4,
        C: float = 0.05,
        max_iter: int = 250,
        tol: float = 2e-5,
    ):
        self.spline_knots = int(spline_knots)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _fit_basis(
        self, thresholds: np.ndarray, lower: float, upper: float
    ) -> np.ndarray:
        u = ((thresholds - lower) / max(upper - lower, 1e-12))[:, None]
        n_knots = min(self.spline_knots, max(2, len(thresholds) - 1))
        self.spline_ = SplineTransformer(
            n_knots=n_knots,
            degree=2,
            include_bias=False,
            extrapolation="linear",
        )
        return np.asarray(self.spline_.fit_transform(u), dtype=np.float64)

    def _basis(
        self, thresholds: np.ndarray, lower: float, upper: float
    ) -> np.ndarray:
        u = ((thresholds - lower) / max(upper - lower, 1e-12))[:, None]
        return np.asarray(self.spline_.transform(u), dtype=np.float64)

    def fit(
        self,
        design,
        labels: np.ndarray,
        thresholds: np.ndarray,
        lower: float,
        upper: float,
        sample_weight: np.ndarray | None = None,
    ) -> "FusedThresholdHead":
        design = sparse.csr_matrix(design, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        basis = self._fit_basis(thresholds, lower, upper)
        n_rows, n_thresholds = target.shape
        design_dim = int(design.shape[1])
        rank = int(basis.shape[1])
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        if weights is not None and len(weights) != n_rows:
            raise ValueError("sample_weight length mismatch")

        if weights is None:
            l2 = 1.0 / max(self.C * n_rows * n_thresholds, 1e-12)
            row_scale = None
        else:
            total_weight = float(weights.sum())
            l2 = 1.0 / max(self.C * total_weight * n_thresholds, 1e-12)
            row_scale = weights[:, None] / max(total_weight * n_thresholds, 1e-12)

        def unpack(theta):
            intercept = float(theta[0])
            threshold_coef = theta[1 : 1 + rank]
            state_coef = theta[1 + rank :].reshape(design_dim, rank)
            return intercept, threshold_coef, state_coef

        def objective_gradient(theta):
            intercept, threshold_coef, state_coef = unpack(theta)
            latent = np.asarray(design @ state_coef) + threshold_coef[None, :]
            logits = latent @ basis.T + intercept
            point_loss = np.logaddexp(0.0, logits) - target * logits
            if row_scale is None:
                loss = float(np.mean(point_loss))
                error = (_sigmoid(logits) - target) / (n_rows * n_thresholds)
            else:
                loss = float(np.sum(point_loss * row_scale))
                error = (_sigmoid(logits) - target) * row_scale
            loss += 0.5 * l2 * (
                float(np.dot(threshold_coef, threshold_coef))
                + float(np.sum(state_coef * state_coef))
            )
            latent_gradient = error @ basis
            grad_intercept = float(error.sum())
            grad_threshold = latent_gradient.sum(axis=0) + l2 * threshold_coef
            grad_state = np.asarray(design.T @ latent_gradient) + l2 * state_coef
            gradient = np.concatenate(
                [[grad_intercept], grad_threshold, grad_state.ravel()]
            )
            return loss, gradient

        initial = np.zeros(1 + rank + design_dim * rank, dtype=np.float64)
        result = minimize(
            objective_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "gtol": 1e-6,
                "maxls": 30,
            },
        )
        self.optimize_result_ = result
        self.intercept_, self.threshold_coef_, self.state_coef_ = unpack(result.x)
        self.thresholds_ = np.asarray(thresholds, dtype=np.float64).copy()
        self.lower_ = float(lower)
        self.upper_ = float(upper)
        self.design_dim_ = design_dim
        self.threshold_basis_dim_ = rank
        self.parameter_bytes_ = int(result.x.nbytes)
        return self

    def predict_survival(self, design) -> np.ndarray:
        design = sparse.csr_matrix(design, dtype=np.float64)
        basis = self._basis(self.thresholds_, self.lower_, self.upper_)
        latent = np.asarray(design @ self.state_coef_) + self.threshold_coef_[None, :]
        return _sigmoid(latent @ basis.T + self.intercept_)

    def predict_residual(self, design) -> np.ndarray:
        return integrate_survival(
            self.lower_,
            self.upper_,
            self.thresholds_,
            self.predict_survival(design),
        )


@dataclass(frozen=True)
class FusedRepresentationConfig:
    max_main_level: int
    n_pairs: int
    n_fine_pairs: int = 0


@dataclass
class FittedNestedRepresentation:
    encoder: WeightedNestedQuantileEncoder
    feature_idx: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    fine_pairs: tuple[tuple[int, int], ...]
    levels: tuple[int, ...]
    max_bins: int
    max_main_level: int
    maps: NestedResidualMaps
    state_encoder: ReferenceStateEncoder

    def transform(self, X: np.ndarray) -> sparse.csr_matrix:
        all_states = self.encoder.transform(np.asarray(X, dtype=np.float64))
        states = {
            level: values[:, self.feature_idx]
            for level, values in all_states.items()
        }
        codes = build_nested_codes(
            states,
            encoder=self.encoder,
            feature_idx=self.feature_idx,
            levels=self.levels,
            max_bins=self.max_bins,
            max_main_level=self.max_main_level,
            pairs=self.pairs,
            fine_pairs=self.fine_pairs,
            maps=self.maps,
            dtype=np.int32,
        )
        return self.state_encoder.transform(codes)


class RawMultiResolutionFusedResidual:
    """Conservative raw 4/8/16 fused residual branch for small samples."""

    def __init__(
        self,
        *,
        n_bins: int = 10,
        resolutions: Sequence[int] = (4, 8, 16),
        pair_resolution: int = 8,
        spline_knots: int = 4,
        max_pairs: int = 6,
        C: float = 0.05,
        max_iter: int = 250,
        tol: float = 2e-5,
    ):
        self.n_bins = int(n_bins)
        self.resolutions = tuple(int(value) for value in resolutions)
        self.pair_resolution = int(pair_resolution)
        self.spline_knots = int(spline_knots)
        self.max_pairs = int(max_pairs)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    @staticmethod
    def _fit_state_set(
        X: np.ndarray,
        bins: int,
        sample_weight: np.ndarray | None = None,
    ):
        edges = []
        states = np.empty(X.shape, dtype=np.int16)
        quantiles = np.arange(1, bins, dtype=np.float64) / bins
        for feature in range(X.shape[1]):
            feature_edges = np.unique(
                _weighted_quantile(X[:, feature], quantiles, sample_weight)
            )
            edges.append(feature_edges)
            states[:, feature] = np.searchsorted(
                feature_edges, X[:, feature], side="right"
            )
        cards = np.asarray([len(edge) + 1 for edge in edges], dtype=np.int16)
        return edges, states, cards

    @staticmethod
    def _transform_set(X: np.ndarray, edges):
        states = np.empty(X.shape, dtype=np.int16)
        for feature, feature_edges in enumerate(edges):
            states[:, feature] = np.searchsorted(
                feature_edges, X[:, feature], side="right"
            )
        return states

    @staticmethod
    def _interaction_score(
        states: np.ndarray,
        cards: np.ndarray,
        target: np.ndarray,
        left: int,
        right: int,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        card_left, card_right = int(cards[left]), int(cards[right])
        state_left, state_right = states[:, left], states[:, right]
        code = state_left.astype(np.int64) * card_right + state_right
        if sample_weight is None:
            overall = float(np.mean(target))
            count_left = np.bincount(state_left, minlength=card_left).astype(float)
            count_right = np.bincount(state_right, minlength=card_right).astype(float)
            count = np.bincount(code, minlength=card_left * card_right).astype(float)
            weighted_target = target
            normalizer = max(len(target), 1)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            normalizer = max(float(weights.sum()), 1e-15)
            overall = float(np.dot(weights, target) / normalizer)
            count_left = np.bincount(
                state_left, weights=weights, minlength=card_left
            ).astype(float)
            count_right = np.bincount(
                state_right, weights=weights, minlength=card_right
            ).astype(float)
            count = np.bincount(
                code, weights=weights, minlength=card_left * card_right
            ).astype(float)
            weighted_target = weights * target
        mean_left = np.divide(
            np.bincount(state_left, weights=weighted_target, minlength=card_left),
            count_left,
            out=np.full(card_left, overall),
            where=count_left > 0,
        )
        mean_right = np.divide(
            np.bincount(state_right, weights=weighted_target, minlength=card_right),
            count_right,
            out=np.full(card_right, overall),
            where=count_right > 0,
        )
        cell = np.divide(
            np.bincount(
                code,
                weights=weighted_target,
                minlength=card_left * card_right,
            ),
            count,
            out=np.full(card_left * card_right, overall),
            where=count > 0,
        )
        left_index = np.repeat(np.arange(card_left), card_right)
        right_index = np.tile(np.arange(card_right), card_left)
        interaction = cell - mean_left[left_index] - mean_right[right_index] + overall
        return float(np.sum(count * interaction * interaction) / normalizer)

    def _fit_states(
        self,
        X: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        self.state_sets_ = {}
        for resolution in self.resolutions:
            self.state_sets_[resolution] = self._fit_state_set(
                X, resolution, sample_weight
            )
        if self.pair_resolution not in self.state_sets_:
            self.state_sets_[self.pair_resolution] = self._fit_state_set(
                X, self.pair_resolution, sample_weight
            )
        _, states, cards = self.state_sets_[self.pair_resolution]
        scores = []
        for left in range(X.shape[1]):
            for right in range(left + 1, X.shape[1]):
                scores.append(
                    (
                        self._interaction_score(
                            states,
                            cards,
                            target,
                            left,
                            right,
                            sample_weight,
                        ),
                        left,
                        right,
                    )
                )
        scores.sort(reverse=True)
        self.pairs_ = [
            (left, right) for _, left, right in scores[: self.max_pairs]
        ]

    def _design(
        self,
        X: np.ndarray,
        *,
        fit: bool = False,
        target: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> sparse.csr_matrix:
        X = np.asarray(X, dtype=np.float64)
        n_rows, n_features = X.shape
        if fit:
            if target is None:
                raise ValueError("target is required when fitting raw states")
            self._fit_states(X, target, sample_weight)
        blocks = []
        for resolution in self.resolutions:
            edges, fitted_states, cards = self.state_sets_[resolution]
            states = fitted_states if fit else self._transform_set(X, edges)
            offsets = np.r_[0, np.cumsum(cards[:-1])]
            columns = np.concatenate(
                [offsets[j] + states[:, j] for j in range(n_features)]
            )
            rows = np.concatenate([np.arange(n_rows) for _ in range(n_features)])
            blocks.append(
                sparse.csr_matrix(
                    (np.ones(len(rows)), (rows, columns)),
                    shape=(n_rows, int(cards.sum())),
                )
            )
        edges, fitted_states, cards = self.state_sets_[self.pair_resolution]
        states = fitted_states if fit else self._transform_set(X, edges)
        rows: list[int] = []
        columns: list[int] = []
        offset = 0
        for left, right in self.pairs_:
            card_right = int(cards[right])
            code = states[:, left].astype(np.int64) * card_right + states[:, right]
            rows.extend(np.arange(n_rows))
            columns.extend(offset + code)
            offset += int(cards[left]) * card_right
        if offset:
            blocks.append(
                sparse.csr_matrix(
                    (
                        np.ones(len(rows)),
                        (np.asarray(rows), np.asarray(columns)),
                    ),
                    shape=(n_rows, offset),
                )
            )
        return sparse.hstack(blocks, format="csr")

    @staticmethod
    def _thresholds(
        target: np.ndarray,
        n_bins: int,
        sample_weight: np.ndarray | None = None,
    ):
        target = np.asarray(target, dtype=np.float64)
        if sample_weight is None:
            active = target
        else:
            active = target[np.asarray(sample_weight, dtype=np.float64) > 0]
        lower = float(np.min(active))
        upper = float(np.max(active))
        thresholds = np.unique(
            _weighted_quantile(
                target,
                np.arange(1, n_bins, dtype=np.float64) / n_bins,
                sample_weight,
            )
        )
        thresholds = thresholds[(thresholds > lower) & (thresholds < upper)]
        if len(thresholds) < 2:
            raise ValueError(
                "residual target does not support enough non-degenerate thresholds"
            )
        return lower, upper, np.asarray(thresholds, dtype=np.float64)

    def fit(
        self,
        X: np.ndarray,
        residual: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        X = np.asarray(X, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64)
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        self.lower_, self.upper_, self.thresholds_ = self._thresholds(
            residual, self.n_bins, weights
        )
        labels = (residual[:, None] > self.thresholds_[None, :]).astype(np.float64)
        design = self._design(
            X,
            fit=True,
            target=residual,
            sample_weight=weights,
        )
        self.head_ = FusedThresholdHead(
            spline_knots=self.spline_knots,
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
        ).fit(
            design,
            labels,
            self.thresholds_,
            self.lower_,
            self.upper_,
            sample_weight=weights,
        )
        self.state_dim_ = int(design.shape[1])
        return self

    def predict_survival(self, X: np.ndarray) -> np.ndarray:
        return self.head_.predict_survival(self._design(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.head_.predict_residual(self._design(X))


class FusedResidualCERMRegressor:
    """Production-shaped internal implementation of fused regression V2.

    Raw inputs follow the same dense/typed adapter contract as CERMRegressor.
    The estimator remains internal and is intentionally not exported publicly.
    """

    def __init__(
        self,
        *,
        n_bins: int = 10,
        max_features: int = 24,
        max_interaction_features: int = 12,
        max_pairs: int = 4,
        max_fine_pairs: int = 0,
        max_bins: int = 16,
        spline_knots: int = 4,
        validation_fraction: float = 0.22,
        parsimony_rmse_fraction: float = 0.002,
        min_relative_improvement: float = 0.01,
        C_scale: float = 5e-5,
        small_n_threshold: int = 1000,
        small_n_folds: int = 5,
        small_n_min_oof_improvement: float = 0.001,
        raw_max_pairs: int = 6,
        random_state: int = 20260818,
        max_iter: int = 250,
        tol: float = 2e-5,
        categorical_features: Sequence[str] | str | None = "auto",
        embedding_features: Sequence[str] | None = None,
        category_policy: str = "auto",
        max_identity_categories: int = 16,
        category_bins: int = 8,
        category_smoothing: float = 20.0,
        category_identity: str = "state",
        missing_policy: str = "observed",
        feature_kinds: Sequence[str] | None = None,
        feature_cardinalities: Sequence[int | None] | None = None,
    ):
        self.n_bins = int(n_bins)
        self.max_features = int(max_features)
        self.max_interaction_features = int(max_interaction_features)
        self.max_pairs = int(max_pairs)
        self.max_fine_pairs = int(max_fine_pairs)
        self.max_bins = int(max_bins)
        self.spline_knots = int(spline_knots)
        self.validation_fraction = float(validation_fraction)
        self.parsimony_rmse_fraction = float(parsimony_rmse_fraction)
        self.min_relative_improvement = float(min_relative_improvement)
        self.C_scale = float(C_scale)
        self.small_n_threshold = int(small_n_threshold)
        self.small_n_folds = int(small_n_folds)
        self.small_n_min_oof_improvement = float(small_n_min_oof_improvement)
        self.raw_max_pairs = int(raw_max_pairs)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.categorical_features = categorical_features
        self.embedding_features = embedding_features
        self.category_policy = category_policy
        self.max_identity_categories = int(max_identity_categories)
        self.category_bins = int(category_bins)
        self.category_smoothing = float(category_smoothing)
        self.category_identity = category_identity
        self.missing_policy = missing_policy
        self.feature_kinds = feature_kinds
        self.feature_cardinalities = feature_cardinalities

    @staticmethod
    def _infer_categorical(frame: pd.DataFrame) -> tuple[str, ...]:
        return tuple(
            column
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column])
        )

    def _validate_fit_inputs(
        self,
        X,
        y: Iterable,
        sample_weight: Iterable[float] | None,
    ):
        if X is None:
            raise ValueError("X cannot be None")
        if sparse.issparse(X):
            raise TypeError(
                "sparse input is not supported; provide a dense ndarray or pandas DataFrame"
            )
        n_samples = num_samples(X)
        y_array = column_or_1d(y, warn=True).astype(np.float64)
        if len(y_array) != n_samples:
            raise ValueError(
                f"X and y have inconsistent lengths: {n_samples} and {len(y_array)}"
            )
        if not np.isfinite(y_array).all():
            raise ValueError("y contains NaN or infinity")
        if sample_weight is None:
            weight_array = None
        else:
            weight_array = column_or_1d(sample_weight, warn=True).astype(np.float64)
            if len(weight_array) != n_samples:
                raise ValueError(
                    "X and sample_weight have inconsistent lengths: "
                    f"{n_samples} and {len(weight_array)}"
                )
            if not np.isfinite(weight_array).all():
                raise ValueError("sample_weight contains NaN or infinity")
            if np.any(weight_array < 0):
                raise ValueError("sample_weight cannot contain negative values")
            if not np.any(weight_array > 0):
                raise ValueError(
                    "sample_weight cannot be all zero; it must contain at least one positive value"
                )
        target_type = type_of_target(y_array, input_name="y", raise_unknown=True)
        if target_type not in {"continuous", "binary", "multiclass"}:
            raise ValueError(
                "FusedResidualCERMRegressor requires a one-dimensional numeric "
                f"target, got {target_type}"
            )
        if n_samples < 8:
            raise ValueError(
                "FusedResidualCERMRegressor requires at least 8 samples; "
                f"got n_samples={n_samples}"
            )

        self.sample_weighted_ = weight_array is not None
        self.sample_weight_sum_ = (
            None if weight_array is None else float(weight_array.sum())
        )
        self.n_input_samples_ = int(n_samples)
        if weight_array is not None and np.any(weight_array == 0):
            active = weight_array > 0
            if isinstance(X, pd.DataFrame):
                X = X.loc[active]
            else:
                X = np.asarray(X)[active]
            y_array = y_array[active]
            weight_array = weight_array[active]
        if len(y_array) < 8:
            raise ValueError(
                "FusedResidualCERMRegressor requires at least 8 samples with "
                "positive sample_weight"
            )
        if weight_array is not None and np.array_equal(
            weight_array, np.ones(len(weight_array), dtype=np.float64)
        ):
            weight_array = None
        self.n_effective_samples_ = int(len(y_array))
        return X, y_array, weight_array

    def _fit_input_adapter(
        self,
        X,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> np.ndarray:
        self._core_feature_kinds_ = self.feature_kinds
        self._core_feature_cardinalities_ = self.feature_cardinalities
        if isinstance(X, pd.DataFrame):
            if X.columns.has_duplicates:
                raise ValueError("DataFrame columns must be unique")
            frame = X
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            self.n_features_in_ = int(frame.shape[1])
            self.input_columns_ = tuple(frame.columns)
            categorical = (
                self._infer_categorical(frame)
                if self.categorical_features == "auto"
                else tuple(self.categorical_features or ())
            )
            self.categorical_features_ = categorical
            self.embedding_features_ = tuple(self.embedding_features or ())
            if categorical or self.embedding_features_ or frame.isna().any().any():
                from ..regression import RegressionTypedAdapter

                self.adapter_ = RegressionTypedAdapter(
                    categorical_columns=categorical,
                    embedding_columns=self.embedding_features_,
                    category_policy=self.category_policy,
                    max_identity_categories=self.max_identity_categories,
                    category_bins=self.category_bins,
                    category_smoothing=self.category_smoothing,
                    category_identity=self.category_identity,
                    missing_policy=self.missing_policy,
                    random_state=self.random_state,
                )
                adapted = self.adapter_.fit_transform(
                    frame,
                    y,
                    sample_weight=sample_weight,
                )
                matrix = adapted.matrix
                self.adapted_feature_names_ = np.asarray(
                    adapted.feature_names, dtype=object
                )
                self._core_feature_kinds_ = tuple(adapted.feature_kinds)
                self._core_feature_cardinalities_ = tuple(adapted.cardinalities)
            else:
                self.adapter_ = None
                matrix = dense_numeric_matrix(frame)
                self.adapted_feature_names_ = np.asarray(
                    tuple(map(str, frame.columns)), dtype=object
                )
        else:
            matrix = dense_numeric_matrix(X)
            self.feature_names_in_ = None
            self.n_features_in_ = int(matrix.shape[1])
            self.input_columns_ = None
            self.adapter_ = None
            self.categorical_features_ = ()
            self.embedding_features_ = ()
            self.adapted_feature_names_ = np.asarray(
                tuple(f"x{i}" for i in range(matrix.shape[1])), dtype=object
            )
        self.n_adapted_features_ = int(matrix.shape[1])
        return np.ascontiguousarray(matrix, dtype=np.float64)

    def _transform_input(self, X) -> np.ndarray:
        if self.input_columns_ is not None:
            if self.adapter_ is not None:
                frame = validate_dataframe_schema(X, self.input_columns_)
                return np.ascontiguousarray(
                    self.adapter_.transform(frame).matrix,
                    dtype=np.float64,
                )
            return dense_numeric_matrix(X, self.input_columns_)
        matrix = dense_numeric_matrix(X)
        if int(matrix.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features, but "
                "FusedResidualCERMRegressor is expecting "
                f"{self.n_features_in_} features as input"
            )
        return matrix

    def _C(
        self,
        n_rows: int,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        mass = (
            float(n_rows)
            if sample_weight is None
            else float(np.asarray(sample_weight, dtype=np.float64).sum())
        )
        return float(np.clip(self.C_scale * mass, 0.01, 0.1))

    def _make_baseline(self, n_features: int):
        from ..regression import FiniteStateRidgeRegressor

        return FiniteStateRidgeRegressor(
            max_features=min(self.max_features, n_features),
            max_bins=self.max_bins,
            max_interaction_features=min(
                self.max_interaction_features, n_features
            ),
            max_interactions=min(
                self.max_pairs, n_features * (n_features - 1) // 2
            ),
            interaction_order=2,
            reg_lambda="auto",
            preset="balanced",
            subsample=1.0,
            n_jobs=1,
            random_state=self.random_state,
            feature_kinds=self._core_feature_kinds_,
            feature_cardinalities=self._core_feature_cardinalities_,
            include_linear=True,
        )

    @staticmethod
    def _thresholds(
        residual: np.ndarray,
        n_bins: int,
        sample_weight: np.ndarray | None = None,
    ):
        residual = np.asarray(residual, dtype=np.float64)
        if sample_weight is None:
            active = residual
        else:
            active = residual[np.asarray(sample_weight, dtype=np.float64) > 0]
        lower = float(np.min(active))
        upper = float(np.max(active))
        thresholds = np.unique(
            _weighted_quantile(
                residual,
                np.arange(1, n_bins, dtype=np.float64) / n_bins,
                sample_weight,
            )
        )
        thresholds = thresholds[(thresholds > lower) & (thresholds < upper)]
        if len(thresholds) < 2:
            raise ValueError(
                "residual target does not support enough non-degenerate thresholds"
            )
        return lower, upper, np.asarray(thresholds, dtype=np.float64)

    @staticmethod
    def _labels(residual: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        return (residual[:, None] > thresholds[None, :]).astype(np.int32)

    @staticmethod
    def _gamma(
        target: np.ndarray,
        correction: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        if sample_weight is None:
            denominator = float(np.dot(correction, correction))
            numerator = float(np.dot(target, correction))
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            denominator = float(np.dot(weights, correction * correction))
            numerator = float(np.dot(weights, target * correction))
        if denominator <= 1e-12:
            return 0.0
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    def _feature_pair_bank(
        self,
        X_train: np.ndarray,
        labels_train: np.ndarray,
        X_valid: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        levels = tuple(level for level in (4, 8, 16) if level <= self.max_bins)
        ranking_level = 8 if self.max_bins >= 8 else 4
        encoder = WeightedNestedQuantileEncoder(
            max_bins=self.max_bins,
            levels=levels,
            feature_kinds=self._core_feature_kinds_,
            feature_cardinalities=self._core_feature_cardinalities_,
        )
        states_train_all = encoder.fit_transform(
            X_train,
            sample_weight=sample_weight,
        )
        states_valid_all = encoder.transform(X_valid)
        scores = aggregate_feature_scores(
            states_train_all[ranking_level],
            labels_train,
            "multilabel",
            sample_weight=sample_weight,
        )
        feature_idx = np.argsort(-scores, kind="stable")[
            : min(self.max_features, X_train.shape[1])
        ]
        states_train = {
            level: values[:, feature_idx]
            for level, values in states_train_all.items()
        }
        states_valid = {
            level: values[:, feature_idx]
            for level, values in states_valid_all.items()
        }
        pairs = rank_pairs(
            states_train[ranking_level],
            labels_train,
            "multilabel",
            self.max_pairs,
            self.max_interaction_features,
            "mean",
            sample_weight=sample_weight,
        )
        fine_pairs = pairs[: min(self.max_fine_pairs, len(pairs))]
        maps = prepare_nested_residual_maps(
            encoder=encoder,
            feature_idx=feature_idx,
            levels=levels,
            pairs=pairs,
            fine_pairs=fine_pairs,
        )
        train_codes = build_nested_codes(
            states_train,
            encoder=encoder,
            feature_idx=feature_idx,
            levels=levels,
            max_bins=self.max_bins,
            max_main_level=self.max_bins,
            pairs=pairs,
            fine_pairs=fine_pairs,
            maps=maps,
            dtype=np.int32,
        )
        valid_codes = build_nested_codes(
            states_valid,
            encoder=encoder,
            feature_idx=feature_idx,
            levels=levels,
            max_bins=self.max_bins,
            max_main_level=self.max_bins,
            pairs=pairs,
            fine_pairs=fine_pairs,
            maps=maps,
            dtype=np.int32,
        )
        bank = EncodedColumnBank.build(
            train_codes,
            valid_codes,
            ReferenceStateEncoder,
        )
        metadata = maximal_code_metadata(
            levels=levels,
            n_features=len(feature_idx),
            n_pairs=len(pairs),
            n_fine_pairs=len(fine_pairs),
        )
        if len(metadata) != train_codes.shape[1]:
            raise RuntimeError("maximal nested metadata/code mismatch")
        return (
            encoder,
            np.asarray(feature_idx, dtype=np.int64),
            tuple(pairs),
            tuple(fine_pairs),
            maps,
            bank,
            metadata,
            levels,
        )

    def _candidate_configs(
        self, n_pairs: int, n_fine_pairs: int
    ) -> list[FusedRepresentationConfig]:
        levels = [level for level in (8, 16) if level <= self.max_bins]
        configs = [FusedRepresentationConfig(level, 0, 0) for level in levels]
        pair_prefixes = []
        for value in (4, self.max_pairs):
            value = min(int(value), int(n_pairs))
            if value > 0 and value not in pair_prefixes:
                pair_prefixes.append(value)
        configs.extend(
            FusedRepresentationConfig(self.max_bins, value, 0)
            for value in pair_prefixes
        )
        if self.max_bins == 16 and n_pairs > 0 and n_fine_pairs > 0:
            configs.append(
                FusedRepresentationConfig(
                    16,
                    n_pairs,
                    min(self.max_fine_pairs, n_fine_pairs),
                )
            )
        return list(dict.fromkeys(configs))

    def _fit_large_n(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        start = time.perf_counter()
        train_idx, valid_idx = train_test_split(
            np.arange(len(y)),
            test_size=self.validation_fraction,
            random_state=self.random_state + 17,
        )
        weight_train = None if sample_weight is None else sample_weight[train_idx]
        weight_valid = None if sample_weight is None else sample_weight[valid_idx]
        selection_baseline = self._make_baseline(X.shape[1]).fit(
            X[train_idx],
            y[train_idx],
            sample_weight=weight_train,
        )
        residual_train = y[train_idx] - selection_baseline.predict(X[train_idx])
        lower, upper, thresholds = self._thresholds(
            residual_train,
            self.n_bins,
            weight_train,
        )
        labels_train = self._labels(residual_train, thresholds)
        (
            encoder,
            feature_idx,
            pairs,
            fine_pairs,
            maps,
            bank,
            metadata,
            levels,
        ) = self._feature_pair_bank(
            X[train_idx],
            labels_train,
            X[valid_idx],
            weight_train,
        )
        configs = self._candidate_configs(len(pairs), len(fine_pairs))
        baseline_valid = selection_baseline.predict(X[valid_idx])
        baseline_rmse = _weighted_rmse(y[valid_idx] - baseline_valid, weight_valid)
        rows = [
            {
                "kind": "current_mean",
                "config": None,
                "rmse": baseline_rmse,
                "design_dim": 0,
                "gamma": 0.0,
            }
        ]
        target_residual = y[valid_idx] - baseline_valid
        for config in configs:
            columns = candidate_code_columns(
                metadata,
                max_main_level=config.max_main_level,
                n_pairs=config.n_pairs,
                n_fine_pairs=config.n_fine_pairs,
            )
            design_train, design_valid = bank.view(columns)
            head = FusedThresholdHead(
                spline_knots=self.spline_knots,
                C=self._C(len(train_idx), weight_train),
                max_iter=self.max_iter,
                tol=self.tol,
            ).fit(
                design_train,
                labels_train,
                thresholds,
                lower,
                upper,
                sample_weight=weight_train,
            )
            correction = head.predict_residual(design_valid)
            gamma = self._gamma(target_residual, correction, weight_valid)
            prediction = baseline_valid + gamma * correction
            rmse = _weighted_rmse(y[valid_idx] - prediction, weight_valid)
            rows.append(
                {
                    "kind": "fused",
                    "config": config,
                    "rmse": rmse,
                    "design_dim": int(design_train.shape[1]),
                    "gamma": gamma,
                    "success": bool(head.optimize_result_.success),
                    "nit": int(head.optimize_result_.nit),
                }
            )

        best_rmse = min(row["rmse"] for row in rows)
        tolerance = max(1e-12, self.parsimony_rmse_fraction * baseline_rmse)
        eligible = [row for row in rows if row["rmse"] <= best_rmse + tolerance]

        def key(row):
            if row["kind"] == "current_mean":
                return (0, 0, row["rmse"])
            config = row["config"]
            return (
                1 + config.n_pairs + config.n_fine_pairs,
                config.max_main_level,
                row["rmse"],
            )

        selected = min(eligible, key=key)
        if selected["kind"] != "current_mean":
            improvement = (
                baseline_rmse - float(selected["rmse"])
            ) / max(baseline_rmse, 1e-12)
            if improvement < self.min_relative_improvement:
                selected = rows[0]

        self.baseline_ = self._make_baseline(X.shape[1]).fit(
            X,
            y,
            sample_weight=sample_weight,
        )
        residual_full = y - self.baseline_.predict(X)
        self.lower_, self.upper_, self.thresholds_ = self._thresholds(
            residual_full,
            self.n_bins,
            sample_weight,
        )
        self.selected_kind_ = selected["kind"]
        self.selected_config_ = selected["config"]
        self.residual_scale_ = float(selected.get("gamma", 0.0))
        self.representation_ = None
        self.head_ = None
        self.raw_correction_ = None
        self.design_dim_ = 0

        if self.selected_kind_ == "fused":
            labels_full = self._labels(residual_full, self.thresholds_)
            (
                encoder,
                feature_idx,
                pairs,
                fine_pairs,
                maps,
                _,
                _,
                levels,
            ) = self._feature_pair_bank(
                X,
                labels_full,
                X[:0],
                sample_weight,
            )
            config = self.selected_config_
            pairs = tuple(pairs[: config.n_pairs])
            fine_pairs = tuple(fine_pairs[: config.n_fine_pairs])
            maps = prepare_nested_residual_maps(
                encoder=encoder,
                feature_idx=feature_idx,
                levels=levels,
                pairs=pairs,
                fine_pairs=fine_pairs,
            )
            all_states = encoder.transform(X)
            states = {
                level: values[:, feature_idx]
                for level, values in all_states.items()
            }
            codes = build_nested_codes(
                states,
                encoder=encoder,
                feature_idx=feature_idx,
                levels=levels,
                max_bins=self.max_bins,
                max_main_level=config.max_main_level,
                pairs=pairs,
                fine_pairs=fine_pairs,
                maps=maps,
                dtype=np.int32,
            )
            state_encoder = ReferenceStateEncoder()
            design = state_encoder.fit_transform(codes)
            self.head_ = FusedThresholdHead(
                spline_knots=self.spline_knots,
                C=self._C(len(y), sample_weight),
                max_iter=self.max_iter,
                tol=self.tol,
            ).fit(
                design,
                labels_full,
                self.thresholds_,
                self.lower_,
                self.upper_,
                sample_weight=sample_weight,
            )
            self.representation_ = FittedNestedRepresentation(
                encoder=encoder,
                feature_idx=feature_idx,
                pairs=pairs,
                fine_pairs=fine_pairs,
                levels=levels,
                max_bins=self.max_bins,
                max_main_level=config.max_main_level,
                maps=maps,
                state_encoder=state_encoder,
            )
            self.design_dim_ = int(design.shape[1])

        self.selection_rows_ = rows
        self.baseline_validation_rmse_ = baseline_rmse
        self.selected_validation_rmse_ = float(selected["rmse"])
        self.selected_relative_improvement_ = float(
            (baseline_rmse - self.selected_validation_rmse_)
            / max(baseline_rmse, 1e-12)
        )
        self.fit_seconds_ = float(time.perf_counter() - start)
        self.fit_diagnostics_ = {
            "selected_kind": self.selected_kind_,
            "selected_config": (
                None
                if self.selected_config_ is None
                else self.selected_config_.__dict__
            ),
            "selection_mode": "large_n_shared_holdout",
            "residual_scale": self.residual_scale_,
            "baseline_validation_rmse": baseline_rmse,
            "selected_validation_rmse": self.selected_validation_rmse_,
            "selected_relative_improvement": self.selected_relative_improvement_,
            "threshold_binary_solves": 0,
            "fused_candidate_solves": len(configs),
            "final_fused_solve": int(self.selected_kind_ == "fused"),
            "design_dim": self.design_dim_,
            "fit_seconds": self.fit_seconds_,
        }
        return self

    def _make_raw_correction(
        self,
        n_rows: int,
        sample_weight: np.ndarray | None = None,
    ):
        return RawMultiResolutionFusedResidual(
            n_bins=self.n_bins,
            resolutions=(4, 8, 16),
            pair_resolution=8,
            spline_knots=self.spline_knots,
            max_pairs=self.raw_max_pairs,
            C=self._C(n_rows, sample_weight),
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _fit_small_n(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        start = time.perf_counter()
        n_splits = min(self.small_n_folds, max(2, len(y) // 20))
        if n_splits < 2:
            return self._fit_large_n(X, y, sample_weight)
        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state + 170141,
        )
        baseline_oof = np.empty(len(y), dtype=np.float64)
        correction_oof = np.empty(len(y), dtype=np.float64)
        fold_id = np.full(len(y), -1, dtype=np.int16)
        fold_fit_seconds = []
        for fold, (train, valid) in enumerate(cv.split(X)):
            t0 = time.perf_counter()
            weight_train = None if sample_weight is None else sample_weight[train]
            baseline = self._make_baseline(X.shape[1]).fit(
                X[train],
                y[train],
                sample_weight=weight_train,
            )
            train_residual = y[train] - baseline.predict(X[train])
            baseline_oof[valid] = baseline.predict(X[valid])
            correction = self._make_raw_correction(
                len(train), weight_train
            ).fit(
                X[train],
                train_residual,
                sample_weight=weight_train,
            )
            correction_oof[valid] = correction.predict(X[valid])
            fold_id[valid] = fold
            fold_fit_seconds.append(float(time.perf_counter() - t0))
        if np.any(fold_id < 0):
            raise RuntimeError("small-N cross-fit did not cover all rows")

        target_residual = y - baseline_oof
        crossfit_scaled = np.empty(len(y), dtype=np.float64)
        fold_gammas = []
        for fold in range(n_splits):
            held = fold_id == fold
            calibration = ~held
            calibration_weight = (
                None
                if sample_weight is None
                else sample_weight[calibration]
            )
            gamma = self._gamma(
                target_residual[calibration],
                correction_oof[calibration],
                calibration_weight,
            )
            fold_gammas.append(gamma)
            crossfit_scaled[held] = gamma * correction_oof[held]
        baseline_rmse = _weighted_rmse(target_residual, sample_weight)
        corrected_rmse = _weighted_rmse(
            target_residual - crossfit_scaled,
            sample_weight,
        )
        relative_improvement = float(
            (baseline_rmse - corrected_rmse) / max(baseline_rmse, 1e-12)
        )
        promote = bool(
            relative_improvement > self.small_n_min_oof_improvement
        )

        self.baseline_ = self._make_baseline(X.shape[1]).fit(
            X,
            y,
            sample_weight=sample_weight,
        )
        residual_full = y - self.baseline_.predict(X)
        self.lower_, self.upper_, self.thresholds_ = self._thresholds(
            residual_full,
            self.n_bins,
            sample_weight,
        )
        self.selected_kind_ = "raw_fused" if promote else "current_mean"
        self.selected_config_ = None
        self.residual_scale_ = 1.0 if promote else 0.0
        self.raw_correction_ = None
        self.representation_ = None
        self.head_ = None
        self.design_dim_ = 0
        if promote:
            self.raw_correction_ = self._make_raw_correction(
                len(y), sample_weight
            ).fit(
                X,
                residual_full,
                sample_weight=sample_weight,
            )
            self.lower_ = float(self.raw_correction_.lower_)
            self.upper_ = float(self.raw_correction_.upper_)
            self.thresholds_ = np.asarray(
                self.raw_correction_.thresholds_, dtype=np.float64
            )
            self.design_dim_ = int(self.raw_correction_.state_dim_)

        self.baseline_validation_rmse_ = baseline_rmse
        self.selected_validation_rmse_ = corrected_rmse if promote else baseline_rmse
        self.selected_relative_improvement_ = relative_improvement if promote else 0.0
        self.selection_rows_ = [
            {"kind": "current_mean", "rmse": baseline_rmse},
            {
                "kind": "raw_fused",
                "rmse": corrected_rmse,
                "crossfit_relative_improvement": relative_improvement,
                "fold_gammas": tuple(float(value) for value in fold_gammas),
            },
        ]
        self.fit_seconds_ = float(time.perf_counter() - start)
        self.fit_diagnostics_ = {
            "selected_kind": self.selected_kind_,
            "selected_config": None,
            "selection_mode": "small_n_crossfit_raw",
            "residual_scale": self.residual_scale_,
            "baseline_validation_rmse": baseline_rmse,
            "selected_validation_rmse": self.selected_validation_rmse_,
            "selected_relative_improvement": self.selected_relative_improvement_,
            "small_n_gate_improvement": relative_improvement,
            "small_n_gate_threshold": self.small_n_min_oof_improvement,
            "crossfit_folds": n_splits,
            "fold_gammas": tuple(float(value) for value in fold_gammas),
            "threshold_binary_solves": 0,
            "fused_candidate_solves": n_splits + int(promote),
            "final_fused_solve": int(promote),
            "design_dim": self.design_dim_,
            "fit_seconds": self.fit_seconds_,
            "fold_fit_seconds": tuple(fold_fit_seconds),
        }
        return self

    def fit(
        self,
        X,
        y: Iterable,
        sample_weight: Iterable[float] | None = None,
    ):
        X_fit, y_array, weight_array = self._validate_fit_inputs(
            X,
            y,
            sample_weight,
        )
        matrix = self._fit_input_adapter(X_fit, y_array, weight_array)
        if len(y_array) <= self.small_n_threshold:
            result = self._fit_small_n(matrix, y_array, weight_array)
        else:
            result = self._fit_large_n(matrix, y_array, weight_array)
        self.fit_diagnostics_["sample_weighted"] = self.sample_weighted_
        self.fit_diagnostics_["sample_weight_sum"] = self.sample_weight_sum_
        self.fit_diagnostics_["input_samples"] = self.n_input_samples_
        self.fit_diagnostics_["effective_samples"] = self.n_effective_samples_
        self.fit_diagnostics_["typed_adapter"] = self.adapter_ is not None
        return result

    def predict_survival(self, X) -> np.ndarray:
        matrix = self._transform_input(X)
        if self.selected_kind_ == "raw_fused":
            return self.raw_correction_.predict_survival(matrix)
        if self.selected_kind_ == "fused":
            return self.head_.predict_survival(self.representation_.transform(matrix))
        return np.zeros((len(matrix), len(self.thresholds_)), dtype=np.float64)

    def predict(self, X) -> np.ndarray:
        matrix = self._transform_input(X)
        baseline = self.baseline_.predict(matrix)
        if self.selected_kind_ == "current_mean":
            return baseline
        if self.selected_kind_ == "raw_fused":
            return baseline + self.raw_correction_.predict(matrix)
        correction = self.head_.predict_residual(self.representation_.transform(matrix))
        return baseline + self.residual_scale_ * correction
