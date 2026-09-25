from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
import tempfile
import time
from typing import Iterable, Sequence

import joblib
import numpy as np
from joblib import effective_n_jobs
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_pinball_loss
from sklearn.utils.validation import check_is_fitted, column_or_1d
from threadpoolctl import ThreadpoolController

from ._compat import num_samples
from ._version import __version__
from .objectives import make_objective_backend
from .params import (
    explain_semantic_parameters,
    format_semantic_parameter_summary,
    resolve_semantic_alias,
    semantic_parameter_values,
)
from .program import _select_features
from .regression import CERMRegressor
from .validation import dense_numeric_matrix, validate_dataframe_schema


_SUPPORTED_LOSSES = {"huber", "quantile", "multi_quantile", "poisson", "gamma", "tweedie"}
_THREADPOOL_CONTROLLER = ThreadpoolController()


def _validated_sample_weight(
    sample_weight: Iterable[float] | None,
    n_samples: int,
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = column_or_1d(sample_weight, warn=True).astype(np.float64)
    if len(weights) != n_samples:
        raise ValueError(
            "X and sample_weight have inconsistent lengths: "
            f"{n_samples} and {len(weights)}"
        )
    if not np.isfinite(weights).all():
        raise ValueError("sample_weight contains NaN or infinity")
    if np.any(weights < 0):
        raise ValueError("sample_weight cannot contain negative values")
    if not np.any(weights > 0):
        raise ValueError(
            "sample_weight cannot be all zero; it must contain at least one positive value"
        )
    return weights


def _validated_optional_vector(
    values: Iterable[float] | None,
    n_samples: int,
    *,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = column_or_1d(values, warn=True).astype(np.float64)
    if len(array) != n_samples:
        raise ValueError(
            f"X and {name} have inconsistent lengths: {n_samples} and {len(array)}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


class CERMGeneralizedRegressor(RegressorMixin, BaseEstimator):
    """CERM finite-state representation with generalized regression heads.

    The estimator learns the ordinary CERM finite-state representation once,
    then replaces the squared-error final head with a loss-specific head on the
    same sparse finite-state design.

    Supported losses are ``"huber"``, ``"quantile"``,
    ``"multi_quantile"``, ``"poisson"``, ``"gamma"`` and ``"tweedie"``.
    Multi-quantile prediction
    returns an ``(n_samples, n_quantiles)`` array in ascending quantile order.

    ``offset`` is a fixed additive predictor.  ``exposure`` is supported for
    Poisson regression and is exactly represented by a target/weight
    transformation, so the fitted coefficient remains the rate model.  Both
    can also be supplied to :meth:`predict` for new observations.

    ``representation_mode="auto"`` preserves the full CERM validation search.
    ``representation_mode="linear"`` is an explicit low-cost path that skips
    state/pair validation and fits the selected linear finite-state feature
    projection directly.  It is appropriate only when a linear representation
    is acceptable; it is not a universal replacement for ``"auto"``.
    """

    VERSION = __version__

    def __init__(
        self,
        *,
        loss: str = "huber",
        quantile: float = 0.5,
        quantiles: Sequence[float] | None = None,
        non_crossing: str = "cumulative_max",
        huber_epsilon: float = 1.35,
        head_alpha: float = 1e-4,
        max_iter: int = 500,
        tol: float = 1e-6,
        tweedie_power: float = 1.5,
        representation_mode: str = "auto",
        max_features: int = 64,
        max_bins: int = 16,
        subsample: float = 1.0,
        colsample: float = 1.0,
        n_jobs: int | None = 1,
        search_effort: str | None = None,
        state_detail: str | None = None,
        feature_budget: int | None = None,
        interaction_search_features: int | None = None,
        interaction_budget: int | None = None,
        selection_fraction: float | None = None,
        feature_fraction: float | None = None,
        l2_regularization: str | float | None = None,
        preset: str = "accurate",
        max_interaction_features: int = 24,
        max_interactions: int | None = None,
        interaction_order: int = 2,
        reg_lambda: str | float = "auto",
        random_state: int = 20260803,
        categorical_features: Sequence[str] | str | None = "auto",
        embedding_features: Sequence[str] | None = None,
        category_policy: str = "auto",
        max_identity_categories: int = 16,
        category_bins: int = 8,
        category_smoothing: float = 20.0,
        category_identity: str = "state",
        missing_policy: str = "observed",
        include_linear: bool = True,
    ):
        self.search_effort = search_effort
        self.state_detail = state_detail
        self.feature_budget = feature_budget
        self.interaction_search_features = interaction_search_features
        self.interaction_budget = interaction_budget
        self.selection_fraction = selection_fraction
        self.feature_fraction = feature_fraction
        self.l2_regularization = l2_regularization
        self.loss = loss
        self.quantile = quantile
        self.quantiles = quantiles
        self.non_crossing = non_crossing
        self.huber_epsilon = huber_epsilon
        self.head_alpha = head_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.tweedie_power = tweedie_power
        self.representation_mode = representation_mode
        self.max_features = max_features
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.n_jobs = n_jobs
        self.preset = preset
        self.max_interaction_features = max_interaction_features
        self.max_interactions = max_interactions
        self.interaction_order = interaction_order
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.categorical_features = categorical_features
        self.embedding_features = embedding_features
        self.category_policy = category_policy
        self.max_identity_categories = max_identity_categories
        self.category_bins = category_bins
        self.category_smoothing = category_smoothing
        self.category_identity = category_identity
        self.missing_policy = missing_policy
        self.include_linear = include_linear
        self._semantic_legacy_inputs = {
            "preset": preset,
            "max_bins": max_bins,
            "max_features": max_features,
            "max_interaction_features": max_interaction_features,
            "max_interactions": max_interactions,
            "subsample": subsample,
            "colsample": colsample,
            "reg_lambda": reg_lambda,
        }
        self._refresh_semantic_aliases()

    def _refresh_semantic_aliases(self) -> None:
        conflicts: list[str] = []
        specs = (
            ("search_effort", "preset", "accurate", {"thorough": "accurate", "balanced": "balanced"}),
            ("state_detail", "max_bins", 16, {"coarse": 4, "medium": 8, "fine": 16}),
            ("feature_budget", "max_features", 64, None),
            ("interaction_search_features", "max_interaction_features", 24, None),
            ("interaction_budget", "max_interactions", None, None),
            ("selection_fraction", "subsample", 1.0, None),
            ("feature_fraction", "colsample", 1.0, None),
            ("l2_regularization", "reg_lambda", "auto", None),
        )
        for semantic_name, legacy_name, legacy_default, mapping in specs:
            legacy_value = self._semantic_legacy_inputs[legacy_name]
            try:
                resolved = resolve_semantic_alias(
                    semantic_name=semantic_name,
                    semantic_value=getattr(self, semantic_name),
                    legacy_name=legacy_name,
                    legacy_value=legacy_value,
                    legacy_default=legacy_default,
                    mapping=mapping,
                )
            except ValueError as exc:
                conflicts.append(str(exc))
                resolved = resolve_semantic_alias(
                    semantic_name=semantic_name,
                    semantic_value=getattr(self, semantic_name),
                    legacy_name=legacy_name,
                    legacy_value=legacy_value,
                    legacy_default=legacy_default,
                    mapping=mapping,
                    strict=False,
                )
            setattr(self, legacy_name, resolved)
        self._semantic_conflicts = tuple(conflicts)

    def set_params(self, **params):
        legacy_names = set(self._semantic_legacy_inputs)
        result = super().set_params(**params)
        for name in legacy_names & set(params):
            self._semantic_legacy_inputs[name] = params[name]
        return result

    def get_user_params(self) -> dict[str, object]:
        """Return the compact, recommended parameter view."""
        return semantic_parameter_values(self, task="regressor")

    def explain_params(self) -> dict[str, dict[str, object]]:
        """Return effective values together with plain-language meanings."""
        return explain_semantic_parameters(self, task="regressor")

    def parameter_summary(self) -> str:
        """Return a human-readable summary of the effective model choices."""
        return format_semantic_parameter_summary(self, task="regressor")

    def _validated_quantiles(self) -> tuple[float, ...]:
        if self.loss == "quantile":
            values = (float(self.quantile),)
        elif self.loss == "multi_quantile":
            values = (
                (0.1, 0.5, 0.9)
                if self.quantiles is None
                else tuple(float(value) for value in self.quantiles)
            )
        else:
            return ()
        if not values:
            raise ValueError("quantiles must contain at least one value")
        if any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in values):
            raise ValueError("all quantiles must be finite and strictly between 0 and 1")
        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            raise ValueError("quantiles must be strictly increasing")
        return values

    def _validate_params(self) -> None:
        self._refresh_semantic_aliases()
        if self._semantic_conflicts:
            raise ValueError("; ".join(self._semantic_conflicts))
        if self.loss not in _SUPPORTED_LOSSES:
            raise ValueError(
                f"loss must be one of {sorted(_SUPPORTED_LOSSES)}, got {self.loss!r}"
            )
        self._validated_quantiles()
        if self.non_crossing not in {"none", "cumulative_max"}:
            raise ValueError("non_crossing must be 'none' or 'cumulative_max'")
        if float(self.huber_epsilon) <= 1.0:
            raise ValueError("huber_epsilon must be greater than 1")
        if not np.isfinite(float(self.head_alpha)) or float(self.head_alpha) < 0.0:
            raise ValueError("head_alpha must be finite and non-negative")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(float(self.tol)) or float(self.tol) <= 0.0:
            raise ValueError("tol must be finite and positive")
        if self.loss == "tweedie":
            power = float(self.tweedie_power)
            if not np.isfinite(power) or not 1.0 < power < 2.0:
                raise ValueError("tweedie_power must be strictly between 1 and 2")
        if self.representation_mode not in {"auto", "linear"}:
            raise ValueError("representation_mode must be 'auto' or 'linear'")
        if self.representation_mode == "linear" and not bool(self.include_linear):
            raise ValueError(
                "representation_mode='linear' requires include_linear=True"
            )

    def _make_backend(self):
        quantiles = self._validated_quantiles()
        return make_objective_backend(
            loss=self.loss,
            quantile=float(self.quantile),
            quantiles=quantiles if self.loss == "multi_quantile" else self.quantiles,
            huber_epsilon=float(self.huber_epsilon),
            head_alpha=float(self.head_alpha),
            max_iter=int(self.max_iter),
            tol=float(self.tol),
            tweedie_power=float(self.tweedie_power),
        )

    def _head_thread_limit(self) -> int | None:
        if self.loss not in {"poisson", "gamma", "tweedie"}:
            return None
        if self.n_jobs in (None, 1):
            return 1
        if isinstance(self.n_jobs, (int, np.integer)) and int(self.n_jobs) > 1:
            return int(self.n_jobs)
        return None

    @staticmethod
    def _design_from_representation(
        representation,
        matrix: np.ndarray,
    ) -> tuple[sparse.csr_matrix, int, int]:
        state_design = sparse.csr_matrix((len(matrix), 0), dtype=np.float64)
        if representation.config_.max_main_level > 0:
            selected_states = representation.encoder_.transform_columns(
                matrix, representation.feature_idx_
            )
            codes = representation._codes(
                selected_states,
                representation.pairs_,
                representation.config_,
                representation.pair_cardinalities_,
            )
            state_design = representation.state_encoder_.transform(codes)

        selected = matrix[:, representation.feature_idx_]
        linear_design = representation._apply_linear_transform(
            selected,
            representation.linear_positions_,
            representation.linear_mean_,
            representation.linear_scale_,
        )
        state_dim = int(state_design.shape[1])
        linear_dim = int(linear_design.shape[1])
        if state_dim == 0:
            design = linear_design
        elif linear_dim == 0:
            design = state_design
        else:
            design = sparse.hstack(
                [state_design, linear_design], format="csr", dtype=np.float64
            )
        return design, state_dim, linear_dim

    @staticmethod
    def _fit_solvers(
        solvers, design, target, sample_weight, thread_limit, n_jobs=1
    ):
        context = (
            _THREADPOOL_CONTROLLER.limit(limits=thread_limit)
            if thread_limit is not None
            else None
        )

        def fit_one(solver):
            solver.fit(design, target, sample_weight=sample_weight)
            return solver

        workers = min(max(1, effective_n_jobs(n_jobs)), len(solvers))
        if context is None and workers > 1 and len(solvers) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(fit_one, solvers))
        if context is None:
            return [fit_one(solver) for solver in solvers]
        with context:
            return [fit_one(solver) for solver in solvers]

    def fit(
        self,
        X,
        y: Iterable,
        sample_weight: Iterable[float] | None = None,
        offset: Iterable[float] | None = None,
        exposure: Iterable[float] | None = None,
    ):
        start = time.perf_counter()
        self._validate_params()
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
        if n_samples < 8:
            noun = "sample" if n_samples == 1 else "samples"
            raise ValueError(
                "CERMGeneralizedRegressor requires at least 8 samples; "
                f"got {n_samples} {noun}"
            )
        weights = _validated_sample_weight(sample_weight, n_samples)
        offset_array = _validated_optional_vector(offset, n_samples, name="offset")
        exposure_array = _validated_optional_vector(exposure, n_samples, name="exposure")
        backend = self._make_backend()
        backend.validate_target(y_array)
        fit_plan = backend.prepare_fit(
            y_array,
            weights,
            offset=offset_array,
            exposure=exposure_array,
        )

        base = CERMRegressor(
            max_features=self.max_features,
            max_bins=self.max_bins,
            subsample=self.subsample,
            colsample=self.colsample,
            n_jobs=self.n_jobs,
            preset=self.preset,
            max_interaction_features=self.max_interaction_features,
            max_interactions=self.max_interactions,
            interaction_order=self.interaction_order,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            embedding_features=self.embedding_features,
            category_policy=self.category_policy,
            max_identity_categories=self.max_identity_categories,
            category_bins=self.category_bins,
            category_smoothing=self.category_smoothing,
            category_identity=self.category_identity,
            missing_policy=self.missing_policy,
            include_linear=self.include_linear,
        )
        base._representation_only_fit = True
        if self.representation_mode == "linear":
            base._forced_representation_config = (0, 0, 1.0)
        base.fit(
            X,
            fit_plan.representation_target,
            sample_weight=fit_plan.sample_weight,
        )

        representation = base.model_
        if base._fit_training_design_is_final_:
            state_design = representation._training_state_design_
            linear_design = representation._training_linear_design_
            state_dim = int(state_design.shape[1])
            linear_dim = int(linear_design.shape[1])
            if state_dim == 0:
                design = linear_design
            elif linear_dim == 0:
                design = state_design
            else:
                design = sparse.hstack(
                    [state_design, linear_design], format="csr", dtype=np.float64
                )
            reused_training_design = True
        else:
            matrix = base._fit_matrix_cache_
            design, state_dim, linear_dim = self._design_from_representation(
                representation, matrix
            )
            reused_training_design = False

        solvers = self._fit_solvers(
            backend.make_solvers(),
            design,
            fit_plan.target,
            fit_plan.sample_weight,
            self._head_thread_limit(),
            self.n_jobs,
        )
        if len(solvers) == 1:
            coef_matrix = np.asarray(
                solvers[0].coef_, dtype=np.float64
            ).reshape(-1, 1)
        else:
            coef_matrix = np.column_stack(
                [
                    np.asarray(solver.coef_, dtype=np.float64).reshape(-1)
                    for solver in solvers
                ]
            )
        if coef_matrix.shape[0] != design.shape[1]:
            raise RuntimeError("generalized regression coefficient layout mismatch")
        intercepts = np.asarray(
            [float(np.asarray(solver.intercept_).reshape(-1)[0]) for solver in solvers],
            dtype=np.float64,
        )
        iterations = np.asarray(
            [int(np.asarray(getattr(solver, "n_iter_", 0)).reshape(-1)[0]) for solver in solvers],
            dtype=np.int64,
        )
        n_heads = int(coef_matrix.shape[1])

        lookup_multi = []
        coefficient_offset = 0
        if state_dim:
            for cardinality in representation.state_encoder_.cardinalities_:
                table = np.zeros((int(cardinality), n_heads), dtype=np.float64)
                width = max(int(cardinality) - 1, 0)
                if width:
                    table[1:, :] = coef_matrix[
                        coefficient_offset : coefficient_offset + width, :
                    ]
                coefficient_offset += width
                lookup_multi.append(table)
        if coefficient_offset != state_dim:
            raise RuntimeError("finite-state coefficient layout mismatch")
        linear_multi = coef_matrix[state_dim:, :].copy()
        if linear_multi.shape[0] != linear_dim:
            raise RuntimeError("linear coefficient layout mismatch")

        if n_heads == 1:
            self.lookup_ = [table[:, 0].copy() for table in lookup_multi]
            self.linear_coef_ = linear_multi[:, 0].copy()
            self.intercept_ = float(intercepts[0])
            self.n_iter_ = int(iterations[0])
        else:
            self.lookup_ = lookup_multi
            self.linear_coef_ = linear_multi
            self.intercept_ = intercepts
            self.n_iter_ = iterations
        self.n_heads_ = n_heads
        self.quantiles_ = np.asarray(
            getattr(backend, "quantiles", ()), dtype=np.float64
        )
        if reused_training_design:
            del representation._training_state_design_
            del representation._training_linear_design_
        del base._fit_matrix_cache_
        del base._fit_training_design_is_final_

        representation.lookup_ = []
        representation.linear_coef_ = np.empty(0, dtype=np.float64)
        representation.linear_intercept_ = 0.0
        representation.intercept_ = 0.0
        representation.regressor_ = None
        representation.linear_regressor_ = None
        self.representation_model_ = representation
        self.objective_backend_ = backend
        self.head_solver_name_ = (
            type(solvers[0]).__name__
            if len(solvers) == 1
            else f"{type(solvers[0]).__name__}[{len(solvers)}]"
        )

        self.adapter_ = base.adapter_
        self.input_columns_ = base.input_columns_
        self.feature_names_in_ = base.feature_names_in_
        self.n_features_in_ = base.n_features_in_
        self.adapted_feature_indices_ = base.adapted_feature_indices_
        self.adapted_feature_names_ = base.adapted_feature_names_
        self.n_full_adapted_features_ = base.n_full_adapted_features_
        self.n_adapted_features_ = base.n_adapted_features_
        self.categorical_features_ = base.categorical_features_
        self.embedding_features_ = base.embedding_features_
        self.numeric_prediction_input_indices_ = (
            None
            if self.adapter_ is not None
            else self.adapted_feature_indices_[representation.feature_idx_].copy()
        )

        self.fit_seconds_ = float(time.perf_counter() - start)
        self.model_bytes_estimate_ = int(
            sum(table.nbytes for table in self.lookup_)
            + self.linear_coef_.nbytes
            + np.asarray(self.intercept_).nbytes
            + representation.encoder_.threshold_bytes_
            + representation.feature_idx_.nbytes
            + np.asarray(representation.pairs_, dtype=np.int32).nbytes
            + representation.pair_cardinalities_.nbytes
            + representation.encoder_.direct_state_mask_.nbytes
            + representation.encoder_.direct_state_cardinalities_.nbytes
            + representation.linear_mean_.nbytes
            + representation.linear_scale_.nbytes
            + (0 if self.adapter_ is None else self.adapter_.nbytes)
        )
        self.fit_diagnostics_ = {
            "task_type": "generalized_regression",
            "loss": self.loss,
            "representation_mode": self.representation_mode,
            "head_solver": self.head_solver_name_,
            "head_alpha": float(self.head_alpha),
            "quantile": float(self.quantile) if self.loss == "quantile" else None,
            "quantiles": self.quantiles_.tolist() if len(self.quantiles_) else None,
            "non_crossing": self.non_crossing if self.loss == "multi_quantile" else None,
            "huber_epsilon": (
                float(self.huber_epsilon) if self.loss == "huber" else None
            ),
            "representation_target": (
                (
                    "log1p_y"
                    if offset_array is None and exposure_array is None
                    else "log1p_rate"
                )
                if self.loss in {"poisson", "gamma", "tweedie"}
                else "offset_adjusted_y"
                if offset_array is not None
                else "y"
            ),
            "tweedie_power": (
                float(self.tweedie_power) if self.loss == "tweedie" else None
            ),
            "sample_weighted": weights is not None,
            "effective_sample_weighted": fit_plan.sample_weight is not None,
            "sample_weight_sum": (
                None
                if fit_plan.sample_weight is None
                else float(fit_plan.sample_weight.sum())
            ),
            "offset_used": offset_array is not None,
            "exposure_used": exposure_array is not None,
            "selected_features": int(len(representation.feature_idx_)),
            "pair_count": int(len(representation.pairs_)),
            "state_design_dimension": state_dim,
            "linear_design_dimension": linear_dim,
            "design_dimension": int(design.shape[1]),
            "n_heads": n_heads,
            "selected_config": {
                "max_main_level": representation.config_.max_main_level,
                "n_pairs": representation.config_.n_pairs,
                "alpha": representation.config_.alpha,
            },
            "head_iterations": np.asarray(self.n_iter_).tolist(),
            "head_thread_limit": self._head_thread_limit(),
            "exact_rewrite_passes": [
                "representation_only_final_fit",
                *(
                    ["forced_linear_representation"]
                    if self.representation_mode == "linear"
                    else []
                ),
                *(
                    ["training_design_reuse"]
                    if reused_training_design
                    else ["typed_final_transform_preserved"]
                ),
                *(
                    ["small_head_threadpool_limit"]
                    if self._head_thread_limit() is not None
                    else []
                ),
                *(["shared_multi_quantile_design"] if n_heads > 1 else []),
            ],
            "model_bytes_estimate": self.model_bytes_estimate_,
            "fit_seconds": self.fit_seconds_,
            "native_compile_supported": False,
            "portable_export_supported": False,
        }
        return self

    def _check_X_schema(self, X) -> None:
        if self.input_columns_ is None:
            shape = getattr(X, "shape", np.asarray(X).shape)
            if len(shape) != 2:
                raise ValueError("Expected 2D array. Reshape your data")
            if int(shape[1]) != self.n_features_in_:
                raise ValueError(
                    f"X has {shape[1]} features, but CERMGeneralizedRegressor "
                    f"is expecting {self.n_features_in_} features as input"
                )
        else:
            validate_dataframe_schema(X, self.input_columns_)

    def _matrix(self, X) -> np.ndarray:
        if self.adapter_ is None:
            matrix = dense_numeric_matrix(X, self.input_columns_)
        else:
            frame = validate_dataframe_schema(X, self.input_columns_ or ())
            matrix = self.adapter_.transform(frame).matrix
        return _select_features(matrix, self.adapted_feature_indices_)

    def _raw_predict(self, X) -> np.ndarray:
        check_is_fitted(self, "representation_model_")
        self._check_X_schema(X)
        representation = self.representation_model_
        projected_numeric = self.adapter_ is None
        if projected_numeric:
            full_matrix = dense_numeric_matrix(X, self.input_columns_)
            matrix = np.ascontiguousarray(
                full_matrix[:, self.numeric_prediction_input_indices_],
                dtype=np.float64,
            )
        else:
            matrix = self._matrix(X)
        if representation.config_.max_main_level > 0:
            execution_states = representation._execution_states(
                matrix, projected=projected_numeric
            )
            raw = representation._decision_from_execution_states_with_lookup(
                execution_states, self.lookup_, self.intercept_
            )
        elif self.n_heads_ == 1:
            raw = np.full(len(matrix), float(self.intercept_), dtype=np.float64)
        else:
            raw = np.broadcast_to(
                np.asarray(self.intercept_, dtype=np.float64),
                (len(matrix), self.n_heads_),
            ).copy()
        if len(self.linear_coef_):
            selected = (
                matrix
                if projected_numeric
                else matrix[:, representation.feature_idx_]
            )
            linear = (
                selected[:, representation.linear_positions_]
                - representation.linear_mean_
            ) / representation.linear_scale_
            if self.n_heads_ == 1:
                raw += linear @ self.linear_coef_
            else:
                # Preserve the exact single-head BLAS accumulation order for
                # every quantile while sharing representation construction.
                for head in range(self.n_heads_):
                    raw[:, head] += linear @ self.linear_coef_[:, head]
        return raw

    def predict(
        self,
        X,
        offset: Iterable[float] | None = None,
        exposure: Iterable[float] | None = None,
    ) -> np.ndarray:
        raw = self._raw_predict(X)
        n_samples = len(raw)
        offset_array = _validated_optional_vector(offset, n_samples, name="offset")
        exposure_array = _validated_optional_vector(exposure, n_samples, name="exposure")
        prediction = self.objective_backend_.inverse_link(
            raw, offset=offset_array, exposure=exposure_array
        )
        if self.loss == "multi_quantile" and self.non_crossing == "cumulative_max":
            prediction = np.maximum.accumulate(prediction, axis=1)
        return prediction

    def score(self, X, y, sample_weight=None):
        if self.loss != "multi_quantile":
            return super().score(X, y, sample_weight=sample_weight)
        target = column_or_1d(y, warn=True).astype(np.float64)
        prediction = self.predict(X)
        losses = [
            mean_pinball_loss(
                target,
                prediction[:, index],
                alpha=float(quantile),
                sample_weight=sample_weight,
            )
            for index, quantile in enumerate(self.quantiles_)
        ]
        return -float(np.mean(losses))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "representation_model_")
        return self.adapted_feature_names_.copy()

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "representation_model_")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CERMGeneralizedRegressor":
        estimator = joblib.load(path)
        if not isinstance(estimator, cls):
            raise TypeError("serialized object is not a CERMGeneralizedRegressor")
        return estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        multi = self.loss == "multi_quantile"
        tags.target_tags.one_d_labels = not multi
        tags.target_tags.multi_output = multi
        tags.target_tags.single_output = not multi
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags
