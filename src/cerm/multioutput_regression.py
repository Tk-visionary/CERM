from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from ._compat import num_samples
from ._version import __version__
from .generalized_regression import (
    CERMGeneralizedRegressor,
    _validated_sample_weight,
)
from .program import _select_features
from .regression import CERMRegressor, RegressionTypedAdapter
from .validation import dense_numeric_matrix, validate_dataframe_schema
from ._internal.cerm_multioutput_subspace import (
    MultiOutputSubspaceFiniteState,
    output_subspace,
)


class CERMMultiOutputRegressor(RegressorMixin, BaseEstimator):
    """Multi-output regression with independent or shared CERM representations.

    ``representation_strategy="shared"`` selects one finite-state structure
    using the first principal component of standardized outputs, then fits one
    multi-target Ridge head on the shared design. ``"shared_subspace"`` is an
    experimental opt-in strategy that selects one finite-state structure from a
    retained output subspace before fitting the final multi-target Ridge head.
    ``"independent"`` fits one ordinary :class:`CERMRegressor` per output and
    remains the default.
    """

    VERSION = __version__

    def __init__(
        self,
        estimator: CERMRegressor | None = None,
        *,
        representation_strategy: str = "independent",
        head_alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        output_names: Sequence[str] | None = None,
        n_jobs: int | None = 1,
    ):
        self.estimator = estimator
        self.representation_strategy = representation_strategy
        self.head_alpha = head_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.output_names = output_names
        self.n_jobs = n_jobs

    def _base_estimator(self) -> CERMRegressor:
        if self.estimator is None:
            return CERMRegressor(n_jobs=self.n_jobs)
        if not isinstance(self.estimator, CERMRegressor):
            raise TypeError("estimator must be a CERMRegressor or None")
        return clone(self.estimator)

    @staticmethod
    def _shared_target(
        target: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if sample_weight is None:
            mean = target.mean(axis=0)
            variance = np.mean((target - mean) ** 2, axis=0)
            weighted = target - mean
        else:
            total = float(sample_weight.sum())
            mean = np.sum(sample_weight[:, None] * target, axis=0) / total
            variance = np.sum(
                sample_weight[:, None] * (target - mean) ** 2, axis=0
            ) / total
            weighted = np.sqrt(sample_weight[:, None]) * (target - mean)
        scale = np.sqrt(np.maximum(variance, np.finfo(np.float64).eps))
        standardized = (target - mean) / scale
        weighted_standardized = weighted / scale
        _, _, vt = np.linalg.svd(weighted_standardized, full_matrices=False)
        component = vt[0].astype(np.float64, copy=True)
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1.0
        return standardized @ component, mean, scale, component

    @staticmethod
    def _validate_target(y, n_samples: int) -> np.ndarray:
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        raw = np.asarray(y)
        if np.iscomplexobj(raw):
            raise ValueError("Complex data not supported")
        target = np.asarray(raw, dtype=np.float64)
        if target.ndim != 2 or target.shape[1] < 1:
            raise ValueError("y must be a two-dimensional continuous target matrix")
        if len(target) != n_samples:
            raise ValueError("X and y have inconsistent lengths")
        if not np.isfinite(target).all():
            raise ValueError("y contains NaN or infinity")
        return target

    @staticmethod
    def _prepare_shared_subspace_matrix(
        base: CERMRegressor,
        X,
        target: np.ndarray,
    ) -> np.ndarray:
        """Fit only the historical regression input-adapter/projection contract.

        ``shared_subspace`` needs the same adapted matrix and fitted schema as
        the historical representation-only bootstrap, but does not consume the
        finite-state representation that bootstrap also fits.  This helper
        stops after typed adaptation plus deterministic ``colsample``.  For a
        target-aware adapter, it returns the fitted full-data transform exactly
        like the former ``_fit_matrix_cache_`` boundary.
        """
        base._validate_params()
        if X is None:
            raise ValueError("X cannot be None")
        if sparse.issparse(X):
            raise TypeError(
                "sparse input is not supported; provide a dense ndarray or pandas DataFrame"
            )

        if isinstance(X, pd.DataFrame):
            if X.columns.has_duplicates:
                raise ValueError("DataFrame columns must be unique")
            frame = X
            base.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            base.n_features_in_ = int(frame.shape[1])
            base.input_columns_ = tuple(frame.columns)
            categorical = (
                base._infer_categorical(frame)
                if base.categorical_features == "auto"
                else tuple(base.categorical_features or ())
            )
            base.categorical_features_ = categorical
            base.embedding_features_ = tuple(base.embedding_features or ())
            if categorical or base.embedding_features_ or frame.isna().any().any():
                base.adapter_ = RegressionTypedAdapter(
                    categorical_columns=categorical,
                    embedding_columns=base.embedding_features_,
                    category_policy=base.category_policy,
                    max_identity_categories=int(base.max_identity_categories),
                    category_bins=int(base.category_bins),
                    category_smoothing=float(base.category_smoothing),
                    category_identity=base.category_identity,
                    missing_policy=base.missing_policy,
                    random_state=int(base.random_state),
                )
                adapted = base.adapter_.fit_transform(
                    frame, target, sample_weight=None
                )
                matrix = base.adapter_.transform(frame).matrix
                names = tuple(adapted.feature_names)
            else:
                base.adapter_ = None
                matrix = dense_numeric_matrix(frame)
                names = tuple(map(str, frame.columns))
        else:
            matrix = dense_numeric_matrix(X)
            base.feature_names_in_ = None
            base.n_features_in_ = int(matrix.shape[1])
            base.input_columns_ = None
            base.adapter_ = None
            base.categorical_features_ = ()
            base.embedding_features_ = ()
            names = tuple(f"x{i}" for i in range(matrix.shape[1]))

        base.n_full_adapted_features_ = int(matrix.shape[1])
        keep = max(
            1,
            min(
                base.n_full_adapted_features_,
                int(
                    np.ceil(
                        float(base.colsample) * base.n_full_adapted_features_
                    )
                ),
            ),
        )
        if keep < base.n_full_adapted_features_:
            rng = np.random.default_rng(int(base.random_state) + 32452843)
            indices = np.sort(
                rng.choice(
                    base.n_full_adapted_features_, size=keep, replace=False
                )
            ).astype(np.int64)
            matrix = np.ascontiguousarray(matrix[:, indices])
            names = tuple(names[index] for index in indices)
        else:
            indices = np.arange(
                base.n_full_adapted_features_, dtype=np.int64
            )
        base.adapted_feature_indices_ = indices
        base.n_adapted_features_ = int(matrix.shape[1])
        base.adapted_feature_names_ = np.asarray(names, dtype=object)
        return np.ascontiguousarray(matrix, dtype=np.float64)

    def fit(
        self,
        X,
        y,
        sample_weight: Iterable[float] | None = None,
    ):
        start = time.perf_counter()
        if self.representation_strategy not in {
            "shared",
            "shared_subspace",
            "independent",
        }:
            raise ValueError(
                "representation_strategy must be 'shared', 'shared_subspace', or 'independent'"
            )
        if not np.isfinite(float(self.head_alpha)) or float(self.head_alpha) < 0:
            raise ValueError("head_alpha must be finite and non-negative")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")
        if not np.isfinite(float(self.tol)) or float(self.tol) <= 0:
            raise ValueError("tol must be finite and positive")
        n_samples = num_samples(X)
        target = self._validate_target(y, n_samples)
        weights = _validated_sample_weight(sample_weight, n_samples)
        self.n_outputs_ = int(target.shape[1])
        if self.output_names is None:
            self.output_names_ = tuple(
                f"target_{index}" for index in range(self.n_outputs_)
            )
        else:
            self.output_names_ = tuple(str(name) for name in self.output_names)
            if len(self.output_names_) != self.n_outputs_:
                raise ValueError("output_names length must match y.shape[1]")

        subspace = None
        subspace_fallback = False
        if self.representation_strategy == "shared_subspace":
            if weights is not None:
                raise ValueError(
                    "sample_weight is not yet supported with "
                    "representation_strategy='shared_subspace'"
                )
            subspace = output_subspace(target, rank_cap=6)
            subspace_fallback = subspace.rank == 0

        if self.representation_strategy == "independent" or subspace_fallback:
            estimators = []
            for index in range(self.n_outputs_):
                estimator = self._base_estimator()
                estimator.fit(X, target[:, index], sample_weight=weights)
                estimators.append(estimator)
            self.estimators_ = estimators
            first = estimators[0]
            self.n_features_in_ = first.n_features_in_
            self.feature_names_in_ = getattr(first, "feature_names_in_", None)
            self.input_columns_ = first.input_columns_
            self.fit_seconds_ = float(time.perf_counter() - start)
            self.fit_diagnostics_ = {
                "task_type": "multioutput_regression",
                "strategy": (
                    "shared_subspace_fallback_independent"
                    if subspace_fallback
                    else "independent"
                ),
                "n_outputs": self.n_outputs_,
                "sample_weighted": weights is not None,
                "fit_seconds": self.fit_seconds_,
                "model_bytes_estimate": self.model_bytes_estimate_,
                "subspace_rank": (
                    None if subspace is None else int(subspace.rank)
                ),
                "subspace_mp_edge": (
                    None if subspace is None else float(subspace.mp_edge)
                ),
                "subspace_explained_energy": (
                    None
                    if subspace is None
                    else float(subspace.explained_energy)
                ),
            }
            return self

        if self.representation_strategy == "shared_subspace":
            base = self._base_estimator()
            representation_target, _, _, _ = self._shared_target(target, None)
            matrix = self._prepare_shared_subspace_matrix(
                base, X, representation_target
            )
            max_interactions = (
                24
                if base.max_interactions is None and base.preset == "accurate"
                else 12 if base.max_interactions is None
                else int(base.max_interactions)
            )
            model = MultiOutputSubspaceFiniteState(
                rank=int(subspace.rank),
                components=subspace.components,
                target_mean=subspace.mean,
                target_scale=subspace.scale,
                eigenvalues=subspace.eigenvalues,
                max_bins=int(base.max_bins),
                linear_feature_budget=min(
                    int(base.max_features), matrix.shape[1]
                ),
                state_feature_budget=min(
                    8, int(base.max_features), matrix.shape[1]
                ),
                interaction_feature_budget=min(
                    int(base.max_interaction_features), matrix.shape[1]
                ),
                max_pairs=min(4, max_interactions),
                head_alpha=float(self.head_alpha),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                random_state=int(base.random_state),
                n_jobs=self.n_jobs,
            ).fit(matrix, target)
            self.subspace_model_ = model
            self.estimators_ = None
            self.adapter_ = base.adapter_
            self.input_columns_ = base.input_columns_
            self.feature_names_in_ = base.feature_names_in_
            self.n_features_in_ = base.n_features_in_
            self.adapted_feature_indices_ = base.adapted_feature_indices_
            self.adapted_feature_names_ = base.adapted_feature_names_
            self.n_full_adapted_features_ = base.n_full_adapted_features_
            self.n_adapted_features_ = base.n_adapted_features_
            self.fit_seconds_ = float(time.perf_counter() - start)
            self.fit_diagnostics_ = {
                "task_type": "multioutput_regression",
                "strategy": "shared_subspace_finite_state",
                "n_outputs": self.n_outputs_,
                "sample_weighted": False,
                "subspace_rank": int(subspace.rank),
                "subspace_mp_edge": float(subspace.mp_edge),
                "subspace_explained_energy": float(
                    subspace.explained_energy
                ),
                "subspace_eigenvalues": subspace.eigenvalues.tolist(),
                "state_features": int(len(model.state_feature_idx_)),
                "linear_features": int(len(model.linear_feature_idx_)),
                "pair_count": int(len(model.pairs_)),
                "design_dimension": int(model.design_dim_),
                "histogram_backend": model.histogram_backend_,
                "fit_seconds": self.fit_seconds_,
                "model_bytes_estimate": self.model_bytes_estimate_,
            }
            return self

        base = self._base_estimator()
        representation_target, target_mean, target_scale, component = (
            self._shared_target(target, weights)
        )
        base._representation_only_fit = True
        base.fit(X, representation_target, sample_weight=weights)
        representation = base.model_
        if base._fit_training_design_is_final_:
            state_design = representation._training_state_design_
            linear_design = representation._training_linear_design_
            design = sparse.hstack(
                [state_design, linear_design], format="csr", dtype=np.float64
            )
            state_dim = int(state_design.shape[1])
            linear_dim = int(linear_design.shape[1])
            reused_training_design = True
        else:
            matrix = base._fit_matrix_cache_
            design, state_dim, linear_dim = (
                CERMGeneralizedRegressor._design_from_representation(
                    representation, matrix
                )
            )
            reused_training_design = False

        head = Ridge(
            alpha=float(self.head_alpha),
            fit_intercept=True,
            solver="lsqr",
            max_iter=int(self.max_iter),
            tol=float(self.tol),
        ).fit(design, target, sample_weight=weights)
        coef = np.asarray(head.coef_, dtype=np.float64)
        if coef.ndim == 1:
            coef = coef[None, :]
        coef = coef.T
        intercept = np.asarray(head.intercept_, dtype=np.float64).reshape(-1)
        if coef.shape != (design.shape[1], self.n_outputs_):
            raise RuntimeError("multi-output coefficient layout mismatch")

        self.lookup_ = []
        offset = 0
        if state_dim:
            for cardinality in representation.state_encoder_.cardinalities_:
                table = np.zeros(
                    (int(cardinality), self.n_outputs_), dtype=np.float64
                )
                width = max(int(cardinality) - 1, 0)
                if width:
                    table[1:, :] = coef[offset : offset + width, :]
                offset += width
                self.lookup_.append(table)
        if offset != state_dim:
            raise RuntimeError("finite-state coefficient layout mismatch")
        self.linear_coef_ = coef[state_dim:, :].copy()
        if self.linear_coef_.shape[0] != linear_dim:
            raise RuntimeError("linear coefficient layout mismatch")
        self.intercept_ = intercept
        self.n_iter_ = np.asarray(getattr(head, "n_iter_", 0)).copy()
        self.target_mean_ = target_mean
        self.target_scale_ = target_scale
        self.representation_component_ = component

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
        self.estimators_ = None
        self.adapter_ = base.adapter_
        self.input_columns_ = base.input_columns_
        self.feature_names_in_ = base.feature_names_in_
        self.n_features_in_ = base.n_features_in_
        self.adapted_feature_indices_ = base.adapted_feature_indices_
        self.adapted_feature_names_ = base.adapted_feature_names_
        self.n_full_adapted_features_ = base.n_full_adapted_features_
        self.n_adapted_features_ = base.n_adapted_features_

        self.fit_seconds_ = float(time.perf_counter() - start)
        self.fit_diagnostics_ = {
            "task_type": "multioutput_regression",
            "strategy": "shared_pc1_finite_state",
            "n_outputs": self.n_outputs_,
            "sample_weighted": weights is not None,
            "selected_features": int(len(representation.feature_idx_)),
            "pair_count": int(len(representation.pairs_)),
            "design_dimension": int(design.shape[1]),
            "head_alpha": float(self.head_alpha),
            "training_design_reused": reused_training_design,
            "fit_seconds": self.fit_seconds_,
            "model_bytes_estimate": self.model_bytes_estimate_,
        }
        return self

    def _check_X_schema(self, X) -> None:
        if self.input_columns_ is None:
            shape = getattr(X, "shape", np.asarray(X).shape)
            if len(shape) != 2:
                raise ValueError("Expected 2D array. Reshape your data")
            if int(shape[1]) != self.n_features_in_:
                raise ValueError(
                    f"X has {shape[1]} features, but CERMMultiOutputRegressor "
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

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "fit_diagnostics_")
        if getattr(self, "estimators_", None) is not None:
            return np.column_stack(
                [estimator.predict(X) for estimator in self.estimators_]
            )
        self._check_X_schema(X)
        matrix = self._matrix(X)
        if self.representation_strategy == "shared_subspace":
            return self.subspace_model_.predict(matrix)
        representation = self.representation_model_
        raw = np.broadcast_to(
            self.intercept_, (len(matrix), self.n_outputs_)
        ).copy()
        if representation.config_.max_main_level > 0:
            codes = representation._codes(
                representation._states(matrix),
                representation.pairs_,
                representation.config_,
                representation.pair_cardinalities_,
            )
            for column, table in enumerate(self.lookup_):
                state = codes[:, column]
                valid = (state >= 0) & (state < len(table))
                raw[valid] += table[state[valid]]
        if len(self.linear_coef_):
            selected = matrix[:, representation.feature_idx_]
            linear = (
                selected[:, representation.linear_positions_]
                - representation.linear_mean_
            ) / representation.linear_scale_
            raw += linear @ self.linear_coef_
        return raw

    @property
    def model_bytes_estimate_(self) -> int:
        if getattr(self, "estimators_", None) is not None:
            return int(
                sum(
                    estimator.model_bytes_estimate_
                    for estimator in self.estimators_
                )
            )
        if getattr(self, "subspace_model_", None) is not None:
            return int(
                self.subspace_model_.model_bytes_estimate
                + (0 if self.adapter_ is None else self.adapter_.nbytes)
            )
        representation = self.representation_model_
        return int(
            sum(table.nbytes for table in self.lookup_)
            + self.linear_coef_.nbytes
            + self.intercept_.nbytes
            + self.target_mean_.nbytes
            + self.target_scale_.nbytes
            + self.representation_component_.nbytes
            + representation.encoder_.threshold_bytes_
            + representation.feature_idx_.nbytes
            + np.asarray(representation.pairs_, dtype=np.int32).nbytes
            + representation.pair_cardinalities_.nbytes
            + (0 if self.adapter_ is None else self.adapter_.nbytes)
        )

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "fit_diagnostics_")
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
    def load(cls, path: str | Path) -> "CERMMultiOutputRegressor":
        estimator = joblib.load(path)
        if not isinstance(estimator, cls):
            raise TypeError("serialized object is not a CERMMultiOutputRegressor")
        return estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        tags.target_tags.one_d_labels = False
        tags.target_tags.two_d_labels = True
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags
