from __future__ import annotations

"""Public fused residual V2 regression family and deployment contracts."""

from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._version import __version__
from .params import (
    explain_semantic_parameters,
    format_semantic_parameter_summary,
    semantic_parameter_values,
)
from ._internal.cerm_fused_regression import FusedResidualCERMRegressor
from ._internal.cerm_fused_regression_contracts import (
    CompiledFusedRegressionProgram,
    PortableFusedRegressionProgram as _PortableFusedRegressionProgram,
    compile_typed_fused_regression_native,
    semantic_program_from_typed_fused_regressor,
)


class PortableFusedRegressionProgram(_PortableFusedRegressionProgram):
    """Public portable fused program preserving the fitted raw schema exactly."""

    @classmethod
    def load(cls, directory: str | Path) -> "PortableFusedRegressionProgram":
        # The internal numeric-core loader predates the public estimator and
        # normalizes input labels with ``str``.  The public package-v3 contract
        # must preserve JSON-native DataFrame labels (notably integer labels),
        # just as the fitted estimator does.
        from .io import verify_export

        loaded = _PortableFusedRegressionProgram.load(directory)
        manifest = verify_export(directory)
        input_columns = manifest.get("input_columns")
        return cls(
            core=loaded.core,
            adapter=loaded.adapter,
            input_columns=(
                None if input_columns is None else tuple(input_columns)
            ),
            adapted_feature_names=loaded.adapted_feature_names,
            input_n_features=loaded.input_n_features,
            core_n_features=loaded.core_n_features,
        )


__all__ = [
    "CERMFusedRegressor",
    "PortableFusedRegressionProgram",
    "CompiledFusedRegressionProgram",
]


class CERMFusedRegressor(RegressorMixin, BaseEstimator):
    """Fused residual distribution regressor with the validated V2 capacity.

    This class is the explicit compatibility name for the fused regression
    family.  The ordinary top-level :class:`cerm.CERMRegressor` subclasses this
    estimator and uses the same constructor/statistical semantics.  The former
    finite-state Ridge implementation is available separately as
    :class:`cerm.CERMRidgeRegressor`.

    Fused and Ridge intentionally keep separate parameter vocabularies rather
    than sharing a strategy switch whose controls change meaning by engine.
    """

    VERSION = __version__
    _parameter_surface_kind = "fused_regression"

    def __init__(
        self,
        *,
        n_bins: int = 10,
        max_features: int = 24,
        max_bins: int = 16,
        max_interaction_features: int = 12,
        max_pairs: int = 4,
        random_state: int = 20260818,
        categorical_features: Sequence[str] | str | None = "auto",
        embedding_features: Sequence[str] | None = None,
        category_policy: str = "auto",
        max_identity_categories: int = 16,
        category_bins: int = 8,
        category_smoothing: float = 20.0,
        category_identity: str = "state",
        missing_policy: str = "observed",
    ):
        self.n_bins = n_bins
        self.max_features = max_features
        self.max_bins = max_bins
        self.max_interaction_features = max_interaction_features
        self.max_pairs = max_pairs
        self.random_state = random_state
        self.categorical_features = categorical_features
        self.embedding_features = embedding_features
        self.category_policy = category_policy
        self.max_identity_categories = max_identity_categories
        self.category_bins = category_bins
        self.category_smoothing = category_smoothing
        self.category_identity = category_identity
        self.missing_policy = missing_policy

    def _validate_params(self) -> None:
        integer_positive = {
            "n_bins": self.n_bins,
            "max_features": self.max_features,
            "max_interaction_features": self.max_interaction_features,
            "max_identity_categories": self.max_identity_categories,
            "category_bins": self.category_bins,
        }
        for name, value in integer_positive.items():
            if not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if int(self.n_bins) < 3:
            raise ValueError("n_bins must be at least 3")
        if int(self.category_bins) < 2:
            raise ValueError("category_bins must be at least 2")
        if int(self.max_bins) not in {4, 8, 16}:
            raise ValueError("max_bins must be one of 4, 8, or 16")
        if not isinstance(self.max_pairs, (int, np.integer)) or int(self.max_pairs) < 0:
            raise ValueError("max_pairs must be a non-negative integer")
        if not isinstance(self.random_state, (int, np.integer)):
            raise TypeError("random_state must be an integer")
        if self.category_policy not in {"auto", "identity", "ordered", "newton"}:
            raise ValueError("invalid category_policy")
        if self.category_identity not in {"state", "binary"}:
            raise ValueError("invalid category_identity")
        if self.missing_policy not in {"observed", "always", "none"}:
            raise ValueError("invalid missing_policy")
        if not np.isfinite(self.category_smoothing) or float(self.category_smoothing) < 0:
            raise ValueError("category_smoothing must be finite and non-negative")
        if self.categorical_features not in (None, "auto") and isinstance(
            self.categorical_features, str
        ):
            raise TypeError(
                "categorical_features must be 'auto', None, or a sequence of column names"
            )
        if isinstance(self.embedding_features, str):
            raise TypeError("embedding_features must be None or a sequence of column names")
        if self.embedding_features is not None and len(self.embedding_features):
            raise ValueError(
                "embedding_features are not yet supported by fused regression; "
                "use numeric or categorical DataFrame columns"
            )

    def _make_model(self) -> FusedResidualCERMRegressor:
        self._validate_params()
        return FusedResidualCERMRegressor(
            n_bins=int(self.n_bins),
            max_features=int(self.max_features),
            max_interaction_features=int(self.max_interaction_features),
            max_pairs=int(self.max_pairs),
            # V2-public keeps the validated fine-pair branch disabled. The
            # internal engine can evolve independently until a separate
            # promotion gate justifies exposing this as a stable parameter.
            max_fine_pairs=0,
            max_bins=int(self.max_bins),
            random_state=int(self.random_state),
            categorical_features=self.categorical_features,
            embedding_features=None,
            category_policy=self.category_policy,
            max_identity_categories=int(self.max_identity_categories),
            category_bins=int(self.category_bins),
            category_smoothing=float(self.category_smoothing),
            category_identity=self.category_identity,
            missing_policy=self.missing_policy,
        )

    def fit(
        self,
        X,
        y: Iterable,
        sample_weight: Iterable[float] | None = None,
    ) -> "CERMFusedRegressor":
        self.model_ = self._make_model().fit(X, y, sample_weight=sample_weight)
        self.program_ = semantic_program_from_typed_fused_regressor(self.model_)
        self._prediction_program_ = self.program_

        self.n_features_in_ = int(self.model_.n_features_in_)
        self.feature_names_in_ = self.model_.feature_names_in_
        self.input_columns_ = self.model_.input_columns_
        self.adapter_ = self.model_.adapter_
        self.n_adapted_features_ = int(self.model_.n_adapted_features_)
        self.adapted_feature_names_ = np.asarray(
            self.model_.adapted_feature_names_, dtype=object
        ).copy()
        self.categorical_features_ = tuple(self.model_.categorical_features_)
        self.embedding_features_ = ()
        self.selected_kind_ = str(self.model_.selected_kind_)
        self.selected_config_ = self.model_.selected_config_
        self.fit_seconds_ = float(self.model_.fit_seconds_)
        self.fit_diagnostics_ = dict(self.model_.fit_diagnostics_)
        self.fit_diagnostics_.update(
            {
                "task_type": "regression",
                "engine": "fused-residual-v2",
                "public_estimator": type(self).__name__,
                "public_capacity": {
                    "n_bins": int(self.n_bins),
                    "max_bins": int(self.max_bins),
                    "max_features": int(self.max_features),
                    "max_interaction_features": int(self.max_interaction_features),
                    "max_pairs": int(self.max_pairs),
                },
            }
        )
        return self

    def _check_input_feature_count(self, X) -> None:
        shape = getattr(X, "shape", None)
        if shape is None or len(shape) != 2:
            return
        n_features = int(shape[1])
        if n_features != int(self.n_features_in_):
            raise ValueError(
                f"X has {n_features} features, but {type(self).__name__} is expecting "
                f"{int(self.n_features_in_)} features as input"
            )

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "program_")
        self._check_input_feature_count(X)
        return np.asarray(self._prediction_program_.predict(X), dtype=np.float64)

    def predict_survival(self, X) -> np.ndarray:
        """Return the fitted residual-survival probabilities at V2 thresholds."""
        check_is_fitted(self, "program_")
        self._check_input_feature_count(X)
        return np.asarray(
            self._prediction_program_.predict_survival(X), dtype=np.float64
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "program_")
        return self.adapted_feature_names_.copy()

    @property
    def model_bytes_estimate_(self) -> int:
        check_is_fitted(self, "program_")
        adapter_bytes = int(getattr(self.program_.adapter, "nbytes", 0) or 0)
        core_bytes = sum(
            int(np.asarray(array).nbytes) for array in self.program_.core.arrays.values()
        )
        return int(adapter_bytes + core_bytes)

    def export(self, directory: str | Path) -> Path:
        """Export the fitted portable semantic program."""
        check_is_fitted(self, "program_")
        return self.program_.export(directory)

    def compile_native(self, prefix: str | Path) -> CompiledFusedRegressionProgram:
        """Compile the fitted fused regression predictor for the local platform."""
        check_is_fitted(self, "program_")
        return compile_typed_fused_regression_native(self.model_, prefix)

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "program_")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CERMFusedRegressor":
        estimator = joblib.load(path)
        if not isinstance(estimator, cls):
            raise TypeError(f"serialized object is not a {cls.__name__}")
        return estimator

    def get_user_params(self) -> dict[str, object]:
        """Return the compact fused-regression parameter view."""
        return semantic_parameter_values(self, task="regressor")

    def explain_params(self) -> dict[str, dict[str, object]]:
        return explain_semantic_parameters(self, task="regressor")

    def parameter_summary(self) -> str:
        return format_semantic_parameter_summary(self, task="regressor")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.one_d_labels = True
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags