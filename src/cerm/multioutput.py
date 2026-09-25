from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import Sequence

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from ._compat import num_samples, safe_indexing
from .estimator import CERMClassifier, _validated_classification_sample_weight
from .program import ConstantBinaryProgram, ProgramBundle
from .params import resolve_estimator_parameters
from .shared_multitask import SharedFiniteStateModel, SharedFiniteStateProgram


class CERMMultiLabelClassifier(ClassifierMixin, BaseEstimator):
    """Independent CERM heads for multilabel indicator targets.

    The first implementation intentionally assumes conditional independence
    between labels.  It shares one public task-neutral program bundle while
    retaining each exact binary CERM semantic program.
    """

    def __init__(
        self,
        estimator: CERMClassifier | None = None,
        *,
        threshold: float | str = 0.5,
        threshold_validation_fraction: float = 0.2,
        threshold_shrinkage: float = 0.25,
        n_jobs: int | None = None,
        output_names: Sequence[str] | None = None,
        representation_strategy: str = "independent",
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.threshold_validation_fraction = threshold_validation_fraction
        self.threshold_shrinkage = threshold_shrinkage
        self.n_jobs = n_jobs
        self.output_names = output_names
        self.representation_strategy = representation_strategy

    def _base_estimator(self) -> CERMClassifier:
        if self.estimator is None:
            return CERMClassifier(multiclass_strategy="error")
        if not isinstance(self.estimator, CERMClassifier):
            raise TypeError("estimator must be a CERMClassifier or None")
        return self.estimator

    def _fit_shared(self, X, target: np.ndarray, names: tuple[str, ...], base: CERMClassifier):
        start = time.perf_counter()
        if self.threshold == "auto":
            raise ValueError("shared multilabel currently supports only fixed thresholds")
        if base.encoder_kind != "quantile":
            raise ValueError("shared multilabel currently requires encoder_kind='quantile'")
        if base.calibration != "none":
            raise ValueError("shared multilabel currently requires calibration='none'")
        reference_index = next(
            (index for index in range(target.shape[1]) if len(np.unique(target[:, index])) == 2),
            None,
        )
        reference = (
            np.zeros(len(target), dtype=np.int32)
            if reference_index is None
            else target[:, reference_index].astype(np.int32, copy=False)
        )
        preparer = clone(base)
        shared = preparer._prepare_target_neutral_input(X, reference)
        if shared is None:
            return None
        resolved = resolve_estimator_parameters(base)
        matrix_full = np.asarray(shared["matrix"], dtype=np.float64)
        adapted_names = tuple(shared["adapted_names"])
        feature_kinds = shared["feature_kinds"]
        feature_cardinalities = shared["feature_cardinalities"]
        n_full = int(matrix_full.shape[1])
        keep = max(1, min(n_full, int(np.ceil(float(resolved.colsample) * n_full))))
        if keep < n_full:
            rng = np.random.default_rng(int(base.random_state) + 32452843)
            feature_indices = np.sort(rng.choice(n_full, size=keep, replace=False)).astype(np.int64)
            matrix = np.ascontiguousarray(matrix_full[:, feature_indices])
            adapted_names = tuple(adapted_names[index] for index in feature_indices)
            if feature_kinds is not None:
                feature_kinds = tuple(feature_kinds[index] for index in feature_indices)
            if feature_cardinalities is not None:
                feature_cardinalities = tuple(feature_cardinalities[index] for index in feature_indices)
        else:
            feature_indices = np.arange(n_full, dtype=np.int64)
            matrix = matrix_full
        thresholds = np.full(target.shape[1], float(self.threshold), dtype=np.float64)
        model = SharedFiniteStateModel(
            task_type="multilabel",
            max_features=int(resolved.max_features),
            pair_feature_limit=int(resolved.max_interaction_features),
            max_bins=int(resolved.max_bins),
            random_state=int(base.random_state),
            feature_kinds=feature_kinds,
            feature_cardinalities=feature_cardinalities,
            threshold=thresholds,
            max_pairs=(int(resolved.max_interactions) if int(resolved.interaction_order) >= 2 else 0),
            fixed_C=resolved.fixed_C,
            search_profile=str(resolved.search_profile),
            selection_subsample=float(resolved.subsample),
            n_jobs=self.n_jobs if self.n_jobs is not None else base.n_jobs,
        ).fit(matrix, target)
        self.estimators_ = [None] * target.shape[1]
        self.thresholds_ = thresholds
        self.output_names_ = names
        self.n_outputs_ = int(target.shape[1])
        self.classes_ = [np.asarray([0, 1], dtype=np.int32) for _ in range(self.n_outputs_)]
        self.adapter_ = shared["adapter"]
        self.n_features_in_ = int(shared["n_features_in"])
        self.feature_names_in_ = shared["feature_names_in"]
        self.input_columns_ = shared["input_columns"]
        self.program_ = SharedFiniteStateProgram(
            model=model,
            adapter=self.adapter_,
            input_columns=self.input_columns_,
            adapted_feature_names=adapted_names,
            feature_indices=tuple(int(index) for index in feature_indices),
            task_type="multilabel",
            output_names=names,
            threshold=tuple(float(value) for value in thresholds),
            library_version=CERMClassifier.VERSION,
            metadata={
                "strategy": "shared-finite-state-logistic",
                "representation": "shared-main-pair",
            },
        )
        self._prediction_program_ = self.program_
        self.fit_seconds_ = float(time.perf_counter() - start)
        self.fit_diagnostics_ = {
            "task_type": "multilabel",
            "strategy": "shared-finite-state-logistic",
            "n_outputs": self.n_outputs_,
            "constant_outputs": int(np.sum(np.all(target == target[:1], axis=0))),
            "shared_preprocessing": True,
            "shared_state_encoder": True,
            "threshold_mode": "fixed",
            "thresholds": thresholds.tolist(),
            "selected_config": model.config_.__dict__,
            "design_dim": int(model.design_dim_),
            "pair_count": int(len(model.pairs_)),
            "model_bytes_estimate": int(self.model_bytes_estimate_),
            "fit_seconds": self.fit_seconds_,
        }
        return self

    def fit(self, X, y, sample_weight=None):
        start = time.perf_counter()
        if self.threshold != "auto":
            if isinstance(self.threshold, str) or not 0 < float(self.threshold) < 1:
                raise ValueError("threshold must be 'auto' or a float in (0, 1)")
        if not 0.05 <= float(self.threshold_validation_fraction) <= 0.5:
            raise ValueError("threshold_validation_fraction must be in [0.05, 0.5]")
        if not 0.0 <= float(self.threshold_shrinkage) <= 1.0:
            raise ValueError("threshold_shrinkage must be in [0, 1]")
        target = np.asarray(y)
        if type_of_target(target, input_name="y", raise_unknown=True) != "multilabel-indicator":
            raise ValueError("CERMMultiLabelClassifier requires a multilabel indicator matrix")
        if target.ndim != 2 or target.shape[1] < 1:
            raise ValueError("y must be a two-dimensional multilabel indicator matrix")
        if len(target) != num_samples(X):
            raise ValueError("X and y have inconsistent lengths")
        weights = _validated_classification_sample_weight(sample_weight, len(target))
        target = target.astype(np.int32, copy=False)
        if not np.isin(target, [0, 1]).all():
            raise ValueError("multilabel targets must contain only 0 and 1")
        if self.representation_strategy not in {"independent", "shared"}:
            raise ValueError("representation_strategy must be 'independent' or 'shared'")
        if self.output_names is None:
            names = tuple(f"label_{index}" for index in range(target.shape[1]))
        else:
            names = tuple(str(name) for name in self.output_names)
            if len(names) != target.shape[1]:
                raise ValueError("output_names length must match y.shape[1]")
        base = self._base_estimator()
        if self.representation_strategy == "shared":
            if weights is not None:
                raise ValueError(
                    "sample_weight is currently supported only with "
                    "representation_strategy='independent'"
                )
            fitted = self._fit_shared(X, target, names, base)
            if fitted is not None:
                return fitted
        reference_index = next(
            (index for index in range(target.shape[1]) if len(np.unique(target[:, index])) == 2),
            None,
        )
        if reference_index is None:
            shared = None
        else:
            preparer = clone(base)
            shared = preparer._prepare_target_neutral_input(
                X, target[:, reference_index].astype(np.int32, copy=False)
            )

        def configured_estimator(index: int):
            estimator = clone(base)
            params = estimator.get_params(deep=False)
            if "multiclass_strategy" in params:
                estimator.set_params(multiclass_strategy="error")
            if hasattr(estimator, "random_state"):
                seed = 0 if estimator.random_state is None else int(estimator.random_state)
                estimator.set_params(random_state=seed + 104729 * index)
            return estimator

        def fit_head(index: int):
            column = target[:, index]
            classes = np.unique(column)
            if len(classes) == 1:
                probability = float(classes[0])
                return None, ConstantBinaryProgram(probability=probability), 0.5

            selected_threshold = float(self.threshold) if self.threshold != "auto" else 0.5
            if self.threshold == "auto":
                indices = np.arange(len(column))
                try:
                    fit_indices, validation_indices = train_test_split(
                        indices,
                        test_size=float(self.threshold_validation_fraction),
                        random_state=1729 + 104729 * index,
                        stratify=column,
                    )
                    calibration_estimator = configured_estimator(index)
                    calibration_estimator.fit(
                        safe_indexing(X, fit_indices), column[fit_indices],
                        sample_weight=None if weights is None else weights[fit_indices],
                    )
                    probability = calibration_estimator.predict_proba(
                        safe_indexing(X, validation_indices)
                    )[:, 1]
                    precision, recall, thresholds = precision_recall_curve(
                        column[validation_indices], probability,
                        sample_weight=None if weights is None else weights[validation_indices],
                    )
                    if len(thresholds):
                        denominator = precision[:-1] + recall[:-1]
                        f1 = np.divide(
                            2.0 * precision[:-1] * recall[:-1],
                            denominator,
                            out=np.zeros_like(denominator),
                            where=denominator > 0.0,
                        )
                        best = int(np.argmax(f1))
                        raw_threshold = float(np.clip(thresholds[best], 0.05, 0.95))
                        shrinkage = float(self.threshold_shrinkage)
                        selected_threshold = float(0.5 + shrinkage * (raw_threshold - 0.5))
                except ValueError:
                    # Rare labels may not support a stratified calibration split.
                    selected_threshold = 0.5

            estimator = configured_estimator(index)
            if shared is None:
                estimator.fit(X, column, sample_weight=weights)
            else:
                estimator._fit_binary_from_prepared(shared, column, sample_weight=weights)
            return estimator, estimator.program_, selected_threshold

        if self.n_jobs in (None, 1):
            fitted = [fit_head(index) for index in range(target.shape[1])]
        else:
            fitted = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(fit_head)(index) for index in range(target.shape[1])
            )
        self.estimators_ = [row[0] for row in fitted]
        programs = tuple(row[1] for row in fitted)
        self.thresholds_ = np.asarray([row[2] for row in fitted], dtype=np.float64)
        self.output_names_ = names
        self.n_outputs_ = int(target.shape[1])
        self.classes_ = [np.asarray([0, 1], dtype=np.int32) for _ in range(self.n_outputs_)]
        self.program_ = ProgramBundle(
            programs=programs,
            task_type="multilabel",
            output_names=names,
            threshold=tuple(float(value) for value in self.thresholds_),
            library_version=CERMClassifier.VERSION,
            metadata={
                "strategy": "independent-binary-heads",
                "preprocessing": "shared-target-neutral" if shared is not None else "independent-target-aware",
            },
        )
        self._prediction_program_ = self.program_
        self.adapter_ = shared["adapter"] if shared is not None else None
        first = next((estimator for estimator in self.estimators_ if estimator is not None), None)
        if first is not None:
            self.n_features_in_ = first.n_features_in_
            self.feature_names_in_ = getattr(first, "feature_names_in_", None)
            self.input_columns_ = first.input_columns_
        else:
            shape = getattr(X, "shape", None)
            self.n_features_in_ = int(shape[1]) if shape is not None and len(shape) == 2 else None
            self.feature_names_in_ = None
            self.input_columns_ = None
        self.fit_seconds_ = float(time.perf_counter() - start)
        self.fit_diagnostics_ = {
            "task_type": "multilabel",
            "strategy": "independent-binary-heads",
            "n_outputs": self.n_outputs_,
            "constant_outputs": int(sum(estimator is None for estimator in self.estimators_)),
            "shared_preprocessing": bool(shared is not None),
            "threshold_mode": "auto" if self.threshold == "auto" else "fixed",
            "thresholds": self.thresholds_.tolist(),
            "threshold_shrinkage": float(self.threshold_shrinkage),
            "model_bytes_estimate": self.model_bytes_estimate_,
            "fit_seconds": self.fit_seconds_,
            "sample_weighted": weights is not None,
            "sample_weight_sum": None if weights is None else float(weights.sum()),
        }
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "program_")
        return self._prediction_program_.predict_proba(X)

    def decision_function(self, X):
        check_is_fitted(self, "program_")
        return self._prediction_program_.decision_function(X)

    def predict(self, X):
        check_is_fitted(self, "program_")
        return self._prediction_program_.predict(X)

    @property
    def model_bytes_estimate_(self) -> int:
        check_is_fitted(self, "program_")
        return self.program_.model_bytes_estimate

    def optimize(self, target: str = "balanced"):
        check_is_fitted(self, "program_")
        self._prediction_program_ = self.program_.optimize(target)
        return self._prediction_program_

    def compile_native(self, prefix: str | Path):
        check_is_fitted(self, "program_")
        return self.program_.compile_native(prefix)

    def export(self, directory: str | Path) -> Path:
        check_is_fitted(self, "program_")
        return self.program_.export(directory, config=self.get_params(deep=False))

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "program_")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CERMMultiLabelClassifier":
        estimator = joblib.load(path)
        if not isinstance(estimator, cls):
            raise TypeError("serialized object is not a CERMMultiLabelClassifier")
        return estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        tags.target_tags.one_d_labels = False
        tags.target_tags.two_d_labels = True
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.multi_label = True
        tags.input_tags.sparse = False
        return tags
