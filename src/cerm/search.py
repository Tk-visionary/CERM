from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import ParameterGrid, check_cv
from sklearn.utils.validation import check_is_fitted

from ._compat import safe_indexing
from .estimator import CERMClassifier
from .resources import CERMResourceLimitError, CERMResourceWarning
from .params import resolve_estimator_parameters


_CACHE_PARAMETERS = ("ranking_l2", "pair_feature_limit", "max_interaction_features")
_CACHE_RANKING_KINDS = {"newton", "residual_newton", "newton_prefilter_mi"}


@dataclass(frozen=True)
class SearchDiagnostics:
    total_candidates: int
    evaluated_candidates: int
    prefilter_groups: int
    proxy_fits: int
    full_cv_fits: int
    candidate_reduction: float
    proxy_seconds: float
    full_cv_seconds: float
    planned_full_cv_fits: int


class CERMSearchCV(BaseEstimator):
    """Cache-aware hyperparameter search for :class:`CERMClassifier`.

    The search is intentionally two-stage.  Candidate groups that differ only
    in ``ranking_l2`` and ``pair_feature_limit`` can first be scored from one
    finite-state Newton histogram cache per CV fold.  Only the strongest proxy
    candidates are then evaluated by ordinary cross-validation.

    The proxy stage never changes the final score: ``best_score_`` and
    ``best_params_`` are always based on complete estimator fits.  Skipped
    candidates remain visible in ``cv_results_`` with ``prefilter_selected``
    set to ``False`` and NaN test scores.

    Parameters
    ----------
    estimator:
        Base CERM estimator.  ``None`` creates ``CERMClassifier()``.
    param_grid:
        sklearn-compatible parameter grid.
    scoring:
        A single sklearn scoring name or callable.  The default optimizes
        held-out log loss.
    cv:
        Integer or sklearn CV splitter.
    cache_prefilter:
        Enable the Newton-histogram proxy for compatible candidate groups.
    prefilter_top_k:
        Maximum number of candidates retained per compatible group.  ``None``
        uses ``ceil(sqrt(group_size))`` with a minimum of four.
    prefilter_min_candidates:
        Groups smaller than this are evaluated exhaustively.
    cache_parameters:
        Parameters allowed to vary inside a cache-compatible group.
    refit:
        Refit the best candidate on all rows.
    error_score:
        Numeric score used when a candidate fit fails, or ``"raise"``.
    """

    def __init__(
        self,
        estimator: CERMClassifier | None = None,
        param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]] | None = None,
        *,
        scoring: str | Any = "neg_log_loss",
        cv: int | Any = 3,
        cache_prefilter: bool = True,
        prefilter_top_k: int | None = 8,
        prefilter_min_candidates: int = 8,
        cache_parameters: Sequence[str] = _CACHE_PARAMETERS,
        refit: bool = True,
        error_score: float | str = np.nan,
        verbose: int = 0,
        n_jobs: int | None = 1,
        max_full_cv_fits: int | None = 128,
        budget_policy: str = "raise",
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.cache_prefilter = cache_prefilter
        self.prefilter_top_k = prefilter_top_k
        self.prefilter_min_candidates = prefilter_min_candidates
        self.cache_parameters = cache_parameters
        self.refit = refit
        self.error_score = error_score
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.max_full_cv_fits = max_full_cv_fits
        self.budget_policy = budget_policy

    def _validate_parameters(self) -> None:
        if not isinstance(self.cache_prefilter, (bool, np.bool_)):
            raise TypeError("cache_prefilter must be boolean")
        if self.prefilter_top_k is not None:
            if not isinstance(self.prefilter_top_k, (int, np.integer)) or int(self.prefilter_top_k) < 1:
                raise ValueError("prefilter_top_k must be a positive integer or None")
        if not isinstance(self.prefilter_min_candidates, (int, np.integer)) or int(self.prefilter_min_candidates) < 2:
            raise ValueError("prefilter_min_candidates must be an integer >= 2")
        if not isinstance(self.refit, (bool, np.bool_)):
            raise TypeError("refit must be boolean")
        if self.error_score != "raise":
            try:
                float(self.error_score)
            except (TypeError, ValueError) as exc:
                raise ValueError("error_score must be numeric or 'raise'") from exc
        if self.n_jobs is not None:
            if not isinstance(self.n_jobs, (int, np.integer)) or int(self.n_jobs) == 0:
                raise ValueError("n_jobs must be None or a non-zero integer")
        if self.max_full_cv_fits is not None:
            if not isinstance(self.max_full_cv_fits, (int, np.integer)) or int(self.max_full_cv_fits) < 1:
                raise ValueError("max_full_cv_fits must be a positive integer or None")
        if self.budget_policy not in {"ignore", "warn", "raise"}:
            raise ValueError("budget_policy must be one of: ignore, raise, warn")
        cache_parameters = tuple(self.cache_parameters)
        unsupported = set(cache_parameters) - set(_CACHE_PARAMETERS)
        if unsupported:
            raise ValueError(
                "cache_parameters currently supports only: "
                + ", ".join(_CACHE_PARAMETERS)
            )

    def _base_estimator(self) -> CERMClassifier:
        estimator = CERMClassifier() if self.estimator is None else self.estimator
        if not isinstance(estimator, CERMClassifier):
            raise TypeError("CERMSearchCV currently requires a CERMClassifier estimator")
        return estimator

    @staticmethod
    def _candidate_key(params: Mapping[str, Any], cache_parameters: tuple[str, ...]) -> tuple:
        return tuple(sorted((key, repr(value)) for key, value in params.items() if key not in cache_parameters))

    @staticmethod
    def _resolved_param(estimator: CERMClassifier, params: Mapping[str, Any], name: str) -> Any:
        return params.get(name, getattr(estimator, name))

    @classmethod
    def _effective_pair_limit(
        cls,
        estimator: CERMClassifier,
        params: Mapping[str, Any],
    ) -> int:
        candidate = clone(estimator).set_params(**dict(params))
        return int(resolve_estimator_parameters(candidate).max_interaction_features)

    def _group_is_cache_compatible(
        self,
        estimator: CERMClassifier,
        candidates: list[dict[str, Any]],
        cache_parameters: tuple[str, ...],
    ) -> bool:
        if len(candidates) < int(self.prefilter_min_candidates):
            return False
        varying = {
            name
            for name in cache_parameters
            if len({repr(self._resolved_param(estimator, params, name)) for params in candidates}) > 1
        }
        if not varying:
            return False
        ranking_kind = self._resolved_param(estimator, candidates[0], "ranking_kind")
        if ranking_kind not in _CACHE_RANKING_KINDS:
            return False
        return True

    @staticmethod
    def _proxy_value(cache, *, ranking_l2: float, pair_feature_limit: int) -> float:
        ranking = cache.rank_pairs(
            max_pairs=max(0, int(pair_feature_limit)),
            l2=float(ranking_l2),
        )
        if not ranking:
            return 0.0
        gains = np.asarray(
            [max(0.0, float(row["incremental_gain"])) for row in ranking],
            dtype=np.float64,
        )
        # The square-root weighting prevents a large weak dictionary from
        # dominating a smaller set of strong interactions.
        weights = 1.0 / np.sqrt(np.arange(1, len(gains) + 1, dtype=np.float64))
        return float(np.dot(gains, weights))

    def _prefilter(
        self,
        X,
        y,
        estimator: CERMClassifier,
        candidates: list[dict[str, Any]],
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, int, int, float]:
        selected = np.ones(len(candidates), dtype=bool)
        proxy_scores = np.full(len(candidates), np.nan, dtype=np.float64)
        if not self.cache_prefilter:
            return selected, proxy_scores, 0, 0, 0.0

        cache_parameters = tuple(self.cache_parameters)
        groups: dict[tuple, list[int]] = defaultdict(list)
        for index, params in enumerate(candidates):
            groups[self._candidate_key(params, cache_parameters)].append(index)

        prefilter_groups = 0
        proxy_fits = 0
        proxy_start = time.perf_counter()
        for indices in groups.values():
            group = [candidates[index] for index in indices]
            if not self._group_is_cache_compatible(estimator, group, cache_parameters):
                continue
            prefilter_groups += 1
            fold_scores = np.zeros((len(group), len(splits)), dtype=np.float64)
            representative = dict(group[0])
            representative["max_interaction_features"] = max(
                self._effective_pair_limit(estimator, params)
                for params in group
            )
            representative["ranking_l2"] = float(
                np.median(
                    [
                        float(self._resolved_param(estimator, params, "ranking_l2"))
                        for params in group
                    ]
                )
            )

            for fold, (train_index, _) in enumerate(splits):
                X_train = safe_indexing(X, train_index)
                y_train = safe_indexing(y, train_index)
                proxy = clone(estimator).set_params(**representative)
                # Calibration adds no information to G/H candidate ranking and
                # can double proxy cost, so the proxy uses the uncalibrated
                # semantic model.  The final CV retains each candidate's
                # requested calibration setting.
                proxy.set_params(
                    cache_training_statistics=True,
                    calibration="none",
                )
                proxy.fit(X_train, y_train)
                proxy_fits += 1
                cache = proxy.training_cache_
                for local_index, params in enumerate(group):
                    fold_scores[local_index, fold] = self._proxy_value(
                        cache,
                        ranking_l2=float(
                            self._resolved_param(estimator, params, "ranking_l2")
                        ),
                        pair_feature_limit=int(
                            self._effective_pair_limit(estimator, params)
                        ),
                    )

            aggregate = fold_scores.mean(axis=1)
            for local_index, global_index in enumerate(indices):
                proxy_scores[global_index] = aggregate[local_index]

            group_size = len(indices)
            if self.prefilter_top_k is None:
                keep = min(group_size, max(4, int(np.ceil(np.sqrt(group_size)))))
            else:
                keep = min(group_size, int(self.prefilter_top_k))
            order = np.argsort(-aggregate, kind="mergesort")
            keep_local = set(order[:keep].tolist())

            # Preserve the best candidate for each pair budget.  This keeps the
            # proxy from collapsing the search to one large dictionary solely
            # because cumulative gain is monotone in the budget.
            budgets = defaultdict(list)
            for local_index, params in enumerate(group):
                budget = int(
                    self._effective_pair_limit(estimator, params)
                )
                budgets[budget].append(local_index)
            for local_indices in budgets.values():
                best_local = max(local_indices, key=lambda i: aggregate[i])
                keep_local.add(best_local)

            for local_index, global_index in enumerate(indices):
                selected[global_index] = local_index in keep_local

            if self.verbose:
                print(
                    f"CERMSearchCV cache prefilter kept {len(keep_local)}/{group_size} "
                    f"candidates in group {prefilter_groups}"
                )

        return selected, proxy_scores, prefilter_groups, proxy_fits, time.perf_counter() - proxy_start

    def fit(self, X, y):
        self._validate_parameters()
        estimator = self._base_estimator()
        param_grid = {} if self.param_grid is None else self.param_grid
        candidates = [dict(params) for params in ParameterGrid(param_grid)]
        if not candidates:
            candidates = [{}]

        y_array = np.asarray(y)
        cv = check_cv(self.cv, y=y_array, classifier=is_classifier(estimator))
        splits = list(cv.split(X, y_array))
        scorer = check_scoring(estimator, scoring=self.scoring)

        selected, proxy_scores, prefilter_groups, proxy_fits, proxy_seconds = self._prefilter(
            X, y_array, estimator, candidates, splits
        )

        n_candidates = len(candidates)
        n_splits = len(splits)
        planned_full_cv_fits = int(selected.sum()) * n_splits
        if (
            self.max_full_cv_fits is not None
            and planned_full_cv_fits > int(self.max_full_cv_fits)
            and self.budget_policy != "ignore"
        ):
            message = (
                "CERMSearchCV planned full fits "
                f"{planned_full_cv_fits:,} exceed limit {int(self.max_full_cv_fits):,}. "
                "Enable cache_prefilter, narrow the grid, or increase the explicit budget."
            )
            if self.budget_policy == "raise":
                raise CERMResourceLimitError(message)
            warnings.warn(message, CERMResourceWarning, stacklevel=2)
        split_scores = np.full((n_candidates, n_splits), np.nan, dtype=np.float64)
        fit_times = np.full((n_candidates, n_splits), np.nan, dtype=np.float64)
        score_times = np.full((n_candidates, n_splits), np.nan, dtype=np.float64)
        errors: list[str | None] = [None] * n_candidates
        full_cv_fits = 0
        full_start = time.perf_counter()

        def evaluate_job(candidate_index, fold, params, train_index, test_index):
            model = clone(estimator).set_params(**params)
            X_train = safe_indexing(X, train_index)
            y_train = safe_indexing(y_array, train_index)
            X_test = safe_indexing(X, test_index)
            y_test = safe_indexing(y_array, test_index)
            try:
                start = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time = time.perf_counter() - start
                start = time.perf_counter()
                score = float(scorer(model, X_test, y_test))
                score_time = time.perf_counter() - start
                return candidate_index, fold, score, fit_time, score_time, None
            except Exception as exc:  # pragma: no cover - behavior tested externally
                if self.error_score == "raise":
                    raise
                return (
                    candidate_index, fold, float(self.error_score),
                    np.nan, np.nan, repr(exc),
                )

        jobs = []
        for candidate_index, params in enumerate(candidates):
            if not selected[candidate_index]:
                continue
            if self.verbose:
                print(
                    f"CERMSearchCV evaluating candidate {candidate_index + 1}/{n_candidates}: {params}"
                )
            for fold, (train_index, test_index) in enumerate(splits):
                jobs.append((candidate_index, fold, params, train_index, test_index))

        if self.n_jobs in (None, 1):
            outputs = [evaluate_job(*job) for job in jobs]
        else:
            outputs = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(evaluate_job)(*job) for job in jobs
            )
        for candidate_index, fold, score, fit_time, score_time, error in outputs:
            split_scores[candidate_index, fold] = score
            fit_times[candidate_index, fold] = fit_time
            score_times[candidate_index, fold] = score_time
            if error is None:
                full_cv_fits += 1
            else:
                errors[candidate_index] = error

        full_cv_seconds = time.perf_counter() - full_start
        mean_scores = np.full(n_candidates, np.nan, dtype=np.float64)
        std_scores = np.full(n_candidates, np.nan, dtype=np.float64)
        mean_fit = np.full(n_candidates, np.nan, dtype=np.float64)
        std_fit = np.full(n_candidates, np.nan, dtype=np.float64)
        mean_score_time = np.full(n_candidates, np.nan, dtype=np.float64)
        std_score_time = np.full(n_candidates, np.nan, dtype=np.float64)
        selected_indices = np.flatnonzero(selected)
        mean_scores[selected_indices] = np.nanmean(split_scores[selected_indices], axis=1)
        std_scores[selected_indices] = np.nanstd(split_scores[selected_indices], axis=1, ddof=0)
        mean_fit[selected_indices] = np.nanmean(fit_times[selected_indices], axis=1)
        std_fit[selected_indices] = np.nanstd(fit_times[selected_indices], axis=1, ddof=0)
        mean_score_time[selected_indices] = np.nanmean(score_times[selected_indices], axis=1)
        std_score_time[selected_indices] = np.nanstd(score_times[selected_indices], axis=1, ddof=0)

        evaluated = np.flatnonzero(selected & np.isfinite(mean_scores))
        if len(evaluated) == 0:
            raise RuntimeError("all CERMSearchCV candidates failed or were skipped")
        order = evaluated[np.argsort(-mean_scores[evaluated], kind="mergesort")]
        ranks = np.full(n_candidates, np.iinfo(np.int32).max, dtype=np.int32)
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)

        self.cv_results_ = {
            "params": candidates,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
            "rank_test_score": ranks,
            "mean_fit_time": mean_fit,
            "std_fit_time": std_fit,
            "mean_score_time": mean_score_time,
            "std_score_time": std_score_time,
            "prefilter_score": proxy_scores,
            "prefilter_selected": selected,
            "fit_error": np.asarray(errors, dtype=object),
        }
        for fold in range(n_splits):
            self.cv_results_[f"split{fold}_test_score"] = split_scores[:, fold]

        self.best_index_ = int(order[0])
        self.best_params_ = dict(candidates[self.best_index_])
        self.best_score_ = float(mean_scores[self.best_index_])
        self.scorer_ = scorer
        self.n_splits_ = n_splits
        self.n_candidates_ = n_candidates
        self.n_evaluated_candidates_ = int(selected.sum())
        self.prefilter_reduction_ = float(1.0 - selected.mean())
        self.search_diagnostics_ = SearchDiagnostics(
            total_candidates=n_candidates,
            evaluated_candidates=int(selected.sum()),
            prefilter_groups=prefilter_groups,
            proxy_fits=proxy_fits,
            full_cv_fits=full_cv_fits,
            candidate_reduction=self.prefilter_reduction_,
            proxy_seconds=proxy_seconds,
            full_cv_seconds=full_cv_seconds,
            planned_full_cv_fits=planned_full_cv_fits,
        )

        if self.refit:
            start = time.perf_counter()
            self.best_estimator_ = clone(estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y_array)
            self.refit_time_ = time.perf_counter() - start
        return self

    def _best(self) -> CERMClassifier:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_

    def predict(self, X):
        return self._best().predict(X)

    def predict_proba(self, X):
        return self._best().predict_proba(X)

    def decision_function(self, X):
        return self._best().decision_function(X)

    def score(self, X, y):
        return float(self.scorer_(self._best(), X, y))
