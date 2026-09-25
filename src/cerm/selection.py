from __future__ import annotations

from dataclasses import asdict
import copy

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ._internal.cerm_hybrid_quotient_block import HybridConfig, HybridQuotientBlockCERM


class CrossFittedHybridCERM(HybridQuotientBlockCERM):
    """Hybrid learner with repeated out-of-fold risk selection.

    Candidate dictionaries are constructed once per fold and reused for every
    configuration.  The final semantic model is fitted once on all rows after
    selection.  This changes training only; inference and compiler IR are the
    same as the ordinary Hybrid learner.
    """

    def __init__(
        self,
        max_features: int = 64,
        pair_feature_limit: int = 24,
        random_state: int = 20260803,
        replacement_objective: str = "balanced",
        cv_folds: int = 3,
        near_tie: float = 0.0015,
        min_improvement: float = 0.0010,
        **core_kwargs,
    ):
        super().__init__(
            max_features=max_features,
            pair_feature_limit=pair_feature_limit,
            random_state=random_state,
            replacement_objective=replacement_objective,
            **core_kwargs,
        )
        if cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if near_tie < 0 or min_improvement < 0:
            raise ValueError("selection tolerances must be non-negative")
        self.cv_folds = int(cv_folds)
        self.near_tie = float(near_tie)
        self.min_improvement = float(min_improvement)

    def _select_config(self, X: np.ndarray, y: np.ndarray) -> HybridConfig:
        X_selection, y_selection, _ = self._selection_sample(X, y, seed_offset=31)
        self.selection_rows_ = int(len(X_selection))
        configs = self._configs(len(y_selection))
        results = {config: {"loss": [], "bytes": []} for config in configs}
        folds = min(self.cv_folds, int(np.bincount(y_selection).min()))
        cv = StratifiedKFold(
            n_splits=max(2, folds),
            shuffle=True,
            random_state=self.random_state,
        )
        splits = list(cv.split(X_selection, y_selection))
        max_ranked_blocks = max(config.n_blocks for config in configs)

        def evaluate_fold(fold, train_idx, valid_idx):
            worker = copy.copy(self)
            cache = worker._hybrid_cache(
                X_selection[train_idx],
                y_selection[train_idx],
                X_selection[valid_idx],
                max_ranked_blocks=max_ranked_blocks,
            )
            worker._prepare_bank_requests(cache, configs)
            design_groups = {}
            for config in configs:
                key = (
                    config.rank_mode,
                    config.n_blocks,
                    float(config.cell_shrinkage),
                )
                design_groups.setdefault(key, []).append(config)
            rows = []
            for group in design_groups.values():
                rows.extend(
                    worker._eval_hybrid_path(
                        cache,
                        y_selection[train_idx],
                        y_selection[valid_idx],
                        group,
                    )
                )
            return fold, rows

        if self.n_jobs in (None, 1):
            fold_outputs = [
                evaluate_fold(fold, train_idx, valid_idx)
                for fold, (train_idx, valid_idx) in enumerate(splits)
            ]
        else:
            fold_outputs = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(evaluate_fold)(fold, train_idx, valid_idx)
                for fold, (train_idx, valid_idx) in enumerate(splits)
            )
        fold_outputs.sort(key=lambda item: item[0])
        for _, fold_rows in fold_outputs:
            for loss, byte_est, config in fold_rows:
                results[config]["loss"].append(loss)
                results[config]["bytes"].append(byte_est)

        rows = []
        for config, values in results.items():
            losses = np.asarray(values["loss"], dtype=float)
            bytes_ = np.asarray(values["bytes"], dtype=float)
            rows.append(
                {
                    "config": config,
                    "mean_loss": float(losses.mean()),
                    "std_loss": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
                    "mean_bytes": float(bytes_.mean()),
                    "fold_losses": losses.tolist(),
                }
            )
        rows.sort(key=lambda row: (row["mean_loss"], row["mean_bytes"]))
        best_loss = rows[0]["mean_loss"]
        eligible = [row for row in rows if row["mean_loss"] <= best_loss + self.near_tie]
        eligible.sort(key=lambda row: (row["mean_bytes"], row["mean_loss"]))
        selected = eligible[0]["config"]

        zero_rows = [row for row in rows if row["config"].n_blocks == 0]
        zero_rows.sort(key=lambda row: row["mean_loss"])
        zero = zero_rows[0]
        selected_row = next(row for row in rows if row["config"] == selected)
        if (
            selected.n_blocks > 0
            and selected_row["mean_loss"] > zero["mean_loss"] - self.min_improvement
        ):
            selected = zero["config"]

        self.cv_selection_results_ = [
            {**{k: v for k, v in row.items() if k != "config"}, **asdict(row["config"])}
            for row in rows
        ]
        self.cv_selected_config_ = selected
        self.cv_zero_loss_ = zero["mean_loss"]
        return selected

    def _fit_final(self, X: np.ndarray, y: np.ndarray, selected: HybridConfig):
        self.selected_hybrid_config_ = selected
        self.primary_selected_config_ = selected
        self.secondary_checked_ = False
        self.secondary_accepted_ = selected.n_blocks > 0

        self.base_ = self._make_backbone().fit(X, y)
        all_states = self.base_.encoder_.transform(X)
        states = {
            level: matrix[:, self.base_.feature_idx_]
            for level, matrix in all_states.items()
        }
        base_codes = self.base_._build_codes_from_states(states)
        base_design = self.base_.oh_.transform(base_codes)
        base_probability = self.base_._probability_from_codes(base_codes)
        if selected.rank_mode == "stable":
            ranked = self._stable_rank_blocks(
                states[4], states[self.block_level], y, base_probability,
                max_results=selected.n_blocks,
            )
        else:
            ranked = self._rank_blocks(
                states[4],
                states[self.block_level],
                y,
                base_probability,
                gain_l2=5.0,
                min_hessian=1.0,
                max_results=selected.n_blocks,
            )
        terms = ranked[: selected.n_blocks]
        self.block_specs_ = []
        block_design = self._block_matrix_fit(
            states[4],
            states[self.block_level],
            base_probability,
            terms,
            selected.cell_shrinkage,
        )
        design = sparse.hstack([base_design, block_design], format="csr")
        self.clf_ = LogisticRegression(
            C=selected.C,
            solver="liblinear",
            max_iter=2500,
            random_state=self.random_state,
        ).fit(design, y)
        self.classes_ = self.clf_.classes_
        self.design_dim_ = int(design.shape[1])
        base_dim = base_design.shape[1]
        self.base_coef_ = self.clf_.coef_.ravel()[:base_dim]
        self.block_coef_ = self.clf_.coef_.ravel()[base_dim:]
        self.intercept_ = float(self.clf_.intercept_[0])
        self._compile_block_tables()
        self._build_replacement_plan_v2(states)
        self.model_bytes_estimate_ = int(
            self.base_.model_bytes_estimate_ + self.replacement_bytes_
        )
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        selected = self._select_config(X, y)
        return self._fit_final(X, y, selected)
