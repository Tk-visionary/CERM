
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

from .cerm_hierarchical_residual import HierarchicalResidualCERM
from .cerm_quotient_block import (
    QuotientBlockCERM, BlockConfig, _block_gain_histograms,
    _iter_block_gain_histograms,
)
from ..training_graph import binary_log_loss, fit_binary_logistic_exact


def _iter_paired_fold_block_gains(
    C4: np.ndarray,
    C16: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    fold_a: np.ndarray,
    fold_b: np.ndarray,
    *,
    feature_limit: int,
    gain_l2: float = 5.0,
    min_hessian: float = 1.0,
    sample_weight: np.ndarray | None = None,
):
    """Yield the historical A/B block gains from one paired histogram scan.

    Rows are concatenated as ``fold_a`` followed by ``fold_b``.  The fold id is encoded as the outer histogram dimension, so each fold receives
    an independent contiguous bin range while additions *within* that range
    occur in exactly the same row order as the two historical separate
    ``np.bincount`` calls.
    This fuses the O(d^2) state-key construction and histogram traversal without
    changing any G/H accumulation order.
    """
    fold_a = np.asarray(fold_a, dtype=np.int64)
    fold_b = np.asarray(fold_b, dtype=np.int64)
    rows = np.concatenate([fold_a, fold_b])
    y_sub = np.asarray(y)[rows]
    p_sub = np.asarray(p, dtype=np.float64)[rows]
    g = np.asarray(y_sub, dtype=np.float64) - p_sub
    h = np.maximum(p_sub * (1.0 - p_sub), 1e-8)
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)[rows]
        g = g * weights
        h = h * weights

    d = min(int(C4.shape[1]), int(feature_limit))
    C4_keys = np.asarray(C4[rows, :d], dtype=np.int64, order="F")
    C16_keys = np.asarray(C16[rows, :d], dtype=np.int64, order="F")
    gate_cards = C4_keys.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    target_cards = C16_keys.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    min_support = np.asarray(
        [
            max(10, int(math.ceil(0.025 * len(fold_a) * 2))),
            max(10, int(math.ceil(0.025 * len(fold_b) * 2))),
        ],
        dtype=np.float64,
    )
    pair_fold_code = np.empty(len(rows), dtype=np.int64)
    support_code = np.empty(len(rows), dtype=np.int64)

    for gate_j in range(d):
        gate_values = C4_keys[:, gate_j]
        gate_card = int(gate_cards[gate_j])
        np.copyto(support_code, gate_values, casting="unsafe")
        support_code[len(fold_a):] += gate_card
        if weights is None:
            support = np.bincount(
                support_code, minlength=gate_card * 2
            ).reshape(2, gate_card)
        else:
            support = np.bincount(
                support_code, weights=weights, minlength=gate_card * 2
            ).reshape(2, gate_card)

        for target_k in range(d):
            if target_k == gate_j:
                continue
            target_card = int(target_cards[target_k])
            np.multiply(gate_values, target_card, out=pair_fold_code)
            np.add(pair_fold_code, C16_keys[:, target_k], out=pair_fold_code)
            base_card = gate_card * target_card
            pair_fold_code[len(fold_a):] += base_card
            total_card = base_card * 2
            G = np.bincount(
                pair_fold_code, weights=g, minlength=total_card
            ).reshape(2, gate_card, target_card)
            H = np.bincount(
                pair_fold_code, weights=h, minlength=total_card
            ).reshape(2, gate_card, target_card)

            # Fold is the outer histogram dimension, so each fold table is
            # already C-contiguous with the exact historical 2D layout.
            fold_tables = ((G[0], H[0]), (G[1], H[1]))
            valid_folds = []
            gain_folds = []
            for fold_index, (G_fold, H_fold) in enumerate(fold_tables):
                total_G = G_fold.sum(axis=1)
                total_H = H_fold.sum(axis=1)
                active = H_fold > 1e-12
                valid_fold = (
                    (support[fold_index] >= min_support[fold_index])
                    & (total_H >= float(min_hessian))
                    & (active.sum(axis=1) > 1)
                )
                parent_gain = total_G * total_G / (total_H + gain_l2)
                child_gain = np.sum(
                    np.where(active, (G_fold * G_fold) / (H_fold + gain_l2), 0.0),
                    axis=1,
                )
                valid_folds.append(valid_fold)
                gain_folds.append(
                    0.5 * np.maximum(0.0, child_gain - parent_gain)
                )
            both = valid_folds[0] & valid_folds[1]
            for gate_state in np.flatnonzero(both):
                yield (
                    (gate_j, int(gate_state), target_k, target_card),
                    float(gain_folds[0][gate_state]),
                    float(gain_folds[1][gate_state]),
                )


@dataclass(frozen=True)
class StableBlockConfig:
    n_blocks: int
    cell_shrinkage: float
    C: float


class StableQuotientBlockCERM(QuotientBlockCERM):
    """Quotient blocks selected by complementary-fold gain stability.

    Selection replaces boosting-style row subsampling with stability in a
    finite operator dictionary. Execution supports exact algebraic replacement
    of multiple gated blocks by one fused pair LUT.
    """

    def __init__(
        self,
        max_features: int = 64,
        pair_feature_limit: int = 24,
        random_state: int = 20260803,
        replacement_objective: str = "balanced",
        **kwargs,
    ):
        super().__init__(
            max_features=max_features,
            pair_feature_limit=pair_feature_limit,
            random_state=random_state,
            **kwargs,
        )
        if replacement_objective not in {"latency", "memory", "balanced", "none"}:
            raise ValueError("invalid replacement objective")
        self.replacement_objective = replacement_objective

    def _gain_items(
        self,
        C4: np.ndarray,
        C16: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        rows: np.ndarray,
        gain_l2: float = 5.0,
        min_hessian: float = 1.0,
        stream: bool = False,
        sample_weight: np.ndarray | None = None,
    ):
        y_sub = y[rows]
        p_sub = p[rows]
        w_sub = None if sample_weight is None else np.asarray(sample_weight)[rows]
        C4_sub = C4[rows]
        C16_sub = C16[rows]
        min_support = max(10, int(math.ceil(0.025 * len(rows) * 2)))
        source = _iter_block_gain_histograms if stream else _block_gain_histograms
        for gate_j, gate_state, target_k, target_card, gain in source(
            C4_sub, C16_sub, y_sub, p_sub,
            feature_limit=self.pair_feature_limit,
            min_support=min_support,
            gain_l2=gain_l2,
            min_hessian=min_hessian,
            sample_weight=w_sub,
        ):
            yield (gate_j, gate_state, target_k, target_card), gain

    def _gain_dictionary(
        self,
        C4: np.ndarray,
        C16: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        rows: np.ndarray,
        gain_l2: float = 5.0,
        min_hessian: float = 1.0,
        sample_weight: np.ndarray | None = None,
    ):
        return dict(self._gain_items(
            C4, C16, y, p, rows,
            gain_l2=gain_l2,
            min_hessian=min_hessian,
            sample_weight=sample_weight,
        ))

    def _stable_rank_blocks(
        self,
        C4: np.ndarray,
        C16: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        max_results: int | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        # Complementary stratified halves. A candidate must have positive
        # second-order gain in both halves.
        rng = np.random.default_rng(self.random_state)
        idx0 = np.flatnonzero(y == 0)
        idx1 = np.flatnonzero(y == 1)
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        fold_a = np.concatenate([idx0[::2], idx1[::2]])
        fold_b = np.concatenate([idx0[1::2], idx1[1::2]])
        if len(fold_b) == 0:
            fold_b = fold_a

        use_bounded_heap = (
            max_results is not None
            and min(C4.shape[1], int(self.pair_feature_limit)) > 128
        )
        ranked = []
        paired_items = _iter_paired_fold_block_gains(
            C4, C16, y, p, fold_a, fold_b,
            feature_limit=self.pair_feature_limit,
            gain_l2=5.0,
            min_hessian=1.0,
            sample_weight=sample_weight,
        )
        for key, a_value, b_value in paired_items:
            a = float(a_value)
            b = float(b_value)
            if a <= 0.0 or b <= 0.0:
                continue
            gate_j, gate_state, target_k, target_card = key
            harmonic = 2.0 * a * b / (a + b + 1e-12)
            agreement = min(a, b) / (max(a, b) + 1e-12)
            bytes_cost = 8.0 * target_card
            score = harmonic * math.sqrt(max(agreement, 0.0)) / math.sqrt(bytes_cost)
            item = (
                score,
                harmonic,
                agreement,
                gate_j,
                gate_state,
                target_k,
                target_card,
                a,
                b,
            )
            if not use_bounded_heap:
                ranked.append(item)
            elif int(max_results) > 0:
                if len(ranked) < int(max_results):
                    heapq.heappush(ranked, item)
                elif item > ranked[0]:
                    heapq.heapreplace(ranked, item)
        ranked.sort(reverse=True)
        if max_results is not None and not use_bounded_heap:
            ranked = ranked[: int(max_results)]
        self.stability_diagnostics_ = ranked
        return [
            (gate_j, gate_state, target_k, target_card, harmonic)
            for (
                _,
                harmonic,
                _,
                gate_j,
                gate_state,
                target_k,
                target_card,
                _,
                _,
            ) in ranked
        ]

    def _selection_cache(self, Xtr, ytr, Xva):
        base = self._make_backbone()
        base._retain_fit_training_graph = True
        base.fit(Xtr, ytr)
        cached = base._consume_fit_training_graph()
        Sv = base.encoder_.transform_columns(Xva, base.feature_idx_)
        codes_v = base._build_codes_from_states(Sv)
        Zv0 = base.oh_.transform(codes_v)
        if cached is None:
            St = base.encoder_.transform_columns(Xtr, base.feature_idx_)
            codes_t = base._build_codes_from_states(St)
            Zt0 = base.oh_.transform(codes_t)
        else:
            St, codes_t, Zt0 = cached
        pbase = base._probability_from_codes(codes_t)
        ranked = self._stable_rank_blocks(St[4], St[self.block_level], ytr, pbase)
        return base, St, Sv, Zt0, Zv0, pbase, ranked

    def _evaluate_stable_candidate(self, cache, ytr, yva, cfg):
        base, St, Sv, Zt0, Zv0, pbase, ranked = cache
        terms = ranked[: cfg.n_blocks]
        self.block_specs_ = []
        It = self._block_matrix_fit(
            St[4], St[self.block_level], pbase, terms, cfg.cell_shrinkage
        )
        specs = [dict(x) for x in self.block_specs_]
        Iv = self._block_matrix_transform(Sv[4], Sv[self.block_level])
        Zt = sparse.hstack([Zt0, It], format="csr")
        Zv = sparse.hstack([Zv0, Iv], format="csr")
        clf = fit_binary_logistic_exact(
            Zt, ytr, C=cfg.C, random_state=self.random_state, max_iter=2500
        )
        pv = clf.predict_proba(Zv)[:, 1]
        ll = binary_log_loss(yva, pv)
        bytes_est = int(
            base.model_bytes_estimate_ + It.shape[1] * 12 + len(specs) * 16
        )
        return ll, bytes_est, specs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        Xa, Xv, ya, yv = train_test_split(
            X,
            y,
            test_size=0.22,
            stratify=y,
            random_state=self.random_state,
        )
        cache = self._selection_cache(Xa, ya, Xv)
        max_blocks = max(8, min(40, len(ya) // 12))
        configs = list(
            dict.fromkeys(
                [
                    StableBlockConfig(0, 0.0, 0.2),
                    StableBlockConfig(min(4, max_blocks), 0.0, 0.2),
                    StableBlockConfig(min(8, max_blocks), 0.0, 0.2),
                    StableBlockConfig(min(16, max_blocks), 0.0, 0.2),
                    StableBlockConfig(min(24, max_blocks), 10.0, 0.2),
                    StableBlockConfig(min(32, max_blocks), 20.0, 0.2),
                    StableBlockConfig(min(16, max_blocks), 0.0, 1.0),
                ]
            )
        )
        scored = []
        for cfg in configs:
            result = self._evaluate_stable_candidate(cache, ya, yv, cfg)
            scored.append((result[0], result[1], cfg))
        scored.sort(key=lambda z: (z[0], z[1]))
        self.validation_scores_ = scored
        selected_cfg = scored[0][2]
        self.selected_stable_config_ = selected_cfg

        # Final backbone and stable ranking on all rows.
        self.base_ = self._make_backbone()
        self.base_._retain_fit_training_graph = True
        self.base_.fit(X, y)
        cached = self.base_._consume_fit_training_graph()
        if cached is None:
            S = self.base_.encoder_.transform_columns(
                X, self.base_.feature_idx_
            )
            codes0 = self.base_._build_codes_from_states(S)
            Z0 = self.base_.oh_.transform(codes0)
        else:
            S, codes0, Z0 = cached
        pbase = self.base_._probability_from_codes(codes0)
        ranked = self._stable_rank_blocks(S[4], S[self.block_level], y, pbase)
        terms = ranked[: selected_cfg.n_blocks]
        self.block_specs_ = []
        I = self._block_matrix_fit(
            S[4], S[self.block_level], pbase, terms, selected_cfg.cell_shrinkage
        )
        Z = sparse.hstack([Z0, I], format="csr")
        self.clf_ = fit_binary_logistic_exact(
            Z, y, C=selected_cfg.C, random_state=self.random_state, max_iter=2500
        )
        self.classes_ = self.clf_.classes_
        self.design_dim_ = int(Z.shape[1])
        base_dim = Z0.shape[1]
        self.base_coef_ = self.clf_.coef_.ravel()[:base_dim]
        self.block_coef_ = self.clf_.coef_.ravel()[base_dim:]
        self.intercept_ = float(self.clf_.intercept_[0])
        self._compile_block_tables()
        self._build_replacement_plan(S)
        self.model_bytes_estimate_ = int(
            self.base_.model_bytes_estimate_ + self.replacement_bytes_
        )
        return self

    def _build_replacement_plan(self, S):
        groups = {}
        for idx, (spec, table) in enumerate(zip(self.block_specs_, self.block_tables_)):
            key = (spec["gate_j"], spec["target_k"])
            groups.setdefault(key, []).append((idx, spec, table))

        self.execution_groups_ = []
        total_bytes = 0
        original_ops = 0
        replacement_ops = 0
        fused_count = 0

        for (gate_j, target_k), items in groups.items():
            gate_card = int(S[4][:, gate_j].max()) + 1
            target_card = max(len(item[2]) for item in items)
            block_bytes = sum(item[2].nbytes + 16 for item in items)
            fused_bytes = gate_card * target_card * 8 + 16
            m = len(items)
            original_ops += 2 * m

            if self.replacement_objective == "latency":
                fuse = m >= 2
            elif self.replacement_objective == "memory":
                fuse = fused_bytes <= block_bytes
            elif self.replacement_objective == "balanced":
                # One mixed-radix lookup replaces m gate tests and up to m LUTs.
                fuse = m >= 2 and fused_bytes <= 1.75 * block_bytes
            else:
                fuse = False

            if fuse:
                table2d = np.zeros((gate_card, target_card), dtype=float)
                for _, spec, table in items:
                    table2d[spec["gate_state"], : len(table)] += table
                self.execution_groups_.append(
                    {
                        "kind": "fused_pair",
                        "gate_j": gate_j,
                        "target_k": target_k,
                        "table": table2d,
                        "members": [item[0] for item in items],
                    }
                )
                total_bytes += table2d.nbytes + 16
                replacement_ops += 1
                fused_count += 1
            else:
                for idx, spec, table in items:
                    self.execution_groups_.append(
                        {
                            "kind": "block",
                            "gate_j": spec["gate_j"],
                            "gate_state": spec["gate_state"],
                            "target_k": spec["target_k"],
                            "table": table,
                            "members": [idx],
                        }
                    )
                    total_bytes += table.nbytes + 16
                    replacement_ops += 2

        self.replacement_bytes_ = int(total_bytes)
        self.replacement_fused_groups_ = int(fused_count)
        self.replacement_original_ops_ = int(original_ops)
        self.replacement_ops_ = int(replacement_ops)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        S = self._states(X)
        Z0 = self.base_.oh_.transform(self.base_._build_codes_from_states(S))
        score = np.asarray(Z0 @ self.base_coef_).ravel() + self.intercept_
        for group in self.execution_groups_:
            if group["kind"] == "block":
                gate = S[4][:, group["gate_j"]] == group["gate_state"]
                score += gate * group["table"][S[self.block_level][:, group["target_k"]]]
            else:
                score += group["table"][
                    S[4][:, group["gate_j"]],
                    S[self.block_level][:, group["target_k"]],
                ]
        return score

    def decision_function_unreplaced(self, X):
        return QuotientBlockCERM.decision_function(self, X)

    def export_ir(self, prefix):
        prefix = Path(prefix)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")
        arrays = {}
        manifest_groups = []
        for i, group in enumerate(self.execution_groups_):
            name = f"group_{i}"
            arrays[name] = group["table"]
            manifest_groups.append(
                {
                    "kind": group["kind"],
                    "gate_j": int(group["gate_j"]),
                    "gate_state": int(group.get("gate_state", -1)),
                    "target_k": int(group["target_k"]),
                    "array": name,
                }
            )
        np.savez_compressed(
            npz_path,
            feature_idx=self.base_.feature_idx_.astype(np.int16),
            thresholds=np.asarray(self.base_.encoder_.thresholds_, dtype=object),
            base_lookup=np.asarray(self.base_.lookup_, dtype=object),
            intercept=np.asarray([self.intercept_]),
            **arrays,
        )
        manifest = {
            "format": "cerm-stable-quotient-block-ir-v1",
            "tree_or_split_sequence": False,
            "selection": "complementary-fold stability",
            "replacement_objective": self.replacement_objective,
            "groups": manifest_groups,
            "fused_groups": self.replacement_fused_groups_,
            "ops_before": self.replacement_original_ops_,
            "ops_after": self.replacement_ops_,
            "model_bytes_estimate": self.model_bytes_estimate_,
            "config": self.selected_stable_config_.__dict__,
        }
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return npz_path, json_path
