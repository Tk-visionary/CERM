
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

from ..training_graph import binary_log_loss, fit_binary_logistic_exact
from .cerm_hierarchical_residual import (
    HierarchicalResidualCERM,
    NestedQuantileEncoder,
    _rank_pairs,
    _select_features,
)
from .cerm_nested_representation import NestedRepresentationMixin


def _iter_block_gain_histograms(
    C4: np.ndarray,
    C16: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    feature_limit: int,
    min_support: int,
    gain_l2: float,
    min_hessian: float,
    sample_weight: np.ndarray | None = None,
):
    """Return exact block gains from fused joint histograms.

    Joint G/H tables are built once per feature pair.  All gate-state support,
    parent gain, and child gain calculations are then evaluated row-wise in one
    vectorized operation.
    """
    p = np.asarray(p, dtype=float)
    g = np.asarray(y, dtype=float) - p
    h = np.maximum(p * (1.0 - p), 1e-8)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights is not None:
        g = g * weights
        h = h * weights
    d = min(C4.shape[1], int(feature_limit))
    # Histogram keys require int64 arithmetic.  Hoist the conversion out of the
    # O(d^2) pair loop so each state matrix is widened only once per ranking.
    C4_keys = np.asarray(C4[:, :d], dtype=np.int64, order="F")
    C16_keys = np.asarray(C16[:, :d], dtype=np.int64, order="F")
    gate_cards = C4_keys.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    target_cards = C16_keys.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    # ``gate * target_card + target`` is only a temporary histogram key.
    # Reuse one int64 row buffer for every directed feature pair instead of
    # allocating an n-row array O(d^2) times.  The final integer codes and the
    # subsequent G/H bincount reductions are identical to the historical path.
    joint = np.empty(len(C4_keys), dtype=np.int64)
    for gate_j in range(d):
        gate_values = C4_keys[:, gate_j]
        gate_card = int(gate_cards[gate_j])
        support = np.bincount(gate_values, weights=weights, minlength=gate_card) if weights is not None else np.bincount(gate_values, minlength=gate_card)
        for target_k in range(d):
            if target_k == gate_j:
                continue
            target_card = int(target_cards[target_k])
            np.multiply(gate_values, target_card, out=joint)
            np.add(joint, C16_keys[:, target_k], out=joint)
            G = np.bincount(joint, weights=g, minlength=gate_card * target_card).reshape(gate_card, target_card)
            H = np.bincount(joint, weights=h, minlength=gate_card * target_card).reshape(gate_card, target_card)
            total_G = G.sum(axis=1)
            total_H = H.sum(axis=1)
            active = H > 1e-12
            valid = (
                (support >= int(min_support))
                & (total_H >= float(min_hessian))
                & (active.sum(axis=1) > 1)
            )
            if not np.any(valid):
                continue
            parent_gain = total_G * total_G / (total_H + gain_l2)
            child_gain = np.sum(np.where(active, (G * G) / (H + gain_l2), 0.0), axis=1)
            gains = 0.5 * np.maximum(0.0, child_gain - parent_gain)
            for gate_state in np.flatnonzero(valid):
                yield (
                    gate_j, int(gate_state), target_k, target_card,
                    float(gains[gate_state]),
                )


def _block_gain_histograms(*args, **kwargs):
    """Materialize block gains for ordinary feature budgets.

    The list form is measurably faster for the common <=128-feature regime.
    High-dimensional ranking paths call the streaming iterator directly.
    """
    return list(_iter_block_gain_histograms(*args, **kwargs))



class _NestedEngineHierarchicalResidualCERM(
    NestedRepresentationMixin, HierarchicalResidualCERM
):
    """Hierarchical binary backbone using the task-neutral code engine."""



@dataclass(frozen=True)
class BlockConfig:
    n_blocks: int
    gain_l2: float
    cell_shrinkage: float
    min_hessian: float
    C: float


class QuotientBlockCERM(HierarchicalResidualCERM):
    """Hierarchical quotient backbone plus sparse conditional LUT blocks.

    Block operator:
        1[Q_gate^(4)=a] * u(Q_target^(16))

    where u is a jointly estimated finite-state vector. Candidate blocks are
    ranked using a second-order logistic-loss gain, not a tree split.
    """

    def __init__(
        self,
        max_features: int = 64,
        pair_feature_limit: int = 24,
        search_profile: str = "full_exact",
        max_interactions: int = 32,
        fixed_C: float | None = None,
        max_bins: int = 16,
        selection_subsample: float = 1.0,
        n_jobs: int | None = 1,
        random_state: int = 20260803,
        *,
        feature_kinds=None,
        feature_cardinalities=None,
        encoder_kind: str = "quantile",
        newton_prebins: int = 64,
        newton_gain_l2: float = 5.0,
        newton_min_hessian: float = 1.0,
        ranking_kind: str = "mi",
        ranking_l2: float = 5.0,
        ranking_prefilter_multiplier: int = 4,
        cost_per_byte: float = 0.0,
        cost_per_operator: float = 0.0,
        block_cost_per_byte: float = 0.0,
        block_cost_per_eval: float = 0.0,
    ):
        super().__init__(
            max_features=max_features,
            pair_feature_limit=pair_feature_limit,
            search_profile=search_profile,
            fixed_C=fixed_C,
            max_bins=max_bins,
            selection_subsample=selection_subsample,
            n_jobs=n_jobs,
            random_state=random_state,
            selection_rule="best",
            feature_kinds=feature_kinds,
            feature_cardinalities=feature_cardinalities,
            encoder_kind=encoder_kind,
            newton_prebins=newton_prebins,
            newton_gain_l2=newton_gain_l2,
            newton_min_hessian=newton_min_hessian,
            ranking_kind=ranking_kind,
            ranking_l2=ranking_l2,
            ranking_prefilter_multiplier=ranking_prefilter_multiplier,
            cost_per_byte=cost_per_byte,
            cost_per_operator=cost_per_operator,
        )
        self.block_cost_per_byte = float(block_cost_per_byte)
        self.block_cost_per_eval = float(block_cost_per_eval)
        self.max_interactions = int(max_interactions)
        self.fixed_C = None if fixed_C is None else float(fixed_C)

    def _make_backbone(self):
        return _NestedEngineHierarchicalResidualCERM(
            max_features=self.max_features,
            pair_feature_limit=self.pair_feature_limit,
            search_profile=self.search_profile,
            fixed_C=self.fixed_C,
            max_bins=self.max_bins,
            selection_subsample=self.selection_subsample,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            selection_rule="best",
            feature_kinds=self.feature_kinds,
            feature_cardinalities=self.feature_cardinalities,
            encoder_kind=self.encoder_kind,
            newton_prebins=self.newton_prebins,
            newton_gain_l2=self.newton_gain_l2,
            newton_min_hessian=self.newton_min_hessian,
            ranking_kind=self.ranking_kind,
            ranking_l2=self.ranking_l2,
            ranking_prefilter_multiplier=self.ranking_prefilter_multiplier,
            cost_per_byte=self.cost_per_byte,
            cost_per_operator=self.cost_per_operator,
        )

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

    def _prepare_backbone_cache(self, Xtr, ytr, Xva):
        # Fit the already validated hierarchical family on the training part.
        base = self._make_backbone()
        base._retain_fit_training_graph = True
        base.fit(Xtr, ytr)
        cached = base._consume_fit_training_graph()
        Sv = base.encoder_.transform_columns(Xva, base.feature_idx_)
        codes_v = base._build_codes_from_states(Sv)
        Zv = base.oh_.transform(codes_v)
        if cached is None:
            St = base.encoder_.transform_columns(Xtr, base.feature_idx_)
            codes_t = base._build_codes_from_states(St)
            Zt = base.oh_.transform(codes_t)
        else:
            St, codes_t, Zt = cached
        return base, St, Sv, Zt, Zv, codes_t

    def _rank_blocks(
        self,
        C4: np.ndarray,
        C16: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        gain_l2: float,
        min_hessian: float,
        max_results: int | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        min_support = max(20, int(math.ceil(0.05 * len(y))))
        candidates = []
        # Python heap maintenance is slower for ordinary CERM budgets.  Switch
        # to bounded streaming only when the directed feature graph itself is
        # large enough for candidate-list memory to become material.
        use_bounded_heap = (
            max_results is not None
            and min(C4.shape[1], int(self.pair_feature_limit)) > 128
        )
        gain_source = (
            _iter_block_gain_histograms if use_bounded_heap
            else _block_gain_histograms
        )
        for gate_j, gate_state, target_k, target_card, gain in gain_source(
            C4, C16, y, p,
            feature_limit=self.pair_feature_limit,
            min_support=min_support,
            gain_l2=gain_l2,
            min_hessian=min_hessian,
            sample_weight=sample_weight,
        ):
            bytes_cost = 8.0 * target_card + 16.0
            normalized_gain = gain / max(len(y), 1)
            penalized_gain = (
                normalized_gain
                - self.block_cost_per_byte * bytes_cost
                - self.block_cost_per_eval * 2.0
            )
            score = penalized_gain / math.sqrt(max(bytes_cost, 1.0))
            if penalized_gain > 0.0 or (
                self.block_cost_per_byte == 0.0
                and self.block_cost_per_eval == 0.0
            ):
                item = (score, gain, gate_j, gate_state, target_k, target_card)
                if not use_bounded_heap:
                    candidates.append(item)
                elif int(max_results) > 0:
                    if len(candidates) < int(max_results):
                        heapq.heappush(candidates, item)
                    elif item > candidates[0]:
                        heapq.heapreplace(candidates, item)

        candidates.sort(reverse=True)
        if max_results is not None and not use_bounded_heap:
            candidates = candidates[: int(max_results)]
        return [
            (gate_j, gate_state, target_k, target_card, gain)
            for _, gain, gate_j, gate_state, target_k, target_card in candidates
        ]

    def _block_matrix_fit(
        self,
        C4: np.ndarray,
        C16: np.ndarray,
        p_base: np.ndarray,
        terms,
        shrinkage: float,
    ):
        if not terms:
            self.block_specs_ = []
            return sparse.csr_matrix((len(C4), 0), dtype=float)

        h = np.maximum(p_base * (1.0 - p_base), 1e-8)
        cols = []
        specs = []
        for gate_j, gate_state, target_k, target_card, gain in terms:
            gate = C4[:, gate_j] == gate_state
            H = np.bincount(
                C16[gate, target_k], weights=h[gate], minlength=target_card
            )
            total_H = float(H.sum())
            if total_H <= 1e-12 or target_card <= 1:
                continue

            # Conditional effect coding. Each column has weighted mean zero
            # inside the gate, so the block is purified against the gate main
            # effect. The final state is omitted to obtain a full-rank basis.
            states = []
            centers = []
            scales = []
            block_cols = []
            for state in range(target_card - 1):
                pi = float(H[state] / total_H)
                variance_mass = total_H * pi * (1.0 - pi)
                if variance_mass <= 1e-12:
                    continue
                scale = math.sqrt(variance_mass / (variance_mass + shrinkage))
                col = gate.astype(float) * (
                    (C16[:, target_k] == state).astype(float) - pi
                ) * scale
                block_cols.append(col)
                states.append(int(state))
                centers.append(pi)
                scales.append(scale)
            if block_cols:
                cols.extend(block_cols)
                specs.append({
                    "gate_j": int(gate_j),
                    "gate_state": int(gate_state),
                    "target_k": int(target_k),
                    "target_card": int(target_card),
                    "states": states,
                    "centers": centers,
                    "scales": scales,
                    "gain": float(gain),
                })
        self.block_specs_ = specs
        if not cols:
            return sparse.csr_matrix((len(C4), 0), dtype=float)
        return sparse.csr_matrix(np.column_stack(cols))

    def _block_matrix_transform(self, C4: np.ndarray, C16: np.ndarray):
        if not getattr(self, "block_specs_", None):
            return sparse.csr_matrix((len(C4), 0), dtype=float)
        cols = []
        for spec in self.block_specs_:
            gate = C4[:, spec["gate_j"]] == spec["gate_state"]
            for state, center, scale in zip(
                spec["states"], spec["centers"], spec["scales"]
            ):
                cols.append(
                    gate.astype(float)
                    * ((C16[:, spec["target_k"]] == state).astype(float) - center)
                    * scale
                )
        return sparse.csr_matrix(np.column_stack(cols))

    def _candidate_cache(self, Xtr, ytr, Xva):
        base, St, Sv, Zt0, Zv0, codes_t = self._prepare_backbone_cache(Xtr, ytr, Xva)
        p_base = base._probability_from_codes(codes_t)
        return {
            "base": base,
            "St": St,
            "Sv": Sv,
            "Zt0": Zt0,
            "Zv0": Zv0,
            "p_base": p_base,
            "ranked": {},
        }

    def _evaluate_candidate_cache(self, cache, ytr, yva, cfg: BlockConfig):
        rank_key = (cfg.gain_l2, cfg.min_hessian)
        if rank_key not in cache["ranked"]:
            cache["ranked"][rank_key] = self._rank_blocks(
                cache["St"][4], cache["St"][self.block_level], ytr, cache["p_base"],
                cfg.gain_l2, cfg.min_hessian,
            )
        terms = cache["ranked"][rank_key][: cfg.n_blocks]
        self.block_specs_ = []
        It = self._block_matrix_fit(
            cache["St"][4], cache["St"][self.block_level], cache["p_base"], terms,
            cfg.cell_shrinkage,
        )
        specs = [dict(x) for x in self.block_specs_]
        Iv = self._block_matrix_transform(cache["Sv"][4], cache["Sv"][self.block_level])
        Zt = sparse.hstack([cache["Zt0"], It], format="csr")
        Zv = sparse.hstack([cache["Zv0"], Iv], format="csr")
        clf = fit_binary_logistic_exact(
            Zt, ytr, C=cfg.C, random_state=self.random_state, max_iter=2500
        )
        pv = clf.predict_proba(Zv)[:, 1]
        ll = binary_log_loss(yva, pv)
        block_cols = int(It.shape[1])
        bytes_est = int(
            cache["base"].model_bytes_estimate_ + block_cols * 12 + len(specs) * 16
        )
        return ll, bytes_est, clf, specs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        Xa, Xv, ya, yv = train_test_split(
            X, y, test_size=.22, stratify=y, random_state=self.random_state
        )
        cache = self._candidate_cache(Xa, ya, Xv)

        max_blocks = max(8, min(48, len(ya) // 12))
        configs = list(dict.fromkeys([
            BlockConfig(0, 5.0, 10.0, 2.0, .2),
            BlockConfig(min(4,max_blocks), 5.0, 10.0, 2.0, .2),
            BlockConfig(min(8,max_blocks), 5.0, 10.0, 2.0, .2),
            BlockConfig(min(16,max_blocks), 5.0, 10.0, 2.0, .2),
            BlockConfig(min(24,max_blocks), 10.0, 20.0, 5.0, .2),
            BlockConfig(min(32,max_blocks), 20.0, 40.0, 10.0, .2),
            BlockConfig(min(16,max_blocks), 5.0, 10.0, 2.0, 1.0),
        ]))
        scored=[]
        for cfg in configs:
            result=self._evaluate_candidate_cache(cache,ya,yv,cfg)
            scored.append((result[0],result[1],cfg,result))
        scored.sort(key=lambda z:(z[0],z[1]))
        self.validation_scores_=[(float(ll),int(b),cfg) for ll,b,cfg,_ in scored]
        selected_cfg=scored[0][2]
        self.best_block_config_=selected_cfg

        self.block_gate_checked_=False
        self.block_gate_accepted_=selected_cfg.n_blocks==0
        if selected_cfg.n_blocks>0:
            Xb,Xw,yb,yw=train_test_split(
                X,y,test_size=.22,stratify=y,random_state=self.random_state+7919
            )
            cache2=self._candidate_cache(Xb,yb,Xw)
            selected_loss=self._evaluate_candidate_cache(cache2,yb,yw,selected_cfg)[0]
            zero_cfg=BlockConfig(
                0,selected_cfg.gain_l2,selected_cfg.cell_shrinkage,
                selected_cfg.min_hessian,selected_cfg.C
            )
            zero_loss=self._evaluate_candidate_cache(cache2,yb,yw,zero_cfg)[0]
            self.block_gate_checked_=True
            self.secondary_block_loss_=selected_loss
            self.secondary_zero_loss_=zero_loss
            if selected_loss <= zero_loss - .0025:
                self.block_gate_accepted_=True
            else:
                selected_cfg=zero_cfg
                self.block_gate_accepted_=False

        self.base_=self._make_backbone()
        self.base_._retain_fit_training_graph = True
        self.base_.fit(X, y)
        cached = self.base_._consume_fit_training_graph()
        if cached is None:
            S=self.base_.encoder_.transform_columns(X, self.base_.feature_idx_)
            codes0=self.base_._build_codes_from_states(S)
            Z0=self.base_.oh_.transform(codes0)
        else:
            S, codes0, Z0 = cached
        p_base=self.base_._probability_from_codes(codes0)
        ranked=self._rank_blocks(
            S[4],S[self.block_level],y,p_base,selected_cfg.gain_l2,selected_cfg.min_hessian
        )
        terms=ranked[:selected_cfg.n_blocks]
        self.block_specs_=[]
        I=self._block_matrix_fit(
            S[4],S[self.block_level],p_base,terms,selected_cfg.cell_shrinkage
        )
        Z=sparse.hstack([Z0,I],format="csr")
        self.clf_=fit_binary_logistic_exact(
            Z, y, C=selected_cfg.C, random_state=self.random_state, max_iter=2500
        )
        self.classes_=self.clf_.classes_
        self.selected_block_config_=selected_cfg
        self.design_dim_=int(Z.shape[1])
        base_dim=Z0.shape[1]
        self.base_coef_=self.clf_.coef_.ravel()[:base_dim]
        self.block_coef_=self.clf_.coef_.ravel()[base_dim:]
        self.intercept_=float(self.clf_.intercept_[0])
        self._compile_block_tables()
        self.model_bytes_estimate_=int(
            self.base_.model_bytes_estimate_+sum(t.nbytes for t in self.block_tables_)
            +len(self.block_tables_)*12
        )
        return self

    def _compile_block_tables(self):
        self.block_tables_ = []
        pos = 0
        for spec in self.block_specs_:
            table = np.zeros(spec["target_card"], dtype=float)
            for state, center, scale in zip(
                spec["states"], spec["centers"], spec["scales"]
            ):
                coef = self.block_coef_[pos] * scale
                table -= coef * center
                table[state] += coef
                pos += 1
            self.block_tables_.append(table)
        if pos != len(self.block_coef_):
            raise RuntimeError("block coefficient accounting mismatch")

    def _states(self, X):
        matrix = np.asarray(X, dtype=float)
        # This is exactly ``encoder.transform(matrix)[level][:, feature_idx]``
        # but does not allocate state columns for unused adapted features.
        return self.base_.encoder_.transform_columns(matrix, self.base_.feature_idx_)

    def decision_function_projected(self, X_selected):
        values = np.asarray(X_selected, dtype=float)
        S = self.base_.encoder_.transform_projected(values, self.base_.feature_idx_)
        Z0 = self.base_.oh_.transform(self.base_._build_codes_from_states(S))
        score = np.asarray(Z0 @ self.base_coef_).ravel() + self.intercept_
        for spec, table in zip(self.block_specs_, self.block_tables_):
            gate = S[4][:, spec["gate_j"]] == spec["gate_state"]
            score += gate * table[S[self.block_level][:, spec["target_k"]]]
        return score

    def predict_proba_projected(self, X_selected):
        p = self._sigmoid(self.decision_function_projected(X_selected))
        return np.column_stack([1.0 - p, p])

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        S = self._states(X)
        Z0 = self.base_.oh_.transform(self.base_._build_codes_from_states(S))
        score = np.asarray(Z0 @ self.base_coef_).ravel() + self.intercept_
        for spec, table in zip(self.block_specs_, self.block_tables_):
            gate = S[4][:, spec["gate_j"]] == spec["gate_state"]
            score += gate * table[S[self.block_level][:, spec["target_k"]]]
        return score

    def predict_proba(self, X):
        p = self._sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict_proba_reference(self, X):
        S = self._states(X)
        Z0 = self.base_.oh_.transform(self.base_._build_codes_from_states(S))
        I = self._block_matrix_transform(S[4], S[self.block_level])
        Z = sparse.hstack([Z0, I], format="csr")
        return self.clf_.predict_proba(Z)

    def export_ir(self, prefix):
        prefix = Path(prefix)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")
        np.savez_compressed(
            npz_path,
            feature_idx=self.base_.feature_idx_.astype(np.int16),
            thresholds=np.asarray(self.base_.encoder_.thresholds_, dtype=object),
            base_lookup=np.asarray(self.base_.lookup_, dtype=object),
            block_tables=np.asarray(self.block_tables_, dtype=object),
            block_specs=np.asarray(
                [
                    (
                        s["gate_j"],
                        s["gate_state"],
                        s["target_k"],
                        s["target_card"],
                    )
                    for s in self.block_specs_
                ],
                dtype=np.int16,
            ).reshape(-1, 4),
            intercept=np.asarray([self.intercept_]),
        )
        manifest = {
            "format": "cerm-quotient-block-ir-v1",
            "tree_or_split_sequence": False,
            "head_count": 1,
            "operator": "coarse-gated-fine-state-lut-block",
            "n_blocks": len(self.block_tables_),
            "model_bytes_estimate": self.model_bytes_estimate_,
            "config": self.selected_block_config_.__dict__,
        }
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return npz_path, json_path
