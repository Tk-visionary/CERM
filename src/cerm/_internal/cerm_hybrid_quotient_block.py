
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import heapq
import math

import numpy as np
from joblib import effective_n_jobs
from scipy import sparse
from sklearn.model_selection import train_test_split

from .cerm_hierarchical_residual import HierarchicalResidualCERM
from .cerm_quotient_block import QuotientBlockCERM
from .cerm_stable_quotient_block import StableQuotientBlockCERM
from .cerm_shared_statistics import (
    iter_paired_fold_block_gains_shared,
    native_paired_fold_block_gain_bank,
    nested_gate_maps,
)
from ..training_graph import (
    BlockColumnBank,
    SemanticBlockDictionary,
    SemanticBlockBank,
    solve_binary_logistic_path,
    solve_binary_logistic_semantic_blocks,
    solve_binary_logistic_semantic_path,
    fit_binary_logistic_semantic_exact,
    semantic_block_solver_available,
    fit_binary_logistic_exact,
    binary_log_loss,
)


@dataclass(frozen=True)
class HybridConfig:
    rank_mode: str
    n_blocks: int
    cell_shrinkage: float
    C: float


_SHARED_RANK_GUARD_REL = 1e-6


def _ranking_boundary_ambiguous(items, max_results, rel_tol=_SHARED_RANK_GUARD_REL):
    """Return whether the retained-prefix boundary is numerically ambiguous."""
    if max_results is None:
        return False
    limit = int(max_results)
    if limit <= 0 or len(items) <= limit:
        return False
    scale = max(abs(float(items[0][0])), 1e-30)
    gap = float(items[limit - 1][0]) - float(items[limit][0])
    return gap <= float(rel_tol) * scale



def _tie_safe_top_indices(score, valid, limit):
    """Return a score-boundary superset before exact lexicographic ordering."""
    ids = np.flatnonzero(valid)
    if limit is None or len(ids) <= int(limit):
        return ids
    limit = int(limit)
    if limit <= 0:
        return np.empty(0, dtype=np.int64)
    local_score = score[ids]
    cut_position = len(ids) - limit
    partition = np.argpartition(local_score, cut_position)
    threshold = local_score[partition[cut_position]]
    return ids[local_score >= threshold]


def _rank_native_candidate_bank(
    bank,
    *,
    n_rows,
    max_results,
    block_cost_per_byte,
    block_cost_per_eval,
):
    """Reproduce Python tuple ranking without materializing all candidates."""
    if max_results is not None and int(max_results) <= 0:
        return [], []

    gate_j = np.asarray(bank["gate_j"])
    gate_state = np.asarray(bank["gate_state"])
    target_k = np.asarray(bank["target_k"])
    target_card = np.asarray(bank["target_card"])
    full_gain = np.asarray(bank["full"], dtype=np.float64)
    gain_a = np.asarray(bank["gain_a"], dtype=np.float64)
    gain_b = np.asarray(bank["gain_b"], dtype=np.float64)
    flags = np.asarray(bank["flags"], dtype=np.uint8)

    retain_limit = (
        None if max_results is None else int(max_results) + 1
    )

    full_bytes = 8.0 * target_card.astype(np.float64) + 16.0
    normalized_gain = full_gain / max(int(n_rows), 1)
    penalized_gain = (
        normalized_gain
        - float(block_cost_per_byte) * full_bytes
        - float(block_cost_per_eval) * 2.0
    )
    full_score = penalized_gain / np.sqrt(np.maximum(full_bytes, 1.0))
    full_valid = (flags & 4) != 0
    if float(block_cost_per_byte) != 0.0 or float(block_cost_per_eval) != 0.0:
        full_valid &= penalized_gain > 0.0

    full_ids = _tie_safe_top_indices(
        full_score,
        full_valid,
        retain_limit,
    )
    if len(full_ids):
        order = np.lexsort(
            (
                target_card[full_ids],
                target_k[full_ids],
                gate_state[full_ids],
                gate_j[full_ids],
                full_gain[full_ids],
                full_score[full_ids],
            )
        )[::-1]
        full_ids = full_ids[order]
        if retain_limit is not None:
            full_ids = full_ids[:retain_limit]
    full_candidates = [
        (
            float(full_score[i]),
            float(full_gain[i]),
            int(gate_j[i]),
            int(gate_state[i]),
            int(target_k[i]),
            int(target_card[i]),
        )
        for i in full_ids
    ]

    stable_valid = ((flags & 3) == 3) & (gain_a > 0.0) & (gain_b > 0.0)
    harmonic = np.zeros_like(gain_a)
    agreement = np.zeros_like(gain_a)
    np.divide(
        2.0 * gain_a * gain_b,
        gain_a + gain_b + 1e-12,
        out=harmonic,
        where=stable_valid,
    )
    np.divide(
        np.minimum(gain_a, gain_b),
        np.maximum(gain_a, gain_b) + 1e-12,
        out=agreement,
        where=stable_valid,
    )
    stable_score = (
        harmonic
        * np.sqrt(np.maximum(agreement, 0.0))
        / np.sqrt(8.0 * target_card.astype(np.float64))
    )
    stable_ids = _tie_safe_top_indices(
        stable_score,
        stable_valid,
        retain_limit,
    )
    if len(stable_ids):
        order = np.lexsort(
            (
                gain_b[stable_ids],
                gain_a[stable_ids],
                target_card[stable_ids],
                target_k[stable_ids],
                gate_state[stable_ids],
                gate_j[stable_ids],
                agreement[stable_ids],
                harmonic[stable_ids],
                stable_score[stable_ids],
            )
        )[::-1]
        stable_ids = stable_ids[order]
        if retain_limit is not None:
            stable_ids = stable_ids[:retain_limit]
    stable_candidates = [
        (
            float(stable_score[i]),
            float(harmonic[i]),
            float(agreement[i]),
            int(gate_j[i]),
            int(gate_state[i]),
            int(target_k[i]),
            int(target_card[i]),
            float(gain_a[i]),
            float(gain_b[i]),
        )
        for i in stable_ids
    ]
    return full_candidates, stable_candidates


class HybridQuotientBlockCERM(StableQuotientBlockCERM):
    """Select between full-data and complementary-fold block dictionaries."""

    def _shared_rank_blocks(
        self,
        C4,
        C16,
        y,
        p,
        *,
        max_results=None,
        sample_weight=None,
    ):
        """Build full/stable rankings from one paired-fold fine histogram scan.

        Return None when the supplied state matrices are not a deterministic
        fine-to-coarse quotient, allowing callers to retain the historical
        independent ranking paths.
        """
        rng = np.random.default_rng(self.random_state)
        idx0 = np.flatnonzero(y == 0)
        idx1 = np.flatnonzero(y == 1)
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        fold_a = np.concatenate([idx0[::2], idx1[::2]])
        fold_b = np.concatenate([idx0[1::2], idx1[1::2]])
        if len(fold_b) == 0:
            # The historical stable path duplicates fold A in this degenerate
            # tiny-sample case. Reusing that duplicated table for the full-data
            # rank would double G/H, so keep the historical separate paths.
            return None

        rows = np.concatenate([fold_a, fold_b])
        maps = nested_gate_maps(
            C4, C16, rows, self.pair_feature_limit
        )
        if maps is None:
            return None

        full_min_support = max(20, int(math.ceil(0.05 * len(y))))
        histogram_backend = {}
        native_bank = native_paired_fold_block_gain_bank(
            C4,
            C16,
            y,
            p,
            fold_a,
            fold_b,
            feature_limit=self.pair_feature_limit,
            full_min_support=full_min_support,
            gain_l2=5.0,
            min_hessian=1.0,
            sample_weight=sample_weight,
            maps=maps,
            n_jobs=self.n_jobs,
            backend_info=histogram_backend,
        )

        if native_bank is not None:
            full_candidates, stable_candidates = _rank_native_candidate_bank(
                native_bank,
                n_rows=len(y),
                max_results=max_results,
                block_cost_per_byte=self.block_cost_per_byte,
                block_cost_per_eval=self.block_cost_per_eval,
            )
        else:
            use_bounded_heap = (
                max_results is not None
                and min(C4.shape[1], int(self.pair_feature_limit)) > 128
            )
            full_candidates = []
            stable_candidates = []

            def retain(buffer, item):
                if not use_bounded_heap:
                    buffer.append(item)
                    return
                limit = int(max_results) + 1
                if limit <= 1:
                    return
                if len(buffer) < limit:
                    heapq.heappush(buffer, item)
                elif item > buffer[0]:
                    heapq.heapreplace(buffer, item)

            shared = iter_paired_fold_block_gains_shared(
                C4,
                C16,
                y,
                p,
                fold_a,
                fold_b,
                feature_limit=self.pair_feature_limit,
                full_min_support=full_min_support,
                gain_l2=5.0,
                min_hessian=1.0,
                sample_weight=sample_weight,
                maps=maps,
                n_jobs=self.n_jobs,
                backend_info=histogram_backend,
            )
            for key, full_gain, stable_a, stable_b in shared:
                gate_j, gate_state, target_k, target_card = key

                if full_gain is not None:
                    gain = float(full_gain)
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
                        retain(
                            full_candidates,
                            (
                                score,
                                gain,
                                gate_j,
                                gate_state,
                                target_k,
                                target_card,
                            ),
                        )

                if stable_a is not None and stable_b is not None:
                    a = float(stable_a)
                    b = float(stable_b)
                    if a <= 0.0 or b <= 0.0:
                        continue
                    harmonic = 2.0 * a * b / (a + b + 1e-12)
                    agreement = min(a, b) / (max(a, b) + 1e-12)
                    bytes_cost = 8.0 * target_card
                    score = (
                        harmonic
                        * math.sqrt(max(agreement, 0.0))
                        / math.sqrt(bytes_cost)
                    )
                    retain(
                        stable_candidates,
                        (
                            score,
                            harmonic,
                            agreement,
                            gate_j,
                            gate_state,
                            target_k,
                            target_card,
                            a,
                            b,
                        ),
                    )

            full_candidates.sort(reverse=True)
            stable_candidates.sort(reverse=True)

        # Practical exactness guard. Quotient aggregation changes only floating-
        # point summation order. If the retained-prefix boundary is extremely
        # tight, recompute that rank mode with the historical direct row scan.
        guard_fallback = []
        if _ranking_boundary_ambiguous(full_candidates, max_results):
            guard_fallback.append("full")
        if _ranking_boundary_ambiguous(stable_candidates, max_results):
            guard_fallback.append("stable")

        if max_results is not None:
            full_candidates = full_candidates[: int(max_results)]
            stable_candidates = stable_candidates[: int(max_results)]

        self.stability_diagnostics_ = stable_candidates
        return {
            "_guard_fallback": tuple(guard_fallback),
            "_histogram_backend": histogram_backend.get(
                "pair_histogram", "python"
            ),
            "full": [
                (gate_j, gate_state, target_k, target_card, gain)
                for (
                    _,
                    gain,
                    gate_j,
                    gate_state,
                    target_k,
                    target_card,
                ) in full_candidates
            ],
            "stable": [
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
                ) in stable_candidates
            ],
        }

    def _hybrid_cache(self, Xtr, ytr, Xva, max_ranked_blocks=None, sample_weight=None):
        base = self._make_backbone()
        base._retain_fit_training_graph = True
        base.fit(Xtr, ytr, sample_weight=sample_weight)
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
        ranked = self._shared_rank_blocks(
            St[4],
            St[self.block_level],
            ytr,
            pbase,
            max_results=max_ranked_blocks,
            sample_weight=sample_weight,
        )
        if ranked is None:
            full = self._rank_blocks(
                St[4], St[self.block_level], ytr, pbase,
                gain_l2=5.0, min_hessian=1.0,
                max_results=max_ranked_blocks,
                sample_weight=sample_weight,
            )
            stable = self._stable_rank_blocks(
                St[4], St[self.block_level], ytr, pbase,
                max_results=max_ranked_blocks,
                sample_weight=sample_weight,
            )
            ranking_backend = "separate"
            ranking_guard_fallback = ()
        else:
            ranking_guard_fallback = ranked["_guard_fallback"]
            if "full" in ranking_guard_fallback:
                full = self._rank_blocks(
                    St[4], St[self.block_level], ytr, pbase,
                    gain_l2=5.0, min_hessian=1.0,
                    max_results=max_ranked_blocks,
                    sample_weight=sample_weight,
                )
            else:
                full = ranked["full"]
            if "stable" in ranking_guard_fallback:
                stable = self._stable_rank_blocks(
                    St[4], St[self.block_level], ytr, pbase,
                    max_results=max_ranked_blocks,
                    sample_weight=sample_weight,
                )
            else:
                stable = ranked["stable"]
            ranking_backend = (
                "shared_fine_pair_"
                + ranked.get("_histogram_backend", "python")
            )
        return {
            "base": base,
            "St": St,
            "Sv": Sv,
            "Zt0": Zt0,
            "Zv0": Zv0,
            "pbase": pbase,
            "ranked": {"full": full, "stable": stable},
            "ranking_backend": ranking_backend,
            "ranking_guard_fallback": ranking_guard_fallback,
            "sample_weight": sample_weight,
        }

    def _block_bank(self, cache, cfg):
        key = (cfg.rank_mode, float(cfg.cell_shrinkage))
        banks = cache.setdefault("column_banks", {})
        if key not in banks:
            ranked = cache["ranked"][cfg.rank_mode]
            requested = cache.get("max_requested_blocks", {}).get(
                key, cfg.n_blocks
            )
            max_terms = min(int(requested), len(ranked))
            banks[key] = BlockColumnBank.build(
                cache["St"][4],
                cache["St"][self.block_level],
                cache["Sv"][4],
                cache["Sv"][self.block_level],
                cache["pbase"],
                ranked[:max_terms],
                cfg.cell_shrinkage,
                sample_weight=cache.get("sample_weight"),
            )
        return banks[key]

    def _semantic_bank(self, cache, cfg):
        key = (cfg.rank_mode, float(cfg.cell_shrinkage))
        banks = cache.setdefault("semantic_banks", {})
        if key not in banks:
            ranked = cache["ranked"][cfg.rank_mode]
            requested = cache.get("max_requested_blocks", {}).get(
                key, cfg.n_blocks
            )
            max_terms = min(int(requested), len(ranked))
            banks[key] = SemanticBlockBank.build(
                cache["St"][4],
                cache["St"][self.block_level],
                cache["Sv"][4],
                cache["pbase"],
                ranked[:max_terms],
                cfg.cell_shrinkage,
                sample_weight=cache.get("sample_weight"),
            )
        return banks[key]

    @staticmethod
    def _prepare_bank_requests(cache, configs):
        requests = {}
        for cfg in configs:
            key = (cfg.rank_mode, float(cfg.cell_shrinkage))
            requests[key] = max(requests.get(key, 0), int(cfg.n_blocks))
        cache["max_requested_blocks"] = requests
        cache["column_banks"] = {}
        cache["semantic_banks"] = {}

    def _eval_design_groups(
        self,
        cache,
        ytr,
        yva,
        design_groups,
        wtr=None,
        wva=None,
    ):
        """Evaluate independent prefix designs concurrently without nesting solvers.

        Column banks are materialized before worker launch so the shared cache is
        read-only during solves.  When multiple design groups exist, outer
        parallelism consumes the job budget and each group's C-path runs with one
        solver worker, avoiding nested oversubscription.
        """
        groups = list(design_groups.values())
        if not groups:
            return [], 1

        semantic_available = (
            semantic_block_solver_available()
            and {"ranked", "St", "Sv", "pbase"}.issubset(cache)
        )

        # Complete shared preparation before worker launch.  A maximal semantic
        # bank serves every non-zero nested prefix for one
        # (rank_mode, shrinkage) family.  Materialized BlockColumnBank remains
        # only as the fallback when the experimental native operator is absent.
        semantic_representatives = {}
        materialized_representatives = {}
        for group in groups:
            cfg = group[0]
            if int(cfg.n_blocks) <= 0:
                continue
            key = (cfg.rank_mode, float(cfg.cell_shrinkage))
            if semantic_available:
                semantic_representatives.setdefault(key, cfg)
            else:
                materialized_representatives.setdefault(key, cfg)
        for cfg in semantic_representatives.values():
            self._semantic_bank(cache, cfg)
        for cfg in materialized_representatives.values():
            self._block_bank(cache, cfg)

        workers = min(
            max(1, effective_n_jobs(self.n_jobs)),
            len(groups),
        )

        def evaluate(group):
            return self._eval_hybrid_path(
                cache,
                ytr,
                yva,
                group,
                wtr,
                wva,
                solver_n_jobs=1 if workers > 1 else self.n_jobs,
            )

        if workers <= 1:
            nested = [evaluate(group) for group in groups]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                nested = list(pool.map(evaluate, groups))
        return [item for group_result in nested for item in group_result], workers

    def _eval_hybrid(self, cache, ytr, yva, cfg, wtr=None, wva=None):
        if int(cfg.n_blocks) > 0 and semantic_block_solver_available():
            semantic = self._semantic_bank(cache, cfg).view(cfg.n_blocks)
            specs = [dict(x) for x in semantic.specs]
            self.block_specs_ = specs
            solution = solve_binary_logistic_semantic_blocks(
                cache["Zt0"],
                cache["St"][self.block_level],
                semantic,
                ytr,
                cache["Zv0"],
                cache["Sv"][self.block_level],
                C=cfg.C,
                random_state=self.random_state,
                max_iter=2500,
                sample_weight=wtr,
            )
            loss = binary_log_loss(
                yva, solution.valid_probability, sample_weight=wva
            )
            byte_est = int(
                cache["base"].model_bytes_estimate_
                + semantic.n_columns * 12
                + len(specs) * 16
            )
            return loss, byte_est, specs

        if int(cfg.n_blocks) <= 0:
            path = solve_binary_logistic_path(
                cache["Zt0"],
                ytr,
                cache["Zv0"],
                [cfg.C],
                random_state=self.random_state,
                max_iter=2500,
                sample_weight=wtr,
                n_jobs=1,
            )
            solution = path[float(cfg.C)]
            loss = binary_log_loss(
                yva, solution.valid_probability, sample_weight=wva
            )
            return loss, int(cache["base"].model_bytes_estimate_), []

        view = self._block_bank(cache, cfg).view(cfg.n_blocks)
        It = view.train
        Iv = view.valid
        specs = [dict(x) for x in view.specs]
        self.block_specs_ = specs
        byte_est = int(
            cache["base"].model_bytes_estimate_ + It.shape[1] * 12 + len(specs) * 16
        )
        Zt = sparse.hstack([cache["Zt0"], It], format="csr")
        Zv = sparse.hstack([cache["Zv0"], Iv], format="csr")
        clf = fit_binary_logistic_exact(
            Zt, ytr, C=cfg.C, random_state=self.random_state, max_iter=2500,
            sample_weight=wtr,
        )
        p = clf.predict_proba(Zv)[:, 1]
        loss = binary_log_loss(yva, p, sample_weight=wva)
        return loss, byte_est, specs

    def _eval_hybrid_path(
        self,
        cache,
        ytr,
        yva,
        configs,
        wtr=None,
        wva=None,
        *,
        solver_n_jobs=None,
    ):
        representative = configs[0]
        if int(representative.n_blocks) > 0 and semantic_block_solver_available():
            semantic = self._semantic_bank(cache, representative).view(
                representative.n_blocks
            )
            path = solve_binary_logistic_semantic_path(
                cache["Zt0"],
                cache["St"][self.block_level],
                semantic,
                ytr,
                cache["Zv0"],
                cache["Sv"][self.block_level],
                [cfg.C for cfg in configs],
                random_state=self.random_state,
                max_iter=2500,
                sample_weight=wtr,
                n_jobs=self.n_jobs if solver_n_jobs is None else solver_n_jobs,
            )
            byte_est = int(
                cache["base"].model_bytes_estimate_
                + semantic.n_columns * 12
                + len(semantic.specs) * 16
            )
            results = []
            for cfg in configs:
                probability = path[float(cfg.C)].valid_probability
                loss = binary_log_loss(
                    yva, probability, sample_weight=wva
                )
                results.append((loss, byte_est, cfg))
            return results

        if int(representative.n_blocks) <= 0:
            path = solve_binary_logistic_path(
                cache["Zt0"],
                ytr,
                cache["Zv0"],
                [cfg.C for cfg in configs],
                random_state=self.random_state,
                max_iter=2500,
                sample_weight=wtr,
                n_jobs=self.n_jobs if solver_n_jobs is None else solver_n_jobs,
            )
            byte_est = int(cache["base"].model_bytes_estimate_)
            return [
                (
                    binary_log_loss(
                        yva,
                        path[float(cfg.C)].valid_probability,
                        sample_weight=wva,
                    ),
                    byte_est,
                    cfg,
                )
                for cfg in configs
            ]

        view = self._block_bank(cache, representative).view(representative.n_blocks)
        It, Iv = view.train, view.valid
        specs = [dict(x) for x in view.specs]
        byte_est = int(
            cache["base"].model_bytes_estimate_ + It.shape[1] * 12 + len(specs) * 16
        )
        Zt = sparse.hstack([cache["Zt0"], It], format="csr")
        Zv = sparse.hstack([cache["Zv0"], Iv], format="csr")
        path = solve_binary_logistic_path(
            Zt, ytr, Zv, [cfg.C for cfg in configs],
            random_state=self.random_state, max_iter=2500, sample_weight=wtr,
            n_jobs=self.n_jobs if solver_n_jobs is None else solver_n_jobs,
        )
        results = []
        for cfg in configs:
            p = path[float(cfg.C)].valid_probability
            loss = binary_log_loss(yva, p, sample_weight=wva)
            results.append((loss, byte_est, cfg))
        return results

    @staticmethod
    def _prefixes(base_prefixes, max_blocks):
        max_blocks = int(max_blocks)
        if max_blocks <= 0:
            return []
        values = [min(int(value), max_blocks) for value in base_prefixes]
        values.append(max_blocks)
        return list(dict.fromkeys(value for value in values if value > 0))

    def _configs(self, n):
        natural_limit = max(8, min(40, int(n) // 12))
        max_blocks = min(int(self.max_interactions), natural_limit)
        base_C = 0.2 if self.fixed_C is None else float(self.fixed_C)
        configs = [HybridConfig("full", 0, 0.0, base_C)]
        if max_blocks <= 0:
            return configs

        if self.search_profile in {"practical", "aggressive"}:
            prefixes = self._prefixes((8, 16, 32), max_blocks)
            for mode in ("full", "stable"):
                for k in prefixes:
                    configs.append(HybridConfig(mode, k, 0.0, base_C))
                if self.fixed_C is None:
                    configs.append(
                        HybridConfig(mode, min(16, max_blocks), 0.0, 1.0)
                    )
            return list(dict.fromkeys(configs))

        prefixes = self._prefixes((4, 8, 16, 24, 32), max_blocks)
        for mode in ("full", "stable"):
            for k in prefixes:
                configs.append(HybridConfig(mode, k, 0.0, base_C))
            configs.append(
                HybridConfig(mode, min(16, max_blocks), 10.0, base_C)
            )
            if self.fixed_C is None:
                configs.append(
                    HybridConfig(mode, min(16, max_blocks), 0.0, 1.0)
                )
        return list(dict.fromkeys(configs))

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

        X_selection, y_selection, w_selection = self._selection_sample(
            X, y, sample_weight, seed_offset=17
        )
        self.selection_rows_ = int(len(X_selection))
        ia, iv = train_test_split(
            np.arange(len(X_selection)), test_size=0.22,
            stratify=y_selection, random_state=self.random_state
        )
        Xa, Xv = X_selection[ia], X_selection[iv]
        ya, yv = y_selection[ia], y_selection[iv]
        wa = None if w_selection is None else w_selection[ia]
        wv = None if w_selection is None else w_selection[iv]
        configs = self._configs(len(ya))
        max_ranked_blocks = max(config.n_blocks for config in configs)
        cache = self._hybrid_cache(
            Xa, ya, Xv, max_ranked_blocks=max_ranked_blocks, sample_weight=wa
        )
        self.primary_ranking_backend_ = cache["ranking_backend"]
        self.primary_ranking_guard_fallback_ = cache["ranking_guard_fallback"]
        self._prepare_bank_requests(cache, configs)
        design_groups = {}
        for cfg in configs:
            key = (cfg.rank_mode, cfg.n_blocks, float(cfg.cell_shrinkage))
            design_groups.setdefault(key, []).append(cfg)
        scored, solver_group_workers = self._eval_design_groups(
            cache,
            ya,
            yv,
            design_groups,
            wa,
            wv,
        )
        self.primary_solver_group_workers_ = int(solver_group_workers)
        self.primary_candidate_count_ = len(configs)
        self.primary_design_count_ = len(design_groups)
        self.primary_column_bank_count_ = len(cache.get("column_banks", {}))
        self.primary_semantic_bank_count_ = len(cache.get("semantic_banks", {}))
        scored.sort(key=lambda z: (z[0], z[1]))

        # Strict near-tie rule avoids selecting a materially larger dictionary
        # for a negligible validation gain.
        best_loss = scored[0][0]
        eligible = [x for x in scored if x[0] <= best_loss + 0.0015]
        eligible.sort(key=lambda z: (z[1], z[0]))
        selected = eligible[0][2]
        self.primary_scores_ = scored
        self.primary_selected_config_ = selected

        # Independent confirmation against the zero-block backbone.
        self.secondary_checked_ = selected.n_blocks > 0
        self.secondary_accepted_ = selected.n_blocks == 0
        if selected.n_blocks > 0:
            ib, iw = train_test_split(
                np.arange(len(X_selection)), test_size=0.22, stratify=y_selection,
                random_state=self.random_state + 7919,
            )
            Xb, Xw = X_selection[ib], X_selection[iw]
            yb, yw = y_selection[ib], y_selection[iw]
            wb = None if w_selection is None else w_selection[ib]
            ww = None if w_selection is None else w_selection[iw]
            cache2 = self._hybrid_cache(
                Xb, yb, Xw, max_ranked_blocks=selected.n_blocks, sample_weight=wb
            )
            self.secondary_ranking_backend_ = cache2["ranking_backend"]
            self.secondary_ranking_guard_fallback_ = cache2[
                "ranking_guard_fallback"
            ]
            zero = HybridConfig("full", 0, 0.0, selected.C)
            self._prepare_bank_requests(cache2, [selected, zero])
            sel_loss = self._eval_hybrid(cache2, yb, yw, selected, wb, ww)[0]
            zero_loss = self._eval_hybrid(cache2, yb, yw, zero, wb, ww)[0]
            self.secondary_selected_loss_ = sel_loss
            self.secondary_zero_loss_ = zero_loss
            if sel_loss <= zero_loss - 0.0015:
                self.secondary_accepted_ = True
            else:
                selected = zero
                self.secondary_accepted_ = False

        self.selected_hybrid_config_ = selected

        # Final joint fit.
        self.base_ = self._make_backbone()
        self.base_._retain_fit_training_graph = True
        self.base_.fit(X, y, sample_weight=sample_weight)
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
        if selected.rank_mode == "stable":
            ranked = self._stable_rank_blocks(
                S[4], S[self.block_level], y, pbase,
                max_results=selected.n_blocks,
                sample_weight=sample_weight,
            )
        else:
            ranked = self._rank_blocks(
                S[4], S[self.block_level], y, pbase,
                gain_l2=5.0, min_hessian=1.0,
                max_results=selected.n_blocks, sample_weight=sample_weight,
            )
        terms = ranked[: selected.n_blocks]
        self.block_specs_ = []
        base_dim = int(Z0.shape[1])

        if (
            selected.n_blocks > 0
            and semantic_block_solver_available()
        ):
            semantic = SemanticBlockDictionary.build(
                S[4],
                S[self.block_level],
                S[4][:0],
                pbase,
                terms,
                selected.cell_shrinkage,
                sample_weight=sample_weight,
            )
            self.block_specs_ = [dict(spec) for spec in semantic.specs]
            self.clf_ = fit_binary_logistic_semantic_exact(
                Z0,
                S[self.block_level],
                semantic,
                y,
                C=selected.C,
                random_state=self.random_state,
                max_iter=2500,
                sample_weight=sample_weight,
            )
            self.design_dim_ = int(base_dim + semantic.n_columns)
            self.final_solver_backend_ = "semantic_blocks"
        else:
            if sample_weight is None:
                I = self._block_matrix_fit(
                    S[4],
                    S[self.block_level],
                    pbase,
                    terms,
                    selected.cell_shrinkage,
                )
            else:
                final_bank = BlockColumnBank.build(
                    S[4],
                    S[self.block_level],
                    S[4],
                    S[self.block_level],
                    pbase,
                    terms,
                    selected.cell_shrinkage,
                    sample_weight=sample_weight,
                )
                final_view = final_bank.view(selected.n_blocks)
                I = final_view.train
                self.block_specs_ = [dict(spec) for spec in final_view.specs]
            Z = sparse.hstack([Z0, I], format="csr")
            self.clf_ = fit_binary_logistic_exact(
                Z,
                y,
                C=selected.C,
                random_state=self.random_state,
                max_iter=2500,
                sample_weight=sample_weight,
            )
            self.design_dim_ = int(Z.shape[1])
            self.final_solver_backend_ = "materialized_csr"

        self.classes_ = self.clf_.classes_
        self.base_coef_ = self.clf_.coef_.ravel()[:base_dim]
        self.block_coef_ = self.clf_.coef_.ravel()[base_dim:]
        self.intercept_ = float(self.clf_.intercept_[0])
        self._compile_block_tables()
        self._build_replacement_plan_v2(S)
        self.model_bytes_estimate_ = int(
            self.base_.model_bytes_estimate_ + self.replacement_bytes_
        )
        return self

    def _build_replacement_plan_v2(self, S):
        # Stage 1: optional block-to-pair LUT fusion.
        pair_groups = {}
        for idx, (spec, table) in enumerate(zip(self.block_specs_, self.block_tables_)):
            pair_groups.setdefault((spec["gate_j"], spec["target_k"]), []).append(
                (idx, spec, table)
            )

        used = set()
        execution = []
        total_bytes = 0
        ops_before = 2 * len(self.block_tables_)
        ops_after = 0
        pair_fusions = 0

        for (gate_j, target_k), items in pair_groups.items():
            gate_card = int(S[4][:, gate_j].max()) + 1
            target_card = max(len(t) for _, _, t in items)
            block_bytes = sum(t.nbytes + 16 for _, _, t in items)
            fused_bytes = gate_card * target_card * 8 + 16
            m = len(items)
            if self.replacement_objective == "latency":
                fuse = m >= 2
            elif self.replacement_objective == "memory":
                fuse = fused_bytes <= block_bytes
            elif self.replacement_objective == "balanced":
                fuse = m >= 2 and fused_bytes <= 1.75 * block_bytes
            else:
                fuse = False
            if fuse:
                table2d = np.zeros((gate_card, target_card), dtype=float)
                for idx, spec, table in items:
                    table2d[spec["gate_state"], : len(table)] += table
                    used.add(idx)
                execution.append({
                    "kind": "fused_pair",
                    "gate_j": gate_j,
                    "target_k": target_k,
                    "table": table2d,
                })
                total_bytes += table2d.nbytes + 16
                ops_after += 1
                pair_fusions += 1

        # Stage 2: hoist a shared gate comparison for remaining blocks.
        gate_groups = {}
        for idx, (spec, table) in enumerate(zip(self.block_specs_, self.block_tables_)):
            if idx in used:
                continue
            gate_groups.setdefault((spec["gate_j"], spec["gate_state"]), []).append(
                (spec, table)
            )
        gate_hoists = 0
        for (gate_j, gate_state), items in gate_groups.items():
            if len(items) >= 2:
                execution.append({
                    "kind": "gate_group",
                    "gate_j": gate_j,
                    "gate_state": gate_state,
                    "targets": [(s["target_k"], t) for s, t in items],
                })
                total_bytes += sum(t.nbytes for _, t in items) + 16 + 8 * len(items)
                ops_after += 1 + len(items)
                gate_hoists += 1
            else:
                spec, table = items[0]
                execution.append({
                    "kind": "block",
                    "gate_j": gate_j,
                    "gate_state": gate_state,
                    "target_k": spec["target_k"],
                    "table": table,
                })
                total_bytes += table.nbytes + 16
                ops_after += 2

        self.execution_groups_ = execution
        self.replacement_bytes_ = int(total_bytes)
        self.replacement_original_ops_ = int(ops_before)
        self.replacement_ops_ = int(ops_after)
        self.replacement_pair_fusions_ = int(pair_fusions)
        self.replacement_gate_hoists_ = int(gate_hoists)

    def _base_lookup_tables(self):
        # ``Z0 @ base_coef_`` is a reference-state lookup sum.  Compile that
        # sparse linear map once so prediction can execute the same additions
        # without allocating a CSR matrix.  The lazy path keeps old pickles
        # compatible.
        tables = getattr(self, "base_lookup_", None)
        if tables is None:
            tables = []
            offset = 0
            for card in self.base_.oh_.cardinalities_:
                table = np.zeros(int(card), dtype=float)
                width = max(int(card) - 1, 0)
                if width:
                    table[1:] = self.base_coef_[offset : offset + width]
                offset += width
                tables.append(table)
            if offset != len(self.base_coef_):
                raise RuntimeError("base coefficient layout mismatch")
            self.base_lookup_ = tables
        return tables

    def _decision_from_projected_states(self, S, n_rows):
        score = self.base_._decision_from_states_with_lookup(
            S, self._base_lookup_tables(), self.intercept_, intercept_last=True
        )
        for group in self.execution_groups_:
            kind = group["kind"]
            if kind == "block":
                gate = S[4][:, group["gate_j"]] == group["gate_state"]
                score += gate * group["table"][S[self.block_level][:, group["target_k"]]]
            elif kind == "fused_pair":
                score += group["table"][
                    S[4][:, group["gate_j"]],
                    S[self.block_level][:, group["target_k"]],
                ]
            else:
                gate = S[4][:, group["gate_j"]] == group["gate_state"]
                if np.any(gate):
                    subtotal = np.zeros(n_rows, dtype=float)
                    for target_k, table in group["targets"]:
                        subtotal += table[S[self.block_level][:, target_k]]
                    score += gate * subtotal
        return score

    def decision_function_projected(self, X_selected):
        values = np.asarray(X_selected, dtype=float)
        if not self.execution_groups_:
            self.base_._ensure_execution_maps()
            states = self.base_.encoder_.transform_level_projected(
                values, self.base_.execution_level_, self.base_.feature_idx_
            )
            return self.base_._decision_from_execution_states_with_lookup(
                states, self._base_lookup_tables(), self.intercept_,
                intercept_last=True,
            )
        states = self.base_.encoder_.transform_projected(
            values, self.base_.feature_idx_
        )
        return self._decision_from_projected_states(states, len(values))

    def predict_proba_projected(self, X_selected):
        p = self._sigmoid(self.decision_function_projected(X_selected))
        return np.column_stack([1.0 - p, p])

    def decision_function(self, X):
        matrix = np.asarray(X, dtype=float)
        if not self.execution_groups_:
            self.base_._ensure_execution_maps()
            states = self.base_.encoder_.transform_level_columns(
                matrix, self.base_.execution_level_, self.base_.feature_idx_
            )
            return self.base_._decision_from_execution_states_with_lookup(
                states, self._base_lookup_tables(), self.intercept_,
                intercept_last=True,
            )
        states = self._states(matrix)
        return self._decision_from_projected_states(states, len(matrix))


    def export_ir(self, prefix):
        import json
        from pathlib import Path
        import numpy as np

        prefix = Path(prefix)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")
        base = self.base_
        arrays = {
            "feature_idx": np.asarray(base.feature_idx_, dtype=np.int16),
            "pairs": np.asarray(base.pairs_, dtype=np.int16).reshape(-1, 2),
            "fine_pairs": np.asarray(base.fine_pairs_, dtype=np.int16).reshape(-1, 2),
            "thresholds": np.asarray(base.encoder_.thresholds_, dtype=object),
            "feature_kinds": np.asarray(base.encoder_.feature_kinds_, dtype=object),
            "direct_state_mask": np.asarray(base.encoder_.direct_state_mask_, dtype=bool),
            "direct_state_cardinalities": np.asarray(base.encoder_.direct_state_cardinalities_, dtype=np.int32),
            "base_lookup": np.asarray(base.lookup_, dtype=object),
            "intercept": np.asarray([self.intercept_], dtype=np.float64),
            "levels": np.asarray(base.levels, dtype=np.int16),
            "block_level": np.asarray([self.block_level], dtype=np.int16),
        }
        for level in base.levels:
            arrays[f"map{level}"] = np.asarray(base.encoder_.maps_[level], dtype=object)
            if level > base.levels[0]:
                arrays[f"main_res{level}"] = np.asarray(base.main_residuals_[level], dtype=object)
                pairs = base.pairs_ if level <= 8 else base.fine_pairs_
                arrays[f"pair_res{level}"] = np.asarray(
                    [base.pair_residuals_[level][pair] for pair in pairs], dtype=object
                )
        groups = []
        for i, group in enumerate(self.execution_groups_):
            kind = group["kind"]
            entry = {"kind": kind, "gate_j": int(group["gate_j"])}
            if kind == "block":
                name = f"group_{i}_table"; arrays[name] = np.asarray(group["table"], dtype=np.float64)
                entry.update(gate_state=int(group["gate_state"]), target_k=int(group["target_k"]), array=name)
            elif kind == "fused_pair":
                name = f"group_{i}_table"; arrays[name] = np.asarray(group["table"], dtype=np.float64)
                entry.update(target_k=int(group["target_k"]), array=name)
            else:
                entry["gate_state"] = int(group["gate_state"]); targets = []
                for q, (target_k, table) in enumerate(group["targets"]):
                    name = f"group_{i}_target_{q}"; arrays[name] = np.asarray(table, dtype=np.float64)
                    targets.append({"target_k": int(target_k), "array": name})
                entry["targets"] = targets
            groups.append(entry)
        np.savez_compressed(npz_path, **arrays)
        manifest = {
            "format": "cerm-hybrid-quotient-block-ir-v3",
            "tree_or_split_sequence": False,
            "semantic_states": True,
            "head_count": 1,
            "input_features": int(base.encoder_.n_features_in_),
            "levels": list(base.levels),
            "max_bins": int(base.max_bins),
            "block_level": int(self.block_level),
            "encoder": {
                "kind": self.encoder_kind,
                "newton_prebins": self.newton_prebins,
                "newton_gain_l2": self.newton_gain_l2,
                "newton_min_hessian": self.newton_min_hessian,
            },
            "ranking": {
                "kind": self.ranking_kind,
                "l2": self.ranking_l2,
                "prefilter_multiplier": self.ranking_prefilter_multiplier,
            },
            "selection": {
                "primary": "full-or-complementary-fold dictionary",
                "secondary_confirmation": True,
                "selected_rank_mode": self.selected_hybrid_config_.rank_mode,
                "n_blocks": int(self.selected_hybrid_config_.n_blocks),
                "cell_shrinkage": float(self.selected_hybrid_config_.cell_shrinkage),
                "C": float(self.selected_hybrid_config_.C),
            },
            "replacement": {
                "objective": self.replacement_objective,
                "pair_fusions": int(self.replacement_pair_fusions_),
                "gate_hoists": int(self.replacement_gate_hoists_),
                "ops_before": int(self.replacement_original_ops_),
                "ops_after": int(self.replacement_ops_),
            },
            "groups": groups,
            "model_bytes_estimate": int(self.model_bytes_estimate_),
        }
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return npz_path, json_path

