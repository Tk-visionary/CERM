from __future__ import annotations

import heapq
import json
from concurrent.futures import ThreadPoolExecutor
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import effective_n_jobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from .cerm_state_design import ReferenceStateEncoder
from ..training_graph import (
    EncodedColumnBank,
    solve_binary_logistic_path,
    fit_binary_logistic_exact,
)


_DIRECT_KINDS = {
    "categorical_identity",
    "categorical_identity_bit",
    "missing_state",
    "missing",
}
_ORDERED_STATE_KINDS = {"categorical_quotient", "embedding_quotient"}


def _unique_if_at_most(values: np.ndarray, limit: int) -> np.ndarray | None:
    """Return all unique values iff their count may be <= ``limit``.

    For a finite array with at least ``limit + 1`` entries, observing
    ``limit + 1`` distinct values in that prefix proves that the full array
    cannot take the low-cardinality branch.  In that common continuous-feature
    case we can skip the historical full-array ``np.unique`` sort entirely.
    If the prefix does not prove high cardinality, fall back to the exact full
    unique computation.
    """
    limit = int(limit)
    if len(values) > limit:
        prefix = np.unique(values[: limit + 1])
        if len(prefix) > limit:
            return None
    return np.unique(values)


@dataclass(frozen=True)
class HierConfig:
    n_pairs: int = 20
    n_fine_pairs: int = 0
    max_main_level: int = 16
    C: float = 1.0


def _normalize_semantics(n_features, feature_kinds=None, feature_cardinalities=None):
    kinds = list(feature_kinds or ["numeric"] * n_features)
    if len(kinds) != n_features:
        raise ValueError("feature_kinds width mismatch")
    cards = list(feature_cardinalities or [None] * n_features)
    if len(cards) != n_features:
        raise ValueError("feature_cardinalities width mismatch")
    return kinds, cards


class NestedQuantileEncoder:
    """Nested finite-state encoder with explicit ordered/nominal semantics."""

    def __init__(
        self,
        max_bins: int = 16,
        levels: tuple[int, ...] = (4, 8, 16),
        feature_kinds: Sequence[str] | None = None,
        feature_cardinalities: Sequence[int | None] | None = None,
    ):
        if max(levels) != max_bins:
            raise ValueError("max(levels) must equal max_bins")
        self.max_bins = int(max_bins)
        self.levels = tuple(sorted(set(int(x) for x in levels)))
        self.feature_kinds = feature_kinds
        self.feature_cardinalities = feature_cardinalities

    @staticmethod
    def _contiguous_map(fine_card: int, target: int) -> np.ndarray:
        fine = np.arange(fine_card, dtype=np.int16)
        target = max(1, min(int(target), fine_card))
        mapping = np.floor(fine.astype(float) * target / fine_card).astype(np.int16)
        return np.minimum(mapping, target - 1)

    @staticmethod
    def _nominal_map(fine_card: int, level: int) -> np.ndarray:
        # Never introduce an arbitrary nominal ordering. Identity appears only
        # once the hierarchy can represent every category exactly.
        if fine_card <= level:
            return np.arange(fine_card, dtype=np.int16)
        return np.zeros(fine_card, dtype=np.int16)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        self.feature_kinds_, self.feature_cardinalities_ = _normalize_semantics(
            X.shape[1], self.feature_kinds, self.feature_cardinalities
        )
        probs = np.arange(1, self.max_bins, dtype=float) / self.max_bins
        self.thresholds_: list[np.ndarray] = []
        self.maps_: dict[int, list[np.ndarray]] = {level: [] for level in self.levels}
        self.cardinalities_: dict[int, np.ndarray] = {
            level: np.zeros(X.shape[1], dtype=np.int16) for level in self.levels
        }
        self.direct_state_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.direct_state_cardinalities_ = np.zeros(X.shape[1], dtype=np.int32)

        for j in range(X.shape[1]):
            kind = self.feature_kinds_[j]
            is_direct = kind in _DIRECT_KINDS or kind in _ORDERED_STATE_KINDS
            if is_direct:
                rounded = np.rint(X[:, j]).astype(np.int64)
                if not np.allclose(X[:, j], rounded):
                    raise ValueError(f"finite-state feature {j} contains non-integers")
                if np.any(rounded < 0):
                    raise ValueError("finite states must be non-negative")
                configured = self.feature_cardinalities_[j]
                fine_card = int(configured or (rounded.max(initial=0) + 1))
                if rounded.max(initial=0) >= fine_card:
                    raise ValueError("observed state exceeds configured cardinality")
                self.thresholds_.append(np.empty(0, dtype=np.float64))
                self.direct_state_mask_[j] = True
                self.direct_state_cardinalities_[j] = fine_card
                for level in self.levels:
                    if kind in _DIRECT_KINDS:
                        mapping = self._nominal_map(fine_card, level)
                    else:
                        mapping = self._contiguous_map(fine_card, level)
                    self.maps_[level].append(mapping)
                    self.cardinalities_[level][j] = int(mapping.max(initial=0)) + 1
                continue

            vals = X[:, j]
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                th = np.empty(0, dtype=np.float64)
            else:
                uniq = _unique_if_at_most(finite, self.max_bins)
                if uniq is not None and len(uniq) <= self.max_bins:
                    th = (uniq[:-1] + uniq[1:]) / 2.0
                else:
                    th = np.unique(np.quantile(finite, probs))
                    mn, mx = float(finite.min()), float(finite.max())
                    th = th[(th > mn) & (th < mx)]
            th = np.asarray(th, dtype=np.float64)
            self.thresholds_.append(th)
            fine_card = len(th) + 1
            for level in self.levels:
                mapping = self._contiguous_map(fine_card, level)
                self.maps_[level].append(mapping)
                self.cardinalities_[level][j] = int(mapping.max(initial=0)) + 1

        self.n_features_in_ = X.shape[1]
        return self

    def transform_fine(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("input width mismatch")
        out = np.empty(X.shape, dtype=np.int16)
        for j, th in enumerate(self.thresholds_):
            if self.direct_state_mask_[j]:
                state = np.rint(X[:, j]).astype(np.int64)
                card = self.direct_state_cardinalities_[j]
                if np.any((state < 0) | (state >= card)):
                    raise ValueError(f"feature {j} contains an unknown finite state")
                out[:, j] = state
            else:
                out[:, j] = np.searchsorted(th, X[:, j], side="right")
        return out

    def transform(self, X: np.ndarray) -> dict[int, np.ndarray]:
        fine = self.transform_fine(X)
        result = {}
        for level in self.levels:
            C = np.empty_like(fine)
            for j, mapping in enumerate(self.maps_[level]):
                C[:, j] = mapping[fine[:, j]]
            result[level] = C
        return result

    def transform_columns(
        self, X: np.ndarray, columns: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """Transform only selected raw columns with exactly the fitted maps.

        This is an exact projection of :meth:`transform`: for every level and
        requested column order, the returned matrix equals
        ``transform(X)[level][:, columns]``.  It avoids materializing states for
        raw features that are not referenced by a fitted finite-state program.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("input width mismatch")
        columns = np.asarray(tuple(int(j) for j in columns), dtype=np.int64)
        if np.any((columns < 0) | (columns >= self.n_features_in_)):
            raise ValueError("selected state column is out of range")
        fine = np.empty((len(X), len(columns)), dtype=np.int16)
        for out_j, raw_j in enumerate(columns):
            th = self.thresholds_[int(raw_j)]
            if self.direct_state_mask_[int(raw_j)]:
                state = np.rint(X[:, int(raw_j)]).astype(np.int64)
                card = self.direct_state_cardinalities_[int(raw_j)]
                if np.any((state < 0) | (state >= card)):
                    raise ValueError(
                        f"feature {int(raw_j)} contains an unknown finite state"
                    )
                fine[:, out_j] = state
            else:
                fine[:, out_j] = np.searchsorted(
                    th, X[:, int(raw_j)], side="right"
                )
        result = {}
        for level in self.levels:
            C = np.empty_like(fine)
            maps = self.maps_[level]
            for out_j, raw_j in enumerate(columns):
                C[:, out_j] = maps[int(raw_j)][fine[:, out_j]]
            result[level] = C
        return result


    def transform_projected(
        self, X_selected: np.ndarray, columns: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """Transform an already projected set of fitted raw columns.

        ``X_selected[:, j]`` must equal the original input column
        ``columns[j]``.  The result is bitwise identical to
        ``transform(X)[level][:, columns]`` while avoiding state generation for
        unreferenced features.
        """
        values = np.asarray(X_selected, dtype=float)
        columns = np.asarray(tuple(int(j) for j in columns), dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != len(columns):
            raise ValueError("projected input width mismatch")
        if np.any((columns < 0) | (columns >= self.n_features_in_)):
            raise ValueError("selected state column is out of range")
        fine = np.empty(values.shape, dtype=np.int16)
        for out_j, raw_j in enumerate(columns):
            raw_j = int(raw_j)
            if self.direct_state_mask_[raw_j]:
                state = np.rint(values[:, out_j]).astype(np.int64)
                card = self.direct_state_cardinalities_[raw_j]
                if np.any((state < 0) | (state >= card)):
                    raise ValueError(
                        f"feature {raw_j} contains an unknown finite state"
                    )
                fine[:, out_j] = state
            else:
                fine[:, out_j] = np.searchsorted(
                    self.thresholds_[raw_j], values[:, out_j], side="right"
                )
        result = {}
        for level in self.levels:
            mapped = np.empty_like(fine)
            maps = self.maps_[level]
            for out_j, raw_j in enumerate(columns):
                mapped[:, out_j] = maps[int(raw_j)][fine[:, out_j]]
            result[level] = mapped
        return result

    def _direct_level_thresholds(self, raw_j: int, level: int):
        """Return exact parent-level thresholds when the quotient is contiguous.

        Numeric quotient maps are fitted by adjacent merges.  When their labels
        are the canonical nondecreasing ``0, 1, ...`` sequence, every parent
        boundary is one of the fitted fine thresholds.  Searching only those
        boundaries is exactly equivalent to fine ``searchsorted`` followed by
        ``maps_[level]`` and avoids comparisons against boundaries that the
        requested quotient level immediately merges away.
        """
        raw_j = int(raw_j)
        level = int(level)
        if self.direct_state_mask_[raw_j]:
            return None
        cache = getattr(self, "_direct_level_threshold_cache_", None)
        if cache is None:
            cache = {}
            self._direct_level_threshold_cache_ = cache
        key = (raw_j, level)
        if key in cache:
            return cache[key]
        mapping = np.asarray(self.maps_[level][raw_j], dtype=np.int64)
        diff = np.diff(mapping)
        cardinality = int(self.cardinalities_[level][raw_j])
        if (
            len(mapping)
            and int(mapping[0]) == 0
            and np.all((diff >= 0) & (diff <= 1))
            and int(mapping[-1]) + 1 == cardinality
        ):
            boundaries = np.flatnonzero(diff != 0)
            thresholds = np.asarray(self.thresholds_[raw_j], dtype=np.float64)[
                boundaries
            ]
            if len(thresholds) + 1 == cardinality:
                cache[key] = thresholds
                return thresholds
        cache[key] = None
        return None

    def transform_level_columns(
        self, X: np.ndarray, level: int, columns: Sequence[int]
    ) -> np.ndarray:
        """Transform selected raw columns at one fitted quotient level."""
        level = int(level)
        if level not in self.levels:
            raise ValueError("requested state level is unavailable")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("input width mismatch")
        columns = np.asarray(tuple(int(j) for j in columns), dtype=np.int64)
        if np.any((columns < 0) | (columns >= self.n_features_in_)):
            raise ValueError("selected state column is out of range")
        out = np.empty((len(X), len(columns)), dtype=np.int16)
        maps = self.maps_[level]
        for out_j, raw_j in enumerate(columns):
            raw_j = int(raw_j)
            if self.direct_state_mask_[raw_j]:
                fine = np.rint(X[:, raw_j]).astype(np.int64)
                card = self.direct_state_cardinalities_[raw_j]
                if np.any((fine < 0) | (fine >= card)):
                    raise ValueError(
                        f"feature {raw_j} contains an unknown finite state"
                    )
                out[:, out_j] = maps[raw_j][fine]
                continue
            direct_thresholds = self._direct_level_thresholds(raw_j, level)
            if direct_thresholds is not None:
                out[:, out_j] = np.searchsorted(
                    direct_thresholds, X[:, raw_j], side="right"
                )
            else:
                fine = np.searchsorted(
                    self.thresholds_[raw_j], X[:, raw_j], side="right"
                )
                out[:, out_j] = maps[raw_j][fine]
        return out

    def transform_level_projected(
        self, X_selected: np.ndarray, level: int, columns: Sequence[int]
    ) -> np.ndarray:
        """Transform already-projected columns at one fitted quotient level."""
        level = int(level)
        if level not in self.levels:
            raise ValueError("requested state level is unavailable")
        values = np.asarray(X_selected, dtype=float)
        columns = np.asarray(tuple(int(j) for j in columns), dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != len(columns):
            raise ValueError("projected input width mismatch")
        if np.any((columns < 0) | (columns >= self.n_features_in_)):
            raise ValueError("selected state column is out of range")
        out = np.empty(values.shape, dtype=np.int16)
        maps = self.maps_[level]
        for out_j, raw_j in enumerate(columns):
            raw_j = int(raw_j)
            column = values[:, out_j]
            if self.direct_state_mask_[raw_j]:
                fine = np.rint(column).astype(np.int64)
                card = self.direct_state_cardinalities_[raw_j]
                if np.any((fine < 0) | (fine >= card)):
                    raise ValueError(
                        f"feature {raw_j} contains an unknown finite state"
                    )
                out[:, out_j] = maps[raw_j][fine]
                continue
            direct_thresholds = self._direct_level_thresholds(raw_j, level)
            if direct_thresholds is not None:
                out[:, out_j] = np.searchsorted(
                    direct_thresholds, column, side="right"
                )
            else:
                fine = np.searchsorted(
                    self.thresholds_[raw_j], column, side="right"
                )
                out[:, out_j] = maps[raw_j][fine]
        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None):
        """Fit the quantile hierarchy and emit training states in one pass.

        The historical implementation called ``fit(X)`` and then traversed
        every feature again in ``transform(X)``.  During fitting we already
        hold the exact rounded direct state or the fitted numeric thresholds
        for the current feature, so the corresponding fine state can be
        emitted immediately and mapped to every quotient level.  Thresholds,
        maps, state labels, and output dtypes are unchanged.
        """
        # NewtonNestedEncoder has a target-dependent fit implementation.  Keep
        # its historical two-stage semantics unless it provides its own
        # specialized path.
        if type(self) is not NestedQuantileEncoder:
            return self.fit(X, y).transform(X)

        X = np.asarray(X, dtype=float)
        self.feature_kinds_, self.feature_cardinalities_ = _normalize_semantics(
            X.shape[1], self.feature_kinds, self.feature_cardinalities
        )
        probs = np.arange(1, self.max_bins, dtype=float) / self.max_bins
        self.thresholds_ = []
        self.maps_ = {level: [] for level in self.levels}
        self.cardinalities_ = {
            level: np.zeros(X.shape[1], dtype=np.int16) for level in self.levels
        }
        self.direct_state_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.direct_state_cardinalities_ = np.zeros(X.shape[1], dtype=np.int32)
        result = {
            level: np.empty(X.shape, dtype=np.int16) for level in self.levels
        }

        for j in range(X.shape[1]):
            kind = self.feature_kinds_[j]
            is_direct = kind in _DIRECT_KINDS or kind in _ORDERED_STATE_KINDS
            if is_direct:
                rounded = np.rint(X[:, j]).astype(np.int64)
                if not np.allclose(X[:, j], rounded):
                    raise ValueError(f"finite-state feature {j} contains non-integers")
                if np.any(rounded < 0):
                    raise ValueError("finite states must be non-negative")
                configured = self.feature_cardinalities_[j]
                fine_card = int(configured or (rounded.max(initial=0) + 1))
                if rounded.max(initial=0) >= fine_card:
                    raise ValueError("observed state exceeds configured cardinality")
                self.thresholds_.append(np.empty(0, dtype=np.float64))
                self.direct_state_mask_[j] = True
                self.direct_state_cardinalities_[j] = fine_card
                for level in self.levels:
                    if kind in _DIRECT_KINDS:
                        mapping = self._nominal_map(fine_card, level)
                    else:
                        mapping = self._contiguous_map(fine_card, level)
                    self.maps_[level].append(mapping)
                    self.cardinalities_[level][j] = int(mapping.max(initial=0)) + 1
                    result[level][:, j] = mapping[rounded]
                continue

            vals = X[:, j]
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                th = np.empty(0, dtype=np.float64)
            else:
                uniq = _unique_if_at_most(finite, self.max_bins)
                if uniq is not None and len(uniq) <= self.max_bins:
                    th = (uniq[:-1] + uniq[1:]) / 2.0
                else:
                    th = np.unique(np.quantile(finite, probs))
                    mn, mx = float(finite.min()), float(finite.max())
                    th = th[(th > mn) & (th < mx)]
            th = np.asarray(th, dtype=np.float64)
            self.thresholds_.append(th)
            fine_card = len(th) + 1
            fine = np.searchsorted(th, vals, side="right")
            for level in self.levels:
                mapping = self._contiguous_map(fine_card, level)
                self.maps_[level].append(mapping)
                self.cardinalities_[level][j] = int(mapping.max(initial=0)) + 1
                result[level][:, j] = mapping[fine]

        self.n_features_in_ = X.shape[1]
        return result

    @property
    def threshold_count_(self) -> int:
        return int(sum(len(x) for x in self.thresholds_))

    @property
    def threshold_bytes_(self) -> int:
        return int(sum(x.nbytes for x in self.thresholds_))


class NewtonNestedEncoder(NestedQuantileEncoder):
    """Nested ordered quotient formed by adjacent regularized Newton merges."""

    def __init__(
        self,
        prebins: int = 64,
        gain_l2: float = 5.0,
        min_hessian: float = 1.0,
        max_bins: int = 16,
        levels: tuple[int, ...] = (4, 8, 16),
        feature_kinds=None,
        feature_cardinalities=None,
    ):
        super().__init__(max_bins, levels, feature_kinds, feature_cardinalities)
        self.prebins = int(prebins)
        self.gain_l2 = float(gain_l2)
        self.min_hessian = float(min_hessian)

    def _numeric_partition(self, values: np.ndarray, y: np.ndarray):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.empty(0), {l: np.zeros(1, np.int16) for l in self.levels}
        uniq = _unique_if_at_most(finite, self.prebins)
        if uniq is not None and len(uniq) <= self.prebins:
            pre_th = (uniq[:-1] + uniq[1:]) / 2.0
        else:
            q = np.arange(1, self.prebins) / self.prebins
            pre_th = np.unique(np.quantile(finite, q))
            pre_th = pre_th[(pre_th > finite.min()) & (pre_th < finite.max())]
        pre = np.searchsorted(pre_th, values, side="right")
        n_pre = int(pre.max(initial=0)) + 1
        if n_pre <= 1:
            return np.empty(0), {l: np.zeros(1, np.int16) for l in self.levels}

        p = np.clip(float(np.mean(y)), 1e-5, 1 - 1e-5)
        g = np.asarray(y, float) - p
        h = np.full(len(y), p * (1 - p), dtype=float)
        G = np.bincount(pre, weights=g, minlength=n_pre).astype(float)
        H = np.bincount(pre, weights=h, minlength=n_pre).astype(float)

        left = np.arange(n_pre, dtype=np.int32)
        right = np.arange(n_pre, dtype=np.int32)
        prev = np.arange(-1, n_pre - 1, dtype=np.int32)
        nxt = np.arange(1, n_pre + 1, dtype=np.int32)
        nxt[-1] = -1
        alive = np.ones(n_pre, dtype=bool)
        version = np.zeros(n_pre, dtype=np.int64)

        def merge_loss(a, b):
            ga, gb, ha, hb = G[a], G[b], H[a], H[b]
            loss = ga * ga / (ha + self.gain_l2) + gb * gb / (hb + self.gain_l2)
            loss -= (ga + gb) ** 2 / (ha + hb + self.gain_l2)
            if ha < self.min_hessian or hb < self.min_hessian:
                loss -= 1e6
            return float(loss)

        heap = []
        for a in range(n_pre - 1):
            heapq.heappush(heap, (merge_loss(a, a + 1), a, version[a], a + 1, version[a + 1]))

        targets = sorted({min(l, n_pre) for l in self.levels}, reverse=True)
        snapshots = {}

        def ordered_segments():
            out = []
            node = next((i for i in range(n_pre) if alive[i] and prev[i] == -1), -1)
            while node != -1:
                out.append((int(left[node]), int(right[node])))
                node = int(nxt[node])
            return out

        count = n_pre
        if count in targets:
            snapshots[count] = ordered_segments()
        stop = min(targets)
        while count > stop:
            while heap:
                _, a, va, b, vb = heapq.heappop(heap)
                if alive[a] and alive[b] and version[a] == va and version[b] == vb and nxt[a] == b:
                    break
            else:
                break
            G[a] += G[b]; H[a] += H[b]; right[a] = right[b]
            nb = int(nxt[b]); nxt[a] = nb
            if nb != -1: prev[nb] = a
            alive[b] = False; version[a] += 1; version[b] += 1
            pa = int(prev[a])
            if pa != -1:
                heapq.heappush(heap, (merge_loss(pa, a), pa, version[pa], a, version[a]))
            if nb != -1:
                heapq.heappush(heap, (merge_loss(a, nb), a, version[a], nb, version[nb]))
            count -= 1
            if count in targets:
                snapshots[count] = ordered_segments()

        # If a target count was skipped because n_pre is smaller, use nearest
        # available finer snapshot (identity at that resolution).
        for target in targets:
            if target not in snapshots:
                available = [k for k in snapshots if k <= target]
                snapshots[target] = snapshots[max(available)] if available else ordered_segments()

        fine_count = min(self.max_bins, n_pre)
        fine_segments = snapshots[fine_count]
        final_thresholds = np.asarray(
            [pre_th[r] for _, r in fine_segments[:-1]], dtype=np.float64
        )
        maps = {}
        for level in self.levels:
            parent = snapshots[min(level, n_pre)]
            mapping = np.zeros(len(fine_segments), dtype=np.int16)
            for i, (fl, fr) in enumerate(fine_segments):
                for q, (pl, pr) in enumerate(parent):
                    if pl <= fl and fr <= pr:
                        mapping[i] = q
                        break
            maps[level] = mapping
        return final_thresholds, maps

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if y is None:
            raise ValueError("NewtonNestedEncoder requires y")
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
        self.feature_kinds_, self.feature_cardinalities_ = _normalize_semantics(
            X.shape[1], self.feature_kinds, self.feature_cardinalities
        )
        self.thresholds_ = []
        self.maps_ = {level: [] for level in self.levels}
        self.cardinalities_ = {level: np.zeros(X.shape[1], np.int16) for level in self.levels}
        self.direct_state_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.direct_state_cardinalities_ = np.zeros(X.shape[1], dtype=np.int32)
        for j, kind in enumerate(self.feature_kinds_):
            if kind in _DIRECT_KINDS or kind in _ORDERED_STATE_KINDS:
                rounded = np.rint(X[:, j]).astype(np.int64)
                card = int(self.feature_cardinalities_[j] or (rounded.max(initial=0) + 1))
                self.thresholds_.append(np.empty(0, dtype=np.float64))
                self.direct_state_mask_[j] = True
                self.direct_state_cardinalities_[j] = card
                for level in self.levels:
                    mapping = self._nominal_map(card, level) if kind in _DIRECT_KINDS else self._contiguous_map(card, level)
                    self.maps_[level].append(mapping)
                    self.cardinalities_[level][j] = int(mapping.max(initial=0)) + 1
            else:
                th, maps = self._numeric_partition(X[:, j], y)
                self.thresholds_.append(th)
                for level in self.levels:
                    self.maps_[level].append(maps[level])
                    self.cardinalities_[level][j] = int(maps[level].max(initial=0)) + 1
        self.n_features_in_ = X.shape[1]
        return self


def _binary_mutual_info_from_joint_codes(
    joint_codes: np.ndarray,
    cardinality: int,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Exact binary MI from precombined ``state * 2 + y`` codes.

    This is the reduction half of :func:`_binary_mutual_info_validated`.  It
    exists so pair ranking can construct the final joint/target code directly
    into one reusable int64 buffer instead of allocating both ``pair_code`` and
    ``pair_code * 2 + y`` for every candidate.  The histogram and floating
    reduction are intentionally identical to the historical implementation.
    """
    card = int(cardinality)
    if card <= 0:
        return 0.0
    if sample_weight is None:
        joint = np.bincount(joint_codes, minlength=card * 2).reshape(card, 2)
        joint = joint.astype(np.float64, copy=False)
    else:
        joint = np.bincount(
            joint_codes,
            weights=np.asarray(sample_weight, dtype=np.float64),
            minlength=card * 2,
        ).reshape(card, 2)
    n = float(joint.sum())
    if n <= 0.0:
        return 0.0
    row = joint.sum(axis=1, keepdims=True)
    col = joint.sum(axis=0, keepdims=True)
    expected_num = row @ col
    mask = joint > 0.0
    return float(
        np.sum((joint[mask] / n) * np.log((joint[mask] * n) / expected_num[mask]))
    )


def _binary_mutual_info_validated(
    code: np.ndarray, y: np.ndarray, cardinality: int | None = None,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Inner binary-MI kernel for aligned non-negative int64 arrays."""
    if code.size == 0:
        return 0.0
    card = (
        int(np.max(code, initial=0)) + 1
        if cardinality is None
        else int(cardinality)
    )
    if card <= 0:
        return 0.0
    target_code = code * 2 + y
    return _binary_mutual_info_from_joint_codes(
        target_code, card, sample_weight
    )


def _validated_binary_mi_inputs(code, y):
    code64 = np.asarray(code, dtype=np.int64)
    y64 = np.asarray(y, dtype=np.int64)
    if code64.ndim != 1 or y64.ndim != 1 or len(code64) != len(y64):
        raise ValueError("code and y must be aligned one-dimensional arrays")
    if np.any(code64 < 0):
        raise ValueError("finite codes must be non-negative")
    if np.any((y64 < 0) | (y64 > 1)):
        raise ValueError("_binary_mutual_info requires a binary target encoded as 0/1")
    return code64, y64


def _binary_mutual_info(code, y, sample_weight=None):
    """Exact mutual information for a finite code and a binary target."""
    code64, y64 = _validated_binary_mi_inputs(code, y)
    return _binary_mutual_info_validated(code64, y64, sample_weight=sample_weight)


def _mi_columns(C, y, sample_weight=None):
    C64 = np.asarray(C, dtype=np.int64)
    y64 = np.asarray(y, dtype=np.int64)
    if C64.ndim != 2 or y64.ndim != 1 or len(C64) != len(y64):
        raise ValueError("state matrix and binary target must be aligned")
    if np.any(C64 < 0):
        raise ValueError("finite codes must be non-negative")
    if np.any((y64 < 0) | (y64 > 1)):
        raise ValueError("_mi_columns requires a binary target encoded as 0/1")
    cards = C64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    target_code = np.empty(len(y64), dtype=np.int64)
    scores = np.empty(C64.shape[1], dtype=np.float64)
    for j in range(C64.shape[1]):
        np.multiply(C64[:, j], 2, out=target_code)
        np.add(target_code, y64, out=target_code)
        scores[j] = _binary_mutual_info_from_joint_codes(
            target_code, int(cards[j]), sample_weight
        )
    return scores


def _select_features(C, y, max_features, sample_weight=None):
    return np.argsort(
        -_mi_columns(C, y, sample_weight=sample_weight), kind="stable"
    )[: min(max_features, C.shape[1])]



def _push_top_ranked(heap, item, limit):
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _use_bounded_pair_heap(d, max_pairs):
    total = int(d) * max(int(d) - 1, 0) // 2
    return int(d) > 512 and 0 < int(max_pairs) < max(total // 8, 1)


def _rank_pairs(C, y, max_pairs, feature_limit=32, sample_weight=None, n_jobs=1):
    d = min(C.shape[1], feature_limit)
    use_heap = _use_bounded_pair_heap(d, max_pairs)
    pair_view = C[:, :d]
    C64 = (
        np.asarray(pair_view, dtype=np.int64, order="F")
        if pair_view.size >= 500_000
        else np.asarray(pair_view, dtype=np.int64)
    )
    y64 = np.asarray(y, dtype=np.int64)
    cards = C64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    target_code = np.empty(len(C64), dtype=np.int64)
    mi1 = []
    for j in range(d):
        np.multiply(C64[:, j], 2, out=target_code)
        np.add(target_code, y64, out=target_code)
        mi1.append(
            _binary_mutual_info_from_joint_codes(
                target_code, int(cards[j]), sample_weight
            )
        )

    def score_js(js):
        local = []
        local_code = np.empty(len(y64), dtype=np.int64)
        for j in js:
            left = C64[:, j]
            for k in range(j + 1, d):
                np.multiply(left, int(cards[k]), out=local_code)
                np.add(local_code, C64[:, k], out=local_code)
                np.multiply(local_code, 2, out=local_code)
                np.add(local_code, y64, out=local_code)
                mij = _binary_mutual_info_from_joint_codes(
                    local_code, int(cards[j]) * int(cards[k]), sample_weight
                )
                item = (mij - max(mi1[j], mi1[k]), mij, j, k)
                if use_heap:
                    _push_top_ranked(local, item, int(max_pairs))
                else:
                    local.append(item)
        return local

    workers = min(max(1, effective_n_jobs(n_jobs)), max(d - 1, 1))
    use_threads = workers > 1 and d <= 32 and len(C64) >= 50_000
    if use_threads:
        chunks = [
            chunk.tolist()
            for chunk in np.array_split(np.arange(d - 1), workers)
            if len(chunk)
        ]
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            parts = list(pool.map(score_js, chunks))
        ranked = [item for part in parts for item in part]
        if use_heap:
            ranked = heapq.nlargest(int(max_pairs), ranked)
    else:
        ranked = []
        for j in range(d):
            left = C64[:, j]
            for k in range(j + 1, d):
                np.multiply(left, int(cards[k]), out=target_code)
                np.add(target_code, C64[:, k], out=target_code)
                np.multiply(target_code, 2, out=target_code)
                np.add(target_code, y64, out=target_code)
                mij = _binary_mutual_info_from_joint_codes(
                    target_code, int(cards[j]) * int(cards[k]), sample_weight
                )
                item = (mij - max(mi1[j], mi1[k]), mij, j, k)
                if use_heap:
                    _push_top_ranked(ranked, item, int(max_pairs))
                else:
                    ranked.append(item)
    ranked.sort(reverse=True)
    return [(j, k) for *_, j, k in (ranked if use_heap else ranked[:max_pairs])]

def _cell_newton_gain_from_gh(code, g, h, l2):
    card = int(np.max(code, initial=0)) + 1
    G = np.bincount(code, weights=g, minlength=card)
    H = np.bincount(code, weights=h, minlength=card)
    return 0.5 * float(np.sum(G * G / (H + l2)))


def _cell_newton_gain(code, y, p, l2, sample_weight=None):
    g = np.asarray(y, float) - p
    h = np.maximum(p * (1 - p), 1e-8)
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        g = g * weights
        h = h * weights
    return _cell_newton_gain_from_gh(code, g, h, l2)


def _rank_pairs_newton(
    C, y, max_pairs, feature_limit=32, l2=5.0, p=None, sample_weight=None, n_jobs=1
):
    d = min(C.shape[1], feature_limit)
    use_heap = _use_bounded_pair_heap(d, max_pairs)
    if p is None:
        mean = np.clip(np.average(y, weights=sample_weight), 1e-5, 1 - 1e-5)
        p = np.full(len(y), mean)

    g = np.asarray(y, float) - p
    h = np.maximum(p * (1 - p), 1e-8)
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        g = g * weights
        h = h * weights

    pair_view = C[:, :d]
    C64 = (
        np.asarray(pair_view, dtype=np.int64, order="F")
        if pair_view.size >= 500_000
        else np.asarray(pair_view, dtype=np.int64)
    )
    cards = C64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    marginal = [
        _cell_newton_gain_from_gh(C64[:, j], g, h, l2) for j in range(d)
    ]

    def score_js(js):
        local = []
        joint = np.empty(len(C64), dtype=np.int64)
        for j in js:
            left = C64[:, j]
            for k in range(j + 1, d):
                np.multiply(left, int(cards[k]), out=joint)
                np.add(joint, C64[:, k], out=joint)
                gain = _cell_newton_gain_from_gh(joint, g, h, l2)
                item = (gain - max(marginal[j], marginal[k]), gain, j, k)
                if use_heap:
                    _push_top_ranked(local, item, int(max_pairs))
                else:
                    local.append(item)
        return local

    workers = min(max(1, effective_n_jobs(n_jobs)), max(d - 1, 1))
    work = len(C64) * d * max(d - 1, 0) // 2
    if workers > 1 and len(C64) >= 50_000 and d <= 40 and work >= 5_000_000:
        chunks = [
            chunk.tolist()
            for chunk in np.array_split(np.arange(d - 1), workers)
            if len(chunk)
        ]
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            parts = list(pool.map(score_js, chunks))
        ranked = [item for part in parts for item in part]
        if use_heap:
            ranked = heapq.nlargest(int(max_pairs), ranked)
    else:
        ranked = []
        joint = np.empty(len(C64), dtype=np.int64)
        for j in range(d):
            left = C64[:, j]
            for k in range(j + 1, d):
                np.multiply(left, int(cards[k]), out=joint)
                np.add(joint, C64[:, k], out=joint)
                gain = _cell_newton_gain_from_gh(joint, g, h, l2)
                item = (gain - max(marginal[j], marginal[k]), gain, j, k)
                if use_heap:
                    _push_top_ranked(ranked, item, int(max_pairs))
                else:
                    ranked.append(item)
    ranked.sort(reverse=True)
    return [(j, k) for *_, j, k in (ranked if use_heap else ranked[:max_pairs])]

def _rank_pairs_prefilter_mi(C, y, max_pairs, feature_limit=32, l2=5.0, multiplier=4, p=None, sample_weight=None):
    d = min(C.shape[1], feature_limit)
    total = d * (d - 1) // 2
    keep = min(total, max(max_pairs, max_pairs * multiplier))
    pre = _rank_pairs_newton(C, y, keep, feature_limit=d, l2=l2, p=p, sample_weight=sample_weight)
    mi1 = _mi_columns(C[:, :d], y, sample_weight=sample_weight)
    ranked = []
    C64 = np.asarray(C[:, :d], dtype=np.int64, order="F")
    y64 = np.asarray(y, dtype=np.int64)
    cards = C64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    target_code = np.empty(len(C64), dtype=np.int64)
    for j, k in pre:
        np.multiply(C64[:, j], int(cards[k]), out=target_code)
        np.add(target_code, C64[:, k], out=target_code)
        np.multiply(target_code, 2, out=target_code)
        np.add(target_code, y64, out=target_code)
        mij = _binary_mutual_info_from_joint_codes(
            target_code,
            int(cards[j]) * int(cards[k]),
            sample_weight,
        )
        ranked.append((mij - max(mi1[j], mi1[k]), mij, j, k))
    ranked.sort(reverse=True)
    return [(j, k) for *_, j, k in ranked[:max_pairs]]

def _crossfit_main_probs(C, y, max_features, random_state, sample_weight=None):
    idx = _select_features(C, y, max_features, sample_weight)
    codes = C[:, idx].astype(np.int64)
    splits = min(3, int(np.bincount(y).min()))
    if splits < 2:
        return np.full(len(y), np.average(y, weights=sample_weight))
    cv = StratifiedKFold(splits, shuffle=True, random_state=random_state)
    out = np.empty(len(y), dtype=float)
    for tr, va in cv.split(codes, y):
        enc = ReferenceStateEncoder()
        Ztr = enc.fit_transform(codes[tr]); Zva = enc.transform(codes[va])
        clf = LogisticRegression(C=.2, solver="liblinear", max_iter=1000, random_state=random_state)
        clf.fit(Ztr, y[tr], sample_weight=None if sample_weight is None else sample_weight[tr]); out[va] = clf.predict_proba(Zva)[:, 1]
    return np.clip(out, 1e-5, 1 - 1e-5)


def _rank_pairs_prefilter_mi(C, y, max_pairs, feature_limit=32, l2=5.0, multiplier=4, p=None, sample_weight=None):
    d = min(C.shape[1], feature_limit)
    total = d * (d - 1) // 2
    keep = min(total, max(max_pairs, max_pairs * multiplier))
    pre = _rank_pairs_newton(C, y, keep, feature_limit=d, l2=l2, p=p, sample_weight=sample_weight)
    mi1 = _mi_columns(C[:, :d], y, sample_weight=sample_weight)
    ranked = []
    C64 = np.asarray(C[:, :d], dtype=np.int64, order="F")
    cards = C64.max(axis=0, initial=0).astype(np.int64, copy=False) + 1
    for j, k in pre:
        joint = C64[:, j] * int(cards[k]) + C64[:, k]
        mij = _binary_mutual_info_validated(
            joint, np.asarray(y, dtype=np.int64), int(cards[j]) * int(cards[k]), sample_weight
        )
        ranked.append((mij - max(mi1[j], mi1[k]), mij, j, k))
    ranked.sort(reverse=True)
    return [(j, k) for *_, j, k in ranked[:max_pairs]]


def _rank_pairs_diversified(
    C,
    y,
    max_pairs,
    feature_limit=32,
    l2=5.0,
    random_state=0,
    sample_weight=None,
):
    """Return a bounded MI/residual pair bank without changing row weights.

    MI remains the primary candidate source.  A one-third residual-Newton
    supplement is added from cross-fitted main-effect residuals so a pair that
    is weak marginally but strong after the main effects can enter the bank.
    The result is deliberately opt-in: it changes candidate ordering, while
    the default ``ranking_kind='mi'`` path remains untouched.
    """
    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        return []
    d = min(int(C.shape[1]), int(feature_limit))
    total = d * max(d - 1, 0) // 2
    if total <= 0:
        return []
    pool_limit = min(total, max(max_pairs * 2, max_pairs + 4))
    mi = _rank_pairs(
        C,
        y,
        pool_limit,
        feature_limit=d,
        sample_weight=sample_weight,
    )
    probabilities = _crossfit_main_probs(
        C,
        y,
        max_features=min(int(C.shape[1]), int(feature_limit)),
        random_state=int(random_state),
        sample_weight=sample_weight,
    )
    residual = _rank_pairs_newton(
        C,
        y,
        pool_limit,
        feature_limit=d,
        l2=float(l2),
        p=probabilities,
        sample_weight=sample_weight,
    )

    # Keep a deterministic 2/3 MI : 1/3 residual quota, then fill any gaps
    # caused by duplicate pairs from the other bank.  All rows participate in
    # both banks; no empirical distribution is changed by this diversification.
    mi_quota = (2 * max_pairs + 2) // 3
    residual_quota = max_pairs - mi_quota
    result = []
    seen = set()

    def append_unique(values, limit=None):
        if limit is not None and limit <= 0:
            return
        added = 0
        for pair in values:
            pair = (int(pair[0]), int(pair[1]))
            if pair in seen:
                continue
            result.append(pair)
            seen.add(pair)
            added += 1
            if len(result) >= max_pairs or (limit is not None and added >= limit):
                return

    append_unique(mi, mi_quota)
    append_unique(residual, residual_quota)
    append_unique(mi)
    append_unique(residual)
    return result[:max_pairs]


def _residual_state_map(child_to_parent):
    """Return the exact residual code map in parent-major order.

    This is an exact vectorized lowering of the previous nested Python loops.
    The first child of every parent keeps reference code zero; all remaining
    children receive consecutive residual codes ordered first by parent label
    and then by child label.
    """
    child_to_parent = np.asarray(child_to_parent, dtype=np.int64)
    n = len(child_to_parent)
    mapping = np.zeros(n, dtype=np.int32)
    if n == 0:
        return mapping
    child = np.arange(n, dtype=np.int64)
    order = np.lexsort((child, child_to_parent))
    ordered_parent = child_to_parent[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = ordered_parent[1:] != ordered_parent[:-1]
    residual_children = order[~first]
    mapping[residual_children] = np.arange(1, len(residual_children) + 1, dtype=np.int32)
    return mapping


def _parent_labels_from_fine(child_map, parent_map, child_card):
    """Compose child and parent quotient maps without repeated searches."""
    child_map = np.asarray(child_map, dtype=np.int64)
    parent_map = np.asarray(parent_map, dtype=np.int64)
    first_index = np.full(int(child_card), len(child_map), dtype=np.int64)
    np.minimum.at(first_index, child_map, np.arange(len(child_map), dtype=np.int64))
    if np.any(first_index == len(child_map)):
        raise ValueError("child quotient contains an unreachable state")
    return parent_map[first_index]


def _pair_parent_map(child_card_j, child_card_k, map_j, map_k, parent_card_k):
    map_j = np.asarray(map_j, dtype=np.int64)[: int(child_card_j)]
    map_k = np.asarray(map_k, dtype=np.int64)[: int(child_card_k)]
    return (map_j[:, None] * int(parent_card_k) + map_k[None, :]).ravel()


class HierarchicalResidualCERM:
    def __init__(
        self,
        max_features=64,
        pair_feature_limit=24,
        search_profile="full_exact",
        fixed_C=None,
        max_bins=16,
        selection_subsample=1.0,
        n_jobs=1,
        random_state=20260802,
        selection_rule="best",
        feature_kinds=None,
        feature_cardinalities=None,
        encoder_kind="quantile",
        newton_prebins=64,
        newton_gain_l2=5.0,
        newton_min_hessian=1.0,
        ranking_kind="mi",
        ranking_l2=5.0,
        ranking_prefilter_multiplier=4,
        cost_per_byte=0.0,
        cost_per_operator=0.0,
    ):
        self.max_features=int(max_features); self.pair_feature_limit=int(pair_feature_limit)
        self.search_profile=str(search_profile)
        self.fixed_C=None if fixed_C is None else float(fixed_C)
        self.max_bins=int(max_bins)
        self.levels=tuple(level for level in (4,8,16) if level <= self.max_bins)
        self.ranking_level=8 if self.max_bins >= 8 else 4
        self.block_level=self.max_bins
        self.selection_subsample=float(selection_subsample)
        self.n_jobs=n_jobs
        self.random_state=int(random_state); self.selection_rule=selection_rule
        self.feature_kinds=feature_kinds; self.feature_cardinalities=feature_cardinalities
        if encoder_kind not in {"quantile","newton"}: raise ValueError("invalid encoder_kind")
        if ranking_kind not in {"mi","newton","residual_newton","newton_prefilter_mi","diversified"}: raise ValueError("invalid ranking_kind")
        self.encoder_kind=encoder_kind; self.newton_prebins=int(newton_prebins)
        self.newton_gain_l2=float(newton_gain_l2); self.newton_min_hessian=float(newton_min_hessian)
        self.ranking_kind=ranking_kind; self.ranking_l2=float(ranking_l2)
        self.ranking_prefilter_multiplier=int(ranking_prefilter_multiplier)
        self.cost_per_byte=float(cost_per_byte); self.cost_per_operator=float(cost_per_operator)

    def _new_encoder(self):
        kwargs=dict(
            max_bins=self.max_bins,
            levels=self.levels,
            feature_kinds=self.feature_kinds,
            feature_cardinalities=self.feature_cardinalities,
        )
        if self.encoder_kind=="newton":
            return NewtonNestedEncoder(
                prebins=self.newton_prebins,
                gain_l2=self.newton_gain_l2,
                min_hessian=self.newton_min_hessian,
                **kwargs,
            )
        return NestedQuantileEncoder(**kwargs)

    def _selection_sample(self, X, y, sample_weight=None, *, seed_offset=0):
        if self.selection_subsample >= 1.0:
            return X, y, sample_weight
        rng=np.random.default_rng(self.random_state + int(seed_offset) + 49979687)
        selected=[]
        for cls in np.unique(y):
            idx=np.flatnonzero(y==cls)
            keep=max(8, int(math.ceil(self.selection_subsample * len(idx))))
            keep=min(len(idx), keep)
            if keep < len(idx):
                idx=rng.choice(idx, size=keep, replace=False)
            selected.append(np.asarray(idx,dtype=np.int64))
        rows=np.concatenate(selected)
        rng.shuffle(rows)
        return X[rows], y[rows], None if sample_weight is None else sample_weight[rows]

    def _rank(self,C,y,max_pairs,sample_weight=None):
        if self.ranking_kind=="mi": return _rank_pairs(C,y,max_pairs,self.pair_feature_limit,sample_weight,self.n_jobs)
        if self.ranking_kind=="newton": return _rank_pairs_newton(C,y,max_pairs,self.pair_feature_limit,self.ranking_l2,sample_weight=sample_weight,n_jobs=self.n_jobs)
        if self.ranking_kind=="diversified":
            return _rank_pairs_diversified(
                C,
                y,
                max_pairs,
                self.pair_feature_limit,
                self.ranking_l2,
                self.random_state,
                sample_weight,
            )
        p=_crossfit_main_probs(C,y,self.max_features,self.random_state,sample_weight)
        if self.ranking_kind=="residual_newton": return _rank_pairs_newton(C,y,max_pairs,self.pair_feature_limit,self.ranking_l2,p,sample_weight,self.n_jobs)
        return _rank_pairs_prefilter_mi(C,y,max_pairs,self.pair_feature_limit,self.ranking_l2,self.ranking_prefilter_multiplier,p,sample_weight)

    def _prepare_residual_maps(self):
        idx=self.feature_idx_
        self.main_residuals_={}
        for level, parent_level in zip(self.levels[1:], self.levels[:-1]):
            values=[]
            for raw_j in idx:
                child_map=self.encoder_.maps_[level][raw_j]
                parent_map=self.encoder_.maps_[parent_level][raw_j]
                child_card=int(self.encoder_.cardinalities_[level][raw_j])
                parent_labels=_parent_labels_from_fine(child_map,parent_map,child_card)
                values.append(_residual_state_map(parent_labels))
            self.main_residuals_[level]=values
        self.main_res8_=self.main_residuals_.get(8,[])
        self.main_res16_=self.main_residuals_.get(16,[])

        self.pair_residuals_={level:{} for level in self.levels[1:]}
        fine_pairs=set(self.fine_pairs_)
        for j,k in self.pairs_:
            rj,rk=int(idx[j]),int(idx[k])
            for level,parent_level in zip(self.levels[1:],self.levels[:-1]):
                if level > 8 and (j,k) not in fine_pairs:
                    continue
                child_card_j=int(self.encoder_.cardinalities_[level][rj])
                child_card_k=int(self.encoder_.cardinalities_[level][rk])
                parent_card_k=int(self.encoder_.cardinalities_[parent_level][rk])
                parent_j=_parent_labels_from_fine(
                    self.encoder_.maps_[level][rj],
                    self.encoder_.maps_[parent_level][rj],
                    child_card_j,
                )
                parent_k=_parent_labels_from_fine(
                    self.encoder_.maps_[level][rk],
                    self.encoder_.maps_[parent_level][rk],
                    child_card_k,
                )
                pair_parent=_pair_parent_map(
                    child_card_j,child_card_k,parent_j,parent_k,parent_card_k
                )
                self.pair_residuals_[level][(j,k)]=_residual_state_map(pair_parent)
        self.pair_res8_=self.pair_residuals_.get(8,{})
        self.pair_res16_=self.pair_residuals_.get(16,{})

    def _build_codes_from_states(self, states):
        coarse=self.levels[0]
        n_rows=len(states[coarse])
        main_levels=[level for level in self.levels if level <= self.config_.max_main_level]
        pair_levels=[level for level in self.levels if level <= min(self.max_bins,8)]
        n_columns=(
            states[coarse].shape[1] * len(main_levels)
            + len(self.pairs_) * len(pair_levels)
            + len(self.fine_pairs_) * int(16 in self.levels)
        )
        if n_columns==0:
            return np.zeros((n_rows,1),dtype=np.int64)
        codes=np.empty((n_rows,n_columns),dtype=np.int64)
        column=0
        for j in range(states[coarse].shape[1]):
            codes[:,column]=states[coarse][:,j]; column+=1
            for level in main_levels[1:]:
                codes[:,column]=self.main_residuals_[level][j][states[level][:,j]]
                column+=1

        joint=np.empty(n_rows,dtype=np.int64)
        fine_pairs=set(self.fine_pairs_)
        for j,k in self.pairs_:
            rj,rk=int(self.feature_idx_[j]),int(self.feature_idx_[k])
            card_k=int(self.encoder_.cardinalities_[coarse][rk])
            np.multiply(states[coarse][:,j],card_k,out=joint,casting="unsafe")
            joint+=states[coarse][:,k]
            codes[:,column]=joint; column+=1
            for level in pair_levels[1:]:
                card_k=int(self.encoder_.cardinalities_[level][rk])
                np.multiply(states[level][:,j],card_k,out=joint,casting="unsafe")
                joint+=states[level][:,k]
                codes[:,column]=self.pair_residuals_[level][(j,k)][joint]
                column+=1
            if 16 in self.levels and (j,k) in fine_pairs:
                card_k=int(self.encoder_.cardinalities_[16][rk])
                np.multiply(states[16][:,j],card_k,out=joint,casting="unsafe")
                joint+=states[16][:,k]
                codes[:,column]=self.pair_residuals_[16][(j,k)][joint]
                column+=1
        if column!=n_columns:
            raise RuntimeError("semantic code layout mismatch")
        return codes

    def _states_for_X(self,X):
        matrix = np.asarray(X, float)
        # The fitted semantic program only references ``feature_idx_``.  The
        # encoder's projected transform is an exact slice of ``transform``
        # (including direct-state validation and map lookup order), so avoid
        # materializing state columns for adapted features that cannot affect
        # this model.
        return self.encoder_.transform_columns(matrix, self.feature_idx_)
    def _codes(self,X): return self._build_codes_from_states(self._states_for_X(X))

    def _compile_lookup(self):
        coef=self.clf_.coef_[0]; self.lookup_=[]; offset=0
        for card in self.oh_.cardinalities_:
            tab=np.zeros(int(card),float); width=max(int(card)-1,0)
            if width: tab[1:]=coef[offset:offset+width]
            offset+=width; self.lookup_.append(tab)
        if offset!=len(coef): raise RuntimeError("coefficient layout mismatch")
        self.intercept_=float(self.clf_.intercept_[0])

    def _prepare_execution_maps(self):
        """Compile exact code maps from one nested quotient execution level."""
        coarse = self.levels[0]
        main_levels = tuple(
            level for level in self.levels
            if level <= int(self.config_.max_main_level)
        )
        pair_levels = tuple(
            level for level in self.levels if level <= min(int(self.max_bins), 8)
        )
        needed = list(main_levels) + list(pair_levels)
        if 16 in self.levels and len(self.fine_pairs_) > 0:
            needed.append(16)
        execution_level = int(max(needed)) if needed else int(coarse)
        self.execution_level_ = execution_level
        execution_cards = np.asarray(
            self.encoder_.cardinalities_[execution_level][self.feature_idx_],
            dtype=np.int64,
        )
        self.execution_cardinalities_ = execution_cards
        parent_cache = {}

        def labels(raw_feature, level):
            key = (int(raw_feature), int(level))
            cached = parent_cache.get(key)
            if cached is not None:
                return cached
            raw_feature = int(raw_feature)
            child_card = int(
                self.encoder_.cardinalities_[execution_level][raw_feature]
            )
            if int(level) == execution_level:
                value = np.arange(child_card, dtype=np.int32)
            else:
                value = _parent_labels_from_fine(
                    self.encoder_.maps_[execution_level][raw_feature],
                    self.encoder_.maps_[int(level)][raw_feature],
                    child_card,
                ).astype(np.int32, copy=False)
            parent_cache[key] = value
            return value

        main_maps = []
        code_maxima = []
        for position, raw_feature in enumerate(self.feature_idx_):
            feature_maps = []
            for index, level in enumerate(main_levels):
                level_state = labels(int(raw_feature), int(level))
                code_map = (
                    level_state
                    if index == 0
                    else self.main_residuals_[level][position][level_state]
                )
                code_map = np.asarray(code_map, dtype=np.int32)
                feature_maps.append(code_map)
                code_maxima.append(int(np.max(code_map, initial=0)))
            main_maps.append(tuple(feature_maps))
        self.execution_main_code_maps_ = tuple(main_maps)

        fine_pairs = set(self.fine_pairs_)
        pair_plans = []
        for j, k in self.pairs_:
            raw_j = int(self.feature_idx_[j])
            raw_k = int(self.feature_idx_[k])
            card_j = int(execution_cards[j])
            card_k = int(execution_cards[k])
            exec_j = np.arange(card_j, dtype=np.int64)[:, None]
            exec_k = np.arange(card_k, dtype=np.int64)[None, :]
            code_maps = []
            for level in pair_levels:
                state_j = labels(raw_j, level)[exec_j]
                state_k = labels(raw_k, level)[exec_k]
                level_card_k = int(self.encoder_.cardinalities_[level][raw_k])
                joint = state_j * level_card_k + state_k
                code = (
                    joint
                    if level == coarse
                    else self.pair_residuals_[level][(j, k)][joint]
                )
                code = np.asarray(code, dtype=np.int32).reshape(-1)
                code_maps.append(code)
                code_maxima.append(int(np.max(code, initial=0)))
            if 16 in self.levels and (j, k) in fine_pairs:
                state_j = labels(raw_j, 16)[exec_j]
                state_k = labels(raw_k, 16)[exec_k]
                level_card_k = int(self.encoder_.cardinalities_[16][raw_k])
                joint = state_j * level_card_k + state_k
                code = np.asarray(
                    self.pair_residuals_[16][(j, k)][joint],
                    dtype=np.int32,
                ).reshape(-1)
                code_maps.append(code)
                code_maxima.append(int(np.max(code, initial=0)))
            pair_plans.append((int(j), int(k), card_k, tuple(code_maps)))
        self.execution_pair_plans_ = tuple(pair_plans)
        self.execution_code_maxima_ = tuple(code_maxima)

    def _ensure_execution_maps(self):
        if not hasattr(self, "execution_main_code_maps_"):
            self._prepare_execution_maps()

    def _decision_from_execution_states_with_lookup(
        self, execution_states, lookup, intercept, *, intercept_last=False
    ):
        """Execute historical semantic lookup order from one state bank."""
        self._ensure_execution_maps()
        execution_states = np.asarray(execution_states)
        score = (
            np.zeros(len(execution_states), dtype=float)
            if intercept_last
            else np.full(len(execution_states), intercept, dtype=float)
        )
        scratch = np.empty(len(execution_states), dtype=np.float64)
        column = 0

        def accumulate(state):
            nonlocal column, score
            table = lookup[column]
            full_domain = (
                column < len(self.execution_code_maxima_)
                and self.execution_code_maxima_[column] < len(table)
            )
            if full_domain:
                np.take(table, state, out=scratch)
                np.add(score, scratch, out=score)
            else:
                valid = (state >= 0) & (state < len(table))
                score[valid] += table[state[valid]]
            column += 1

        for position, code_maps in enumerate(self.execution_main_code_maps_):
            execution_state = execution_states[:, position]
            for code_map in code_maps:
                accumulate(code_map[execution_state])
        joint = np.empty(len(execution_states), dtype=np.int64)
        for left, right, right_cardinality, code_maps in self.execution_pair_plans_:
            np.multiply(
                execution_states[:, left], right_cardinality,
                out=joint, casting="unsafe",
            )
            joint += execution_states[:, right]
            for code_map in code_maps:
                accumulate(code_map[joint])
        if column != len(lookup):
            raise RuntimeError("semantic compiled lookup layout mismatch")
        if intercept_last:
            score += intercept
        return score

    def _fit_structure(self,X,y,cfg,sample_weight=None):
        self.encoder_=self._new_encoder()
        if X.shape[1] >= 512:
            self.encoder_.fit(X, y)
            ranking_states = self.encoder_.transform_level_columns(
                X, self.ranking_level, range(X.shape[1])
            )
            self.feature_idx_=_select_features(
                ranking_states,y,self.max_features,sample_weight
            )
            states=self.encoder_.transform_columns(X,self.feature_idx_)
        else:
            states_all=self.encoder_.fit_transform(X,y)
            self.feature_idx_=_select_features(
                states_all[self.ranking_level],y,self.max_features,sample_weight
            )
            states={l:C[:,self.feature_idx_] for l,C in states_all.items()}
        rank=self._rank(states[self.ranking_level],y,max(cfg.n_pairs,1),sample_weight); self.pairs_=rank[:cfg.n_pairs]; self.fine_pairs_=self.pairs_[:min(cfg.n_fine_pairs,len(self.pairs_))]; self.config_=cfg
        self._prepare_residual_maps(); codes=self._build_codes_from_states(states)
        self.oh_=ReferenceStateEncoder(); Z=self.oh_.fit_transform(codes)
        # Ephemeral TrainingGraph artifacts.  Composite learners opt in and
        # consume these immediately; standalone estimators do not retain the
        # training rows after fit.
        if getattr(self, "_retain_fit_training_graph", False):
            self._fit_training_states_ = states
            self._fit_training_codes_ = codes
            self._fit_training_design_ = Z
        self.clf_=fit_binary_logistic_exact(
            Z, y, C=cfg.C, random_state=self.random_state, max_iter=1500, sample_weight=sample_weight
        )
        self.classes_=self.clf_.classes_; self.design_dim_=int(Z.shape[1]); self.nonzero_coef_=int(np.count_nonzero(np.abs(self.clf_.coef_)>1e-10)); self._compile_lookup(); self._prepare_execution_maps()
        self.operator_columns_=int(codes.shape[1]); self.lookup_bytes_=int(sum(t.nbytes for t in self.lookup_))
        self.model_bytes_estimate_=int(self.lookup_bytes_+self.encoder_.threshold_bytes_+len(self.feature_idx_)*2+(len(self.pairs_)+len(self.fine_pairs_))*4)
        return self

    def _maximal_code_metadata(self, n_features, n_pairs, n_fine_pairs):
        metadata=[]
        for j in range(int(n_features)):
            for level in self.levels:
                metadata.append((f"main{level}",j))
        for pair_idx in range(int(n_pairs)):
            for level in self.levels:
                if level <= 8 or pair_idx < int(n_fine_pairs):
                    metadata.append((f"pair{level}",pair_idx))
        return metadata

    @staticmethod
    def _candidate_code_columns(metadata,cfg):
        selected=[]
        for column,(kind,idx) in enumerate(metadata):
            if kind.startswith("main"):
                level=int(kind[4:])
                if level <= cfg.max_main_level:
                    selected.append(column)
            elif kind.startswith("pair"):
                level=int(kind[4:])
                if level <= 8 and idx < cfg.n_pairs:
                    selected.append(column)
                elif level > 8 and idx < cfg.n_fine_pairs:
                    selected.append(column)
        return selected

    def fit(self,X,y,sample_weight=None):
        X=np.asarray(X,float); y=np.asarray(y,int)
        sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        Xsel,ysel,wsel=self._selection_sample(X,y,sample_weight)
        split = train_test_split(
            np.arange(len(Xsel)), test_size=.22, stratify=ysel,
            random_state=self.random_state
        )
        ia, iv = split
        Xa, Xv, ya, yv = Xsel[ia], Xsel[iv], ysel[ia], ysel[iv]
        wa = None if wsel is None else wsel[ia]
        wv = None if wsel is None else wsel[iv]
        if self.max_bins==16:
            candidates=[HierConfig(0,0,8,.2),HierConfig(0,0,16,.2),HierConfig(0,0,16,1.),HierConfig(10,0,16,.2),HierConfig(10,0,16,1.),HierConfig(30,0,16,.2),HierConfig(30,0,16,1.),HierConfig(30,5,16,.2),HierConfig(30,5,16,1.)]
        elif self.max_bins==8:
            candidates=[HierConfig(0,0,4,.2),HierConfig(0,0,8,.2),HierConfig(0,0,8,1.),HierConfig(10,0,8,.2),HierConfig(10,0,8,1.),HierConfig(30,0,8,.2),HierConfig(30,0,8,1.)]
        else:
            candidates=[HierConfig(0,0,4,.2),HierConfig(0,0,4,1.),HierConfig(10,0,4,.2),HierConfig(10,0,4,1.),HierConfig(30,0,4,.2),HierConfig(30,0,4,1.)]
        if self.fixed_C is not None:
            candidates=list(dict.fromkeys(
                HierConfig(c.n_pairs,c.n_fine_pairs,c.max_main_level,self.fixed_C)
                for c in candidates
            ))
        enc=self._new_encoder()
        if Xa.shape[1] >= 512:
            enc.fit(Xa, ya)
            Sa_rank=enc.transform_level_columns(
                Xa,self.ranking_level,range(Xa.shape[1])
            )
            idx=_select_features(Sa_rank,ya,self.max_features,wa)
            Sa=enc.transform_columns(Xa,idx)
            Sv=enc.transform_columns(Xv,idx)
        else:
            Sa_all=enc.fit_transform(Xa,ya); Sv_all=enc.transform(Xv)
            idx=_select_features(Sa_all[self.ranking_level],ya,self.max_features,wa)
            Sa={l:C[:,idx] for l,C in Sa_all.items()}; Sv={l:C[:,idx] for l,C in Sv_all.items()}
        max_pairs=max(c.n_pairs for c in candidates)
        rank=self._rank(Sa[self.ranking_level],ya,max(max_pairs,1),wa)

        cache=object.__new__(HierarchicalResidualCERM); cache.__dict__.update(self.__dict__)
        cache.encoder_=enc; cache.feature_idx_=idx
        max_pair_count=min(max_pairs,len(rank)); max_fine=max(c.n_fine_pairs for c in candidates)
        max_cfg=HierConfig(max_pair_count,min(max_fine,max_pair_count),self.max_bins,.2)
        cache.config_=max_cfg; cache.pairs_=rank[:max_pair_count]
        cache.fine_pairs_=cache.pairs_[:max_cfg.n_fine_pairs]
        cache._prepare_residual_maps()
        max_ca=cache._build_codes_from_states(Sa); max_cv=cache._build_codes_from_states(Sv)
        bank=EncodedColumnBank.build(max_ca,max_cv,ReferenceStateEncoder)
        metadata=self._maximal_code_metadata(len(idx),len(cache.pairs_),len(cache.fine_pairs_))
        if len(metadata)!=max_ca.shape[1]:
            raise RuntimeError("maximal hierarchical code metadata mismatch")

        scores=[]; design_groups={}
        for cfg in candidates:
            key=(cfg.n_pairs,cfg.n_fine_pairs,cfg.max_main_level)
            design_groups.setdefault(key,[]).append(cfg)

        groups = list(design_groups.values())
        group_workers = min(max(1, effective_n_jobs(self.n_jobs)), len(groups))

        def evaluate_group(group):
            representative=group[0]
            code_columns=self._candidate_code_columns(metadata,representative)
            Za,Zv=bank.view(code_columns)
            path=solve_binary_logistic_path(
                Za,ya,Zv,[cfg.C for cfg in group],
                random_state=self.random_state,max_iter=1500,sample_weight=wa,
                n_jobs=1 if group_workers > 1 else self.n_jobs,
            )
            actual_pairs=min(representative.n_pairs,len(cache.pairs_))
            actual_fine=min(representative.n_fine_pairs,actual_pairs,len(cache.fine_pairs_))
            lookup_bytes=bank.lookup_bytes(code_columns)
            bytes_est=int(lookup_bytes+enc.threshold_bytes_+len(idx)*2+(actual_pairs+actual_fine)*4)
            ops=int(len(code_columns))
            local=[]
            for cfg in group:
                p=np.clip(path[float(cfg.C)].valid_probability,1e-10,1-1e-10)
                losses=-(yv*np.log(p)+(1-yv)*np.log(1-p))
                mean_loss=float(np.mean(losses) if wv is None else np.average(losses,weights=wv))
                if wv is None:
                    se_loss=float(losses.std(ddof=1)/math.sqrt(len(losses)))
                else:
                    centered=losses-mean_loss
                    se_loss=float(np.sqrt(np.average(centered*centered,weights=wv)/max(np.count_nonzero(wv),1)))
                objective=float(mean_loss+self.cost_per_byte*bytes_est+self.cost_per_operator*ops)
                local.append(dict(mean_loss=mean_loss,se_loss=se_loss,objective=objective,cfg=cfg,bytes=bytes_est,operators=ops,design_dim=int(Za.shape[1])))
            return local

        if group_workers > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=group_workers) as pool:
                group_rows = list(pool.map(evaluate_group, groups))
            scores = [row for rows_group in group_rows for row in rows_group]
        else:
            for group in groups:
                scores.extend(evaluate_group(group))
        scores.sort(key=lambda row:(row["objective"],row["mean_loss"])); self.validation_scores_=scores
        self.training_graph_columns_=int(max_ca.shape[1]); self.training_graph_design_dim_=int(bank.train.shape[1])
        self.selection_rows_=int(len(Xsel))
        if self.selection_rule=="best": chosen=scores[0]
        else:
            best=min(scores,key=lambda row:row["mean_loss"])
            tol=best["se_loss"] if self.selection_rule=="one_se_cost" else min(.0025,.25*best["se_loss"])
            eligible=[row for row in scores if row["mean_loss"]<=best["mean_loss"]+tol]
            chosen=min(eligible,key=lambda row:(row["bytes"],row["operators"],row["mean_loss"]))
        self.best_config_=chosen["cfg"]; self.selected_validation_loss_=chosen["mean_loss"]; self.selected_validation_bytes_=chosen["bytes"]
        return self._fit_structure(X,y,self.best_config_,sample_weight)

    def _consume_fit_training_graph(self):
        """Return and clear exact artifacts generated by the latest final fit."""
        names = (
            "_fit_training_states_",
            "_fit_training_codes_",
            "_fit_training_design_",
        )
        if not all(hasattr(self, name) for name in names):
            return None
        artifacts = tuple(getattr(self, name) for name in names)
        for name in names:
            delattr(self, name)
        return artifacts

    def _decision_from_codes(self, codes):
        codes=np.asarray(codes,dtype=np.int64)
        score=np.full(len(codes),self.intercept_,float)
        for col,tab in enumerate(self.lookup_):
            values=codes[:,col]; valid=(values>=0)&(values<len(tab)); score[valid]+=tab[values[valid]]
        return score

    def _lookup_full_domain_flags(self):
        flags = getattr(self, "lookup_full_domain_", None)
        if flags is not None:
            return flags
        maxima = []
        coarse = self.levels[0]
        main_levels = [
            level for level in self.levels if level <= self.config_.max_main_level
        ]
        pair_levels = [level for level in self.levels if level <= min(self.max_bins, 8)]
        for j in range(len(self.feature_idx_)):
            raw_j = int(self.feature_idx_[j])
            maxima.append(int(np.max(self.encoder_.maps_[coarse][raw_j], initial=0)))
            for level in main_levels[1:]:
                maxima.append(int(np.max(self.main_residuals_[level][j], initial=0)))
        fine_pairs = set(self.fine_pairs_)
        for j, k in self.pairs_:
            rj, rk = int(self.feature_idx_[j]), int(self.feature_idx_[k])
            card_j = int(self.encoder_.cardinalities_[coarse][rj])
            card_k = int(self.encoder_.cardinalities_[coarse][rk])
            maxima.append(card_j * card_k - 1)
            for level in pair_levels[1:]:
                maxima.append(
                    int(np.max(self.pair_residuals_[level][(j, k)], initial=0))
                )
            if 16 in self.levels and (j, k) in fine_pairs:
                maxima.append(
                    int(np.max(self.pair_residuals_[16][(j, k)], initial=0))
                )
        if len(maxima) != len(self.lookup_):
            raise RuntimeError("semantic lookup domain layout mismatch")
        flags = np.asarray(
            [maximum < len(table) for maximum, table in zip(maxima, self.lookup_)],
            dtype=bool,
        )
        self.lookup_full_domain_ = flags
        return flags

    def _decision_from_states_with_lookup(
        self, states, lookup, intercept, *, intercept_last=False
    ):
        """Execute the semantic code stream without materializing ``codes``.

        Code columns are generated and accumulated in exactly the same order as
        ``_build_codes_from_states`` followed by ``_decision_from_codes``.
        ``intercept_last`` exists for the hybrid refit path, whose historical
        sparse matvec adds its intercept after the row sum.
        """
        coarse = self.levels[0]
        n_rows = len(states[coarse])
        score = (
            np.zeros(n_rows, dtype=float)
            if intercept_last
            else np.full(n_rows, intercept, dtype=float)
        )
        main_levels = [
            level for level in self.levels if level <= self.config_.max_main_level
        ]
        pair_levels = [level for level in self.levels if level <= min(self.max_bins, 8)]
        column = 0
        full_domain = self._lookup_full_domain_flags()

        def accumulate(values):
            nonlocal column, score
            table = lookup[column]
            if full_domain[column]:
                score += table[values]
            else:
                valid = (values >= 0) & (values < len(table))
                score[valid] += table[values[valid]]
            column += 1

        for j in range(states[coarse].shape[1]):
            accumulate(states[coarse][:, j])
            for level in main_levels[1:]:
                accumulate(self.main_residuals_[level][j][states[level][:, j]])

        joint = np.empty(n_rows, dtype=np.int64)
        fine_pairs = set(self.fine_pairs_)
        for j, k in self.pairs_:
            rk = int(self.feature_idx_[k])
            card_k = int(self.encoder_.cardinalities_[coarse][rk])
            np.multiply(states[coarse][:, j], card_k, out=joint, casting="unsafe")
            joint += states[coarse][:, k]
            accumulate(joint)
            for level in pair_levels[1:]:
                card_k = int(self.encoder_.cardinalities_[level][rk])
                np.multiply(states[level][:, j], card_k, out=joint, casting="unsafe")
                joint += states[level][:, k]
                accumulate(self.pair_residuals_[level][(j, k)][joint])
            if 16 in self.levels and (j, k) in fine_pairs:
                card_k = int(self.encoder_.cardinalities_[16][rk])
                np.multiply(states[16][:, j], card_k, out=joint, casting="unsafe")
                joint += states[16][:, k]
                accumulate(self.pair_residuals_[16][(j, k)][joint])

        if column != len(lookup):
            raise RuntimeError("semantic lookup layout mismatch")
        if intercept_last:
            score += intercept
        return score

    def _decision_from_states(self, states):
        return self._decision_from_states_with_lookup(
            states, self.lookup_, self.intercept_, intercept_last=False
        )

    def _probability_from_codes(self, codes):
        score=self._decision_from_codes(codes)
        return 1/(1+np.exp(-np.clip(score,-40,40)))

    def decision_function_projected(self, X_selected):
        """Score already-projected ``feature_idx_`` columns exactly."""
        self._ensure_execution_maps()
        states = self.encoder_.transform_level_projected(
            X_selected, self.execution_level_, self.feature_idx_
        )
        return self._decision_from_execution_states_with_lookup(
            states, self.lookup_, self.intercept_
        )

    def predict_proba_projected(self, X_selected):
        s = self.decision_function_projected(X_selected)
        p = 1 / (1 + np.exp(-np.clip(s, -40, 40)))
        return np.column_stack([1 - p, p])

    def decision_function(self,X):
        self._ensure_execution_maps()
        matrix = np.asarray(X, dtype=float)
        states = self.encoder_.transform_level_columns(
            matrix, self.execution_level_, self.feature_idx_
        )
        return self._decision_from_execution_states_with_lookup(
            states, self.lookup_, self.intercept_
        )
    def predict_proba(self,X):
        s=self.decision_function(X); p=1/(1+np.exp(-np.clip(s,-40,40))); return np.column_stack([1-p,p])
    def predict_proba_reference(self,X): return self.clf_.predict_proba(self.oh_.transform(self._codes(X)))

    def export_ir(self,prefix):
        prefix=Path(prefix); npz=prefix.with_suffix('.npz'); js=prefix.with_suffix('.json')
        arrays={
            "feature_idx":np.asarray(self.feature_idx_,np.int16),
            "pairs":np.asarray(self.pairs_,np.int16).reshape(-1,2),
            "fine_pairs":np.asarray(self.fine_pairs_,np.int16).reshape(-1,2),
            "thresholds":np.asarray(self.encoder_.thresholds_,dtype=object),
            "feature_kinds":np.asarray(self.encoder_.feature_kinds_,dtype=object),
            "direct_state_mask":self.encoder_.direct_state_mask_,
            "direct_state_cardinalities":self.encoder_.direct_state_cardinalities_,
            "base_lookup":np.asarray(self.lookup_,dtype=object),
            "lookup":np.asarray(self.lookup_,dtype=object),
            "intercept":np.asarray([self.intercept_]),
            "levels":np.asarray(self.levels,dtype=np.int16),
        }
        for level in self.levels:
            arrays[f"map{level}"]=np.asarray(self.encoder_.maps_[level],dtype=object)
            if level>self.levels[0]:
                arrays[f"main_res{level}"]=np.asarray(self.main_residuals_[level],dtype=object)
                pairs=self.pairs_ if level<=8 else self.fine_pairs_
                arrays[f"pair_res{level}"]=np.asarray(
                    [self.pair_residuals_[level][pair] for pair in pairs],dtype=object
                )
        np.savez_compressed(npz,**arrays)
        manifest=dict(
            format='cerm-hierarchical-ir-v3',head_count=1,semantic_states=True,
            input_features=self.encoder_.n_features_in_,encoder_kind=self.encoder_kind,
            ranking_kind=self.ranking_kind,levels=list(self.levels),max_bins=self.max_bins,
            config=self.config_.__dict__,model_bytes_estimate=self.model_bytes_estimate_,
        )
        js.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return npz,js

    def independent_threshold_count(self,X):
        total=0
        for bins in self.levels: total+=NestedQuantileEncoder(max_bins=bins,levels=(bins,),feature_kinds=self.feature_kinds,feature_cardinalities=self.feature_cardinalities).fit(np.asarray(X,float)).threshold_count_
        return int(total)


def _hier_operator_metadata(model: HierarchicalResidualCERM):
    metadata=[]
    main_levels=[level for level in model.levels if level <= model.config_.max_main_level]
    for local_j,raw_j in enumerate(model.feature_idx_):
        for position,level in enumerate(main_levels):
            metadata.append({"kind":"main","scope":[int(raw_j)],"level":level,"residual":position>0})
    fine_pairs=set(model.fine_pairs_)
    for j,k in model.pairs_:
        scope=[int(model.feature_idx_[j]),int(model.feature_idx_[k])]
        for level in model.levels:
            if level <= 8 or (j,k) in fine_pairs:
                metadata.append({"kind":"pair","scope":scope,"level":level,"residual":level>4})
    return metadata


HierarchicalResidualCERM.operator_metadata = _hier_operator_metadata
