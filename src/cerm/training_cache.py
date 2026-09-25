from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HistogramStats:
    G: np.ndarray
    H: np.ndarray


class PackedStateMatrix:
    """Compact storage for non-negative finite-state features."""

    def __init__(self, states: np.ndarray):
        array = np.asarray(states)
        if array.ndim != 2 or np.any(array < 0):
            raise ValueError("states must be a non-negative 2D array")
        maximum = int(array.max(initial=0))
        dtype = np.uint8 if maximum <= 255 else np.uint16
        self.array = np.ascontiguousarray(array, dtype=dtype)
        self.cardinalities = (
            self.array.max(axis=0, initial=0).astype(np.int64) + 1
        )

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes + self.cardinalities.nbytes)


@dataclass(frozen=True)
class ValueHistogramStats:
    """Finite-state cell mass and sums for one or more aligned value columns."""

    mass: np.ndarray
    sums: np.ndarray


class FiniteStateValueHistogramCache:
    """Reusable multi-value histograms for finite-state main/pair candidates.

    This is the regression/vector-target analogue of :class:`NewtonHistogramCache`.
    The native training-core ABI already accepts an ``(n_samples, n_values)``
    value matrix, so multiple target directions are accumulated in one state/pair
    pass. A leading all-one value column recovers exact cell mass without a new
    native ABI.
    """

    def __init__(
        self,
        states: np.ndarray,
        values: np.ndarray,
        *,
        feature_limit: int | None = None,
        pairs: Iterable[tuple[int, int]] | None = None,
        backend: str = "auto",
        n_jobs: int | None = None,
    ):
        self.states = PackedStateMatrix(states)
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or len(matrix) != len(self.states.array):
            raise ValueError("values must have shape (n_samples, n_values)")
        if matrix.shape[1] < 1:
            raise ValueError("values must contain at least one column")
        if not np.isfinite(matrix).all():
            raise ValueError("values contain NaN or infinity")
        self.values = np.ascontiguousarray(matrix, dtype=np.float64)
        self.feature_limit = min(
            self.states.array.shape[1],
            self.states.array.shape[1]
            if feature_limit is None
            else int(feature_limit),
        )
        if self.feature_limit < 1:
            raise ValueError("feature_limit must be positive")

        if pairs is None:
            pair_keys = [
                (j, k)
                for j in range(self.feature_limit)
                for k in range(j + 1, self.feature_limit)
            ]
        else:
            pair_keys = [(int(j), int(k)) for j, k in pairs]
        for j, k in pair_keys:
            if not (0 <= j < k < self.feature_limit):
                raise ValueError(
                    "pair indices must satisfy 0 <= left < right < feature_limit"
                )
        self.pair_keys = tuple(pair_keys)

        backend = str(backend).lower()
        if backend not in {"auto", "python", "native"}:
            raise ValueError("backend must be 'auto', 'python', or 'native'")
        self.backend_requested_ = backend
        self.backend_ = "python"
        self.native_error_ = None
        self.n_jobs_ = NewtonHistogramCache._resolve_n_jobs(self, n_jobs)

        if backend != "python" and self.states.array.dtype == np.uint8:
            try:
                self._build_native()
                self.backend_ = "native"
                return
            except Exception as exc:
                self.native_error_ = repr(exc)
                if backend == "native":
                    raise
        elif backend == "native":
            raise NotImplementedError(
                "native training-cache ABI supports uint8 state banks only"
            )
        self._build_python()

    @staticmethod
    def _split_aggregates(aggregate: np.ndarray) -> ValueHistogramStats:
        return ValueHistogramStats(
            mass=np.asarray(aggregate[:, 0], dtype=np.float64).copy(),
            sums=np.asarray(aggregate[:, 1:], dtype=np.float64).copy(),
        )

    def _augmented_values(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.column_stack(
                [np.ones(len(self.values), dtype=np.float64), self.values]
            )
        )

    def _build_native(self) -> None:
        from ._internal.cerm_training_core_runtime import load_native_training_core

        core = load_native_training_core()
        states = self.states.array[:, : self.feature_limit]
        cards = self.states.cardinalities[: self.feature_limit]
        values = self._augmented_values()
        offsets, sums = core.state_histogram(
            states, cards, values, n_threads=self.n_jobs_
        )
        self.main = []
        for j in range(self.feature_limit):
            start, stop = int(offsets[j]), int(offsets[j + 1])
            self.main.append(self._split_aggregates(sums[start:stop]))

        if not self.pair_keys:
            self.pairs = {}
            return
        pair_array = np.asarray(self.pair_keys, dtype=np.int32).reshape(-1, 2)
        pair_offsets, pair_sums = core.pair_histogram(
            states, cards, pair_array, values, n_threads=self.n_jobs_
        )
        self.pairs = {}
        for index, key in enumerate(self.pair_keys):
            start, stop = int(pair_offsets[index]), int(pair_offsets[index + 1])
            self.pairs[key] = self._split_aggregates(pair_sums[start:stop])

    def _build_python(self) -> None:
        self.main = []
        for j in range(self.feature_limit):
            code = self.states.array[:, j]
            card = int(self.states.cardinalities[j])
            mass = np.bincount(code, minlength=card).astype(np.float64)
            sums = np.column_stack(
                [
                    np.bincount(code, weights=self.values[:, r], minlength=card)
                    for r in range(self.values.shape[1])
                ]
            )
            self.main.append(ValueHistogramStats(mass=mass, sums=sums))

        self.pairs = {}
        pair_code = np.empty(len(self.states.array), dtype=np.int64)
        for j, k in self.pair_keys:
            card_k = int(self.states.cardinalities[k])
            np.multiply(
                self.states.array[:, j], card_k, out=pair_code, dtype=np.int64
            )
            np.add(pair_code, self.states.array[:, k], out=pair_code)
            card = int(self.states.cardinalities[j]) * card_k
            mass = np.bincount(pair_code, minlength=card).astype(np.float64)
            sums = np.column_stack(
                [
                    np.bincount(
                        pair_code, weights=self.values[:, r], minlength=card
                    )
                    for r in range(self.values.shape[1])
                ]
            )
            self.pairs[(j, k)] = ValueHistogramStats(mass=mass, sums=sums)

    @property
    def nbytes(self) -> int:
        return int(
            self.states.nbytes
            + self.values.nbytes
            + sum(stats.mass.nbytes + stats.sums.nbytes for stats in self.main)
            + sum(
                stats.mass.nbytes + stats.sums.nbytes
                for stats in self.pairs.values()
            )
        )


class NewtonHistogramCache:
    """Reusable G/H histograms for finite-state main and pair candidates.

    ``backend="auto"`` tries the optional cached C++ histogram core for uint8
    state banks and falls back to the historical NumPy implementation when the
    compiler, source, platform, or ABI is unavailable. ``backend="python"``
    forces the reference implementation; ``backend="native"`` requires the
    native path and raises instead of falling back.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        states: np.ndarray,
        y: np.ndarray,
        prediction: np.ndarray | None = None,
        *,
        feature_names: Iterable[str] | None = None,
        feature_limit: int | None = None,
        backend: str = "auto",
        n_jobs: int | None = None,
    ):
        self.states = PackedStateMatrix(states)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(target) != len(self.states.array):
            raise ValueError("states and y must have the same number of rows")
        if prediction is None:
            mean = float(np.clip(target.mean(), 1e-6, 1.0 - 1e-6))
            prediction = np.full(len(target), mean, dtype=np.float64)
        prediction = np.clip(
            np.asarray(prediction, dtype=np.float64).reshape(-1),
            1e-6,
            1.0 - 1e-6,
        )
        if len(prediction) != len(target):
            raise ValueError("prediction and y must have the same length")

        self.g = target - prediction
        self.h = prediction * (1.0 - prediction)
        self.feature_limit = min(
            self.states.array.shape[1],
            self.states.array.shape[1]
            if feature_limit is None
            else int(feature_limit),
        )
        if self.feature_limit < 1:
            raise ValueError("feature_limit must be positive")
        names = tuple(feature_names or ())
        if names and len(names) < self.feature_limit:
            raise ValueError("feature_names is shorter than feature_limit")
        self.feature_names = (
            names[: self.feature_limit]
            if names
            else tuple(f"state_{j}" for j in range(self.feature_limit))
        )

        backend = str(backend).lower()
        if backend not in {"auto", "python", "native"}:
            raise ValueError("backend must be 'auto', 'python', or 'native'")
        self.backend_requested_ = backend
        self.backend_ = "python"
        self.native_error_ = None
        self.n_jobs_ = self._resolve_n_jobs(n_jobs)

        if backend != "python" and self.states.array.dtype == np.uint8:
            try:
                self._build_native()
                self.backend_ = "native"
                return
            except Exception as exc:
                self.native_error_ = repr(exc)
                if backend == "native":
                    raise
        elif backend == "native":
            raise NotImplementedError(
                "native training-cache ABI v1 supports uint8 state banks only"
            )

        self._build_python()

    def _resolve_n_jobs(self, n_jobs: int | None) -> int:
        if n_jobs is None:
            return min(4, max(1, os.cpu_count() or 1), self.feature_limit)
        value = int(n_jobs)
        if value == 0:
            raise ValueError("n_jobs must not be zero")
        if value < 0:
            value = max(1, (os.cpu_count() or 1) + 1 + value)
        return min(max(1, value), max(1, self.feature_limit))

    def _build_python(self) -> None:
        self.main = []
        for j in range(self.feature_limit):
            card = int(self.states.cardinalities[j])
            code = self.states.array[:, j]
            self.main.append(
                HistogramStats(
                    np.bincount(code, weights=self.g, minlength=card),
                    np.bincount(code, weights=self.h, minlength=card),
                )
            )

        self.pairs = {}
        pair_code = np.empty(len(self.states.array), dtype=np.int64)
        for j in range(self.feature_limit):
            left = self.states.array[:, j]
            for k in range(j + 1, self.feature_limit):
                card_k = int(self.states.cardinalities[k])
                np.multiply(left, card_k, out=pair_code, dtype=np.int64)
                np.add(pair_code, self.states.array[:, k], out=pair_code)
                card = int(self.states.cardinalities[j]) * card_k
                self.pairs[(j, k)] = HistogramStats(
                    np.bincount(pair_code, weights=self.g, minlength=card),
                    np.bincount(pair_code, weights=self.h, minlength=card),
                )

    def _build_native(self) -> None:
        from ._internal.cerm_training_core_runtime import load_native_training_core

        core = load_native_training_core()
        states = self.states.array[:, : self.feature_limit]
        cards = self.states.cardinalities[: self.feature_limit]
        values = np.ascontiguousarray(np.column_stack([self.g, self.h]))
        offsets, sums = core.state_histogram(
            states,
            cards,
            values,
            n_threads=self.n_jobs_,
        )
        self.main = []
        for j in range(self.feature_limit):
            start, stop = int(offsets[j]), int(offsets[j + 1])
            self.main.append(
                HistogramStats(sums[start:stop, 0].copy(), sums[start:stop, 1].copy())
            )

        pair_keys = [
            (j, k)
            for j in range(self.feature_limit)
            for k in range(j + 1, self.feature_limit)
        ]
        pair_array = np.asarray(pair_keys, dtype=np.int32).reshape(-1, 2)
        pair_offsets, pair_sums = core.pair_histogram(
            states,
            cards,
            pair_array,
            values,
            n_threads=self.n_jobs_,
        )
        self.pairs = {}
        for index, key in enumerate(pair_keys):
            start, stop = int(pair_offsets[index]), int(pair_offsets[index + 1])
            self.pairs[key] = HistogramStats(
                pair_sums[start:stop, 0].copy(),
                pair_sums[start:stop, 1].copy(),
            )

    @staticmethod
    def gain(stats: HistogramStats, l2: float) -> float:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        G, H = stats.G, stats.H
        return float(
            0.5
            * max(
                0.0,
                np.sum(G * G / (H + l2))
                - G.sum() ** 2 / (H.sum() + l2),
            )
        )

    def rank_pairs(self, max_pairs: int = 30, l2: float = 5.0):
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        main = np.asarray([self.gain(stats, l2) for stats in self.main])
        ranked = []
        for (j, k), stats in self.pairs.items():
            gain = self.gain(stats, l2)
            incremental = gain - max(main[j], main[k])
            ranked.append((incremental, gain, j, k))
        ranked.sort(reverse=True)
        return [
            {
                "left": j,
                "right": k,
                "left_name": self.feature_names[j],
                "right_name": self.feature_names[k],
                "incremental_gain": float(incremental),
                "joint_gain": float(gain),
            }
            for incremental, gain, j, k in ranked[:max_pairs]
        ]

    def sweep_pair_rankings(
        self,
        l2_values: Iterable[float],
        *,
        max_pairs: int = 30,
    ) -> dict[float, list[dict]]:
        return {
            float(l2): self.rank_pairs(max_pairs=max_pairs, l2=float(l2))
            for l2 in l2_values
        }

    @property
    def nbytes(self) -> int:
        return int(
            self.states.nbytes
            + self.g.nbytes
            + self.h.nbytes
            + sum(s.G.nbytes + s.H.nbytes for s in self.main)
            + sum(s.G.nbytes + s.H.nbytes for s in self.pairs.values())
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pair_keys = np.asarray(list(self.pairs), dtype=np.int32).reshape(-1, 2)
        arrays = {
            "format_version": np.asarray([self.FORMAT_VERSION], dtype=np.int16),
            "states": self.states.array,
            "cardinalities": self.states.cardinalities,
            "g": self.g,
            "h": self.h,
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "feature_limit": np.asarray([self.feature_limit], dtype=np.int32),
            "pair_keys": pair_keys,
        }
        for index, stats in enumerate(self.main):
            arrays[f"main_G_{index}"] = stats.G
            arrays[f"main_H_{index}"] = stats.H
        for index, key in enumerate(self.pairs):
            arrays[f"pair_G_{index}"] = self.pairs[key].G
            arrays[f"pair_H_{index}"] = self.pairs[key].H
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NewtonHistogramCache":
        with np.load(path, allow_pickle=False) as arrays:
            version = int(arrays["format_version"][0])
            if version != cls.FORMAT_VERSION:
                raise ValueError(f"unsupported histogram cache version: {version}")
            obj = cls.__new__(cls)
            obj.states = PackedStateMatrix(arrays["states"])
            obj.states.cardinalities = np.asarray(
                arrays["cardinalities"], dtype=np.int64
            )
            obj.g = np.asarray(arrays["g"], dtype=np.float64)
            obj.h = np.asarray(arrays["h"], dtype=np.float64)
            obj.feature_names = tuple(arrays["feature_names"].tolist())
            obj.feature_limit = int(arrays["feature_limit"][0])
            obj.backend_requested_ = "loaded"
            obj.backend_ = "loaded"
            obj.native_error_ = None
            obj.n_jobs_ = 1
            obj.main = []
            for index in range(obj.feature_limit):
                obj.main.append(
                    HistogramStats(
                        np.asarray(arrays[f"main_G_{index}"], dtype=np.float64),
                        np.asarray(arrays[f"main_H_{index}"], dtype=np.float64),
                    )
                )
            obj.pairs = {}
            pair_keys = np.asarray(arrays["pair_keys"], dtype=np.int32).reshape(-1, 2)
            for index, key in enumerate(pair_keys):
                obj.pairs[(int(key[0]), int(key[1]))] = HistogramStats(
                    np.asarray(arrays[f"pair_G_{index}"], dtype=np.float64),
                    np.asarray(arrays[f"pair_H_{index}"], dtype=np.float64),
                )
            return obj
