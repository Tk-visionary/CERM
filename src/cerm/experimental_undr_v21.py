"""Opt-in UNDR-v2.1 refinement for a fitted binary CERM model.

This module is deliberately additive: importing it does not alter CERM defaults.
The strict API expects the wrapped CERM estimator to have been fitted on search/
fit rows only. Candidate search, conditional-deficit selection and quotient fit
use only those rows. A separately supplied fresh holdout is inspected exactly
once, after the composite operator has been frozen.

Research scope: numeric ndarray binary classification only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb, log
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

from . import _experimental_undr_geometry as geom


@dataclass(frozen=True)
class UNDRConfig:
    alpha2: float = 0.001
    maximum_features: int = 12
    maximum_stage2_triads: int = 16
    budget: int = 2
    l2: float = 20.0
    learning_rate: float = 0.5
    holdout_delta: float = 0.05
    family_volume_rho: float = 0.0
    second_cost_eta: float = 0.0
    byte_unit: int = 64

    def validate(self) -> None:
        if not (0.0 < float(self.alpha2) < 1.0):
            raise ValueError("alpha2 must lie in (0, 1)")
        if int(self.maximum_features) < 1:
            raise ValueError("maximum_features must be positive")
        if int(self.maximum_stage2_triads) < 1:
            raise ValueError("maximum_stage2_triads must be positive")
        if int(self.budget) < 0:
            raise ValueError("budget must be non-negative")
        if float(self.l2) < 0 or not np.isfinite(float(self.l2)):
            raise ValueError("l2 must be finite and non-negative")
        if not (0.0 < float(self.learning_rate) <= 1.0):
            raise ValueError("learning_rate must lie in (0, 1]")
        if not (0.0 < float(self.holdout_delta) < 1.0):
            raise ValueError("holdout_delta must lie in (0, 1)")
        if float(self.family_volume_rho) < 0 or float(self.second_cost_eta) < 0:
            raise ValueError("routing penalties must be non-negative")
        if int(self.byte_unit) < 1:
            raise ValueError("byte_unit must be positive")


@dataclass(frozen=True)
class DeficitCandidate:
    family: str
    name: str
    p_ref: float
    deficit: float
    df: int
    feature: Optional[int] = None          # local selected-feature index
    raw_feature: Optional[int] = None      # adapted/raw index in CERM encoder
    source_q: Optional[int] = None
    target_q: Optional[int] = None
    triad: Optional[Tuple[int, int, int]] = None       # local indices
    raw_triad: Optional[Tuple[int, int, int]] = None   # encoder indices
    estimated_bytes: int = 0


@dataclass
class ProjectedTrainingGraph:
    selected_raw_features: np.ndarray
    states_by_q: Dict[int, np.ndarray]
    y: np.ndarray
    score: np.ndarray
    g: np.ndarray
    h: np.ndarray
    transform_calls: Dict[int, str] = field(default_factory=dict)
    state_encoder: object | None = None
    state_encoder_projected: bool = False
    # Training-only sufficient statistics retained across search -> selection.
    # This cache is never exported into FrozenUNDRProgram.
    basis_stat_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        repr=False,
    )
    # Training-only final selected-span state.  This is populated only when
    # conditional selection maintained an exact incremental Gram throughout.
    selection_gram_names: Tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    selection_gram_cache: Optional[Tuple[np.ndarray, np.ndarray]] = field(
        default=None,
        repr=False,
    )


@dataclass
class Selection:
    candidates: List[DeficitCandidate]
    bases: List[geom.StateQuotientBasis]
    trace: List[dict]
    workspace_backend: str = "none"
    workspace_packed_bytes: int = 0


@dataclass
class UNDRFitAudit:
    selected_names: Tuple[str, ...]
    selected_families: Tuple[str, ...]
    stage2_resolution: int
    stage2_triad: int
    triad_exact_solves: int
    first_family_trials: Dict[str, int]
    holdout_gain: float
    holdout_radius: float
    holdout_lcb: float
    accepted: bool
    operator_bytes: int
    search_metadata: dict
    selection_trace: List[dict]


def _numeric_matrix(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("UNDR-v2.1 currently requires a finite numeric 2-D ndarray")
    return X


def _require_binary_fitted(estimator):
    if not hasattr(estimator, "model_") or not hasattr(estimator.model_, "base_"):
        raise TypeError("estimator must be a fitted binary CERMClassifier exposing model_.base_")
    base = estimator.model_.base_
    if not hasattr(base, "encoder_") or not hasattr(base, "feature_idx_"):
        raise TypeError("CERM base model lacks nested encoder/feature_idx_ contract")
    return estimator.model_, base


def _model_score(model, X: np.ndarray) -> np.ndarray:
    score = np.asarray(model.decision_function(X), dtype=np.float64)
    if score.ndim != 1 or len(score) != len(X):
        raise ValueError("binary CERM decision_function must return one score per row")
    return score


def _available_levels(encoder) -> Tuple[int, ...]:
    levels = tuple(sorted(int(q) for q in getattr(encoder, "levels", (4, 8, 16))))
    return levels


def _compact_state_level(values, q: int) -> np.ndarray:
    """Validate one UNDR quotient bank and store it in exact uint8 form."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("state transform must return a 2-D matrix")
    if len(array) and (
        np.min(array, initial=0) < 0
        or np.max(array, initial=0) >= int(q)
    ):
        raise ValueError("state transform returned a code outside the requested quotient")
    return np.ascontiguousarray(array, dtype=np.uint8)

def _transform_level_columns(encoder, X, q: int, columns: np.ndarray):
    # CERM 0.12.0a10+ exact projected TrainingGraph path.
    if hasattr(encoder, "transform_level_columns"):
        return _compact_state_level(encoder.transform_level_columns(X, int(q), columns), q), "transform_level_columns"
    # Exact older fallback, still projected by columns when available.
    if hasattr(encoder, "transform_columns"):
        result = encoder.transform_columns(X, columns)
        return _compact_state_level(result[int(q)], q), "transform_columns"
    result = encoder.transform(X)
    return _compact_state_level(result[int(q)][:, columns], q), "transform+slice"


def _shadow_multiresolution_encoder(base, X, y, raw):
    """Build a selected-feature 4/8/16 encoder when the fitted base stopped earlier.

    The shadow encoder is fit only on the same training rows already used by the
    fitted base.  Existing quotient levels must reproduce the base states exactly;
    otherwise the extension is rejected and the historical encoder is retained.
    """
    encoder = base.encoder_
    existing = _available_levels(encoder)
    if all(q in existing for q in (4, 8, 16)):
        return encoder, False, "base_encoder"
    try:
        from ._internal import cerm_hierarchical_residual_legacy as legacy
        from ._internal.cerm_hierarchical_residual import (
            NestedQuantileEncoder, NewtonNestedEncoder,
        )
    except ImportError:
        return encoder, False, "shadow_unavailable"
    # Unweighted production fits intentionally retain the historical encoder
    # classes, while weighted fits use the facade subclasses. Both implement
    # the same nested-state contract and are valid shadow-extension sources.
    if not isinstance(encoder, legacy.NestedQuantileEncoder):
        return encoder, False, "shadow_unsupported_encoder"

    raw = np.asarray(raw, dtype=np.int64)
    projected = np.asarray(X[:, raw], dtype=np.float64)
    fitted_kinds = getattr(encoder, "feature_kinds_", None)
    fitted_cards = getattr(encoder, "feature_cardinalities_", None)
    kinds = None if fitted_kinds is None else [fitted_kinds[int(j)] for j in raw]
    cards = None if fitted_cards is None else [fitted_cards[int(j)] for j in raw]
    if isinstance(encoder, legacy.NewtonNestedEncoder):
        shadow = NewtonNestedEncoder(
            prebins=int(getattr(encoder, "prebins", 64)),
            gain_l2=float(getattr(encoder, "gain_l2", 5.0)),
            min_hessian=float(getattr(encoder, "min_hessian", 1.0)),
            max_bins=16, levels=(4, 8, 16),
            feature_kinds=kinds, feature_cardinalities=cards,
        )
    else:
        shadow = NestedQuantileEncoder(
            max_bins=16, levels=(4, 8, 16),
            feature_kinds=kinds, feature_cardinalities=cards,
        )
    shadow.fit(projected, y)
    local = np.arange(len(raw), dtype=np.int64)
    for q in existing:
        if q not in (4, 8, 16):
            continue
        base_state, _ = _transform_level_columns(encoder, X, q, raw)
        shadow_state, _ = _transform_level_columns(shadow, projected, q, local)
        if not np.array_equal(base_state, shadow_state):
            return encoder, False, f"shadow_mismatch_q{q}"
    return shadow, True, "shadow_selected_features"


def _state_transform(encoder, projected: bool, X, q: int, raw: np.ndarray):
    if projected:
        local = np.arange(len(raw), dtype=np.int64)
        return _transform_level_columns(encoder, np.asarray(X)[:, raw], q, local)
    return _transform_level_columns(encoder, X, q, raw)


def _state_transforms(
    encoder,
    projected: bool,
    X,
    levels: Sequence[int],
    raw: np.ndarray,
):
    """Transform all requested selected-feature levels in one fine-state pass.

    Fitted CERM nested encoders expose transform_columns, which computes the
    selected fine states once and maps them to every fitted quotient level.
    Older/custom encoders without that fused API retain the historical
    per-level path.
    """
    requested = tuple(int(q) for q in levels)
    if not requested:
        return {}, {}

    if len(requested) > 1 and hasattr(encoder, "transform_columns"):
        if projected:
            matrix = np.asarray(X)[:, raw]
            columns = np.arange(len(raw), dtype=np.int64)
        else:
            matrix = X
            columns = raw
        result = encoder.transform_columns(matrix, columns)
        states = {}
        paths = {}
        expected_shape = (len(np.asarray(X)), len(raw))
        for q in requested:
            if int(q) not in result:
                raise ValueError(
                    f"fused state transform did not return requested level {q}"
                )
            array = _compact_state_level(result[int(q)], q)
            if array.shape != expected_shape:
                raise ValueError(
                    "fused state transform returned an unexpected shape"
                )
            states[int(q)] = array
            paths[int(q)] = "transform_columns_fused"
        return states, paths

    states = {}
    paths = {}
    for q in requested:
        states[int(q)], paths[int(q)] = _state_transform(
            encoder, projected, X, int(q), raw
        )
    return states, paths

def extract_projected_training_graph(estimator, X, y, *, maximum_features: int = 12) -> ProjectedTrainingGraph:
    X = _numeric_matrix(X)
    y = np.asarray(y, dtype=np.int32)
    if y.ndim != 1 or len(y) != len(X) or set(np.unique(y)).difference({0, 1}):
        raise ValueError("y must be a binary 0/1 vector aligned with X")
    model, base = _require_binary_fitted(estimator)
    raw = np.asarray(base.feature_idx_, dtype=np.int64)
    raw = raw[: min(int(maximum_features), len(raw))]
    if raw.size == 0:
        raise ValueError("fitted CERM base selected no features")
    state_encoder, projected, encoder_path = _shadow_multiresolution_encoder(
        base, X, y, raw
    )
    levels = [q for q in (4, 8, 16) if q in _available_levels(state_encoder)]
    states, calls = _state_transforms(
        state_encoder, projected, X, levels, raw
    )
    paths: Dict[int, str] = {
        int(q): (
            calls[int(q)]
            if encoder_path == "base_encoder"
            else f"{encoder_path}:{calls[int(q)]}"
        )
        for q in levels
    }
    score = _model_score(model, X)
    g, h = geom.logistic_gh(y, score)
    return ProjectedTrainingGraph(
        raw, states, y, score, g, h, paths, state_encoder, projected
    )


def _resolution_increment(
    states_fine,
    g,
    h,
    local_feature: int,
    source_q: int,
    target_q: int,
    raw_feature: int,
) -> Tuple[DeficitCandidate, np.ndarray, np.ndarray]:
    # ``extract_projected_training_graph`` already owns the gradient and Hessian.
    # Reusing them keeps every candidate on the exact same floating-point inputs
    # and avoids two full-length allocations per resolution scan.
    z = np.asarray(states_fine[:, local_feature])
    ratio = int(target_q) // int(source_q)
    # CERM state banks are int16.  bincount accepts that representation, so only
    # the genuinely new coarse-state temporary is materialized here.
    parent = np.floor_divide(z, ratio)
    Gf = np.bincount(z, weights=g, minlength=target_q)
    Hf = np.bincount(z, weights=h, minlength=target_q)
    Nf = np.bincount(z, minlength=target_q)
    Gc = np.bincount(parent, weights=g, minlength=source_q)
    Hc = np.bincount(parent, weights=h, minlength=source_q)
    Nc = np.bincount(parent, minlength=source_q)
    vf, vc = Hf > 0, Hc > 0
    fine_gain = 0.5 * float(np.sum(Gf[vf] ** 2 / Hf[vf])) if np.any(vf) else 0.0
    coarse_gain = 0.5 * float(np.sum(Gc[vc] ** 2 / Hc[vc])) if np.any(vc) else 0.0
    deficit = max(0.0, fine_gain - coarse_gain)
    df = max(0, int(np.count_nonzero(Nf) - np.count_nonzero(Nc)))
    p_ref = float(chi2.sf(2.0 * deficit, df)) if df else 1.0
    candidate = DeficitCandidate(
        family="resolution",
        name=f"q{source_q}->q{target_q}:x{raw_feature}",
        p_ref=p_ref,
        deficit=deficit,
        df=df,
        feature=int(local_feature),
        raw_feature=int(raw_feature),
        source_q=int(source_q),
        target_q=int(target_q),
        estimated_bytes=int(target_q) * 8,
    )
    return candidate, Gf, Hf


def _candidate_diagnostic(candidate: DeficitCandidate, *, passed: bool) -> dict:
    """Return a prediction-neutral, training-only audit record."""
    return {
        "family": str(candidate.family),
        "name": str(candidate.name),
        "p_ref": float(candidate.p_ref),
        "deficit": float(candidate.deficit),
        "df": int(candidate.df),
        "source_q": (
            None if candidate.source_q is None else int(candidate.source_q)
        ),
        "target_q": (
            None if candidate.target_q is None else int(candidate.target_q)
        ),
        "raw_feature": (
            None if candidate.raw_feature is None else int(candidate.raw_feature)
        ),
        "raw_triad": (
            None
            if candidate.raw_triad is None
            else tuple(int(value) for value in candidate.raw_triad)
        ),
        "passed_stage2": bool(passed),
    }




def _nested_state_levels_equal(
    coarse: np.ndarray,
    fine: np.ndarray,
    ratio: int,
    *,
    chunk_rows: int = 32768,
) -> bool:
    """Check exact parent nesting with bounded temporary memory."""
    coarse = np.asarray(coarse)
    fine = np.asarray(fine)
    if coarse.shape != fine.shape or coarse.ndim != 2:
        return False
    if int(ratio) <= 0 or int(chunk_rows) <= 0:
        return False
    for start in range(0, len(fine), int(chunk_rows)):
        stop = min(len(fine), start + int(chunk_rows))
        if not np.array_equal(
            np.floor_divide(fine[start:stop], int(ratio)),
            coarse[start:stop],
        ):
            return False
    return True


def _native_resolution_stat_banks(graph: ProjectedTrainingGraph):
    """Return per-level G/H banks by scanning only the finest nested level.

    This path is enabled only when the observed q4/q8/q16 state banks satisfy
    exact parent nesting.  The finest available level is accumulated in native
    code once; coarser banks are then formed by exact additive bin aggregation.
    Floating-point addition order differs slightly from independent row scans,
    so selection parity is covered by near-tie stress tests.
    """
    transitions = [
        (q0, q1)
        for q0, q1 in ((4, 8), (8, 16))
        if q1 in graph.states_by_q
    ]
    if not transitions:
        return None

    for q0, q1 in transitions:
        if q0 not in graph.states_by_q:
            return None
        coarse = np.asarray(graph.states_by_q[q0])
        fine = np.asarray(graph.states_by_q[q1])
        ratio = int(q1) // int(q0)
        if ratio <= 0 or int(q1) % int(q0):
            return None
        if not _nested_state_levels_equal(coarse, fine, ratio):
            return None

    try:
        from ._internal.cerm_training_core_runtime import (
            _prebuilt_library_path,
            load_native_training_core,
        )
        already_loaded = load_native_training_core.cache_info().currsize > 0
        prebuilt = _prebuilt_library_path()
        if not already_loaded and not (prebuilt is not None and prebuilt.is_file()):
            return None
        core = load_native_training_core()
    except Exception:
        return None

    width = int(len(graph.selected_raw_features))
    n_threads = min(4, max(1, os.cpu_count() or 1), max(1, width))
    values = np.column_stack(
        [
            np.asarray(graph.g, dtype=np.float64),
            np.asarray(graph.h, dtype=np.float64),
        ]
    )
    levels = sorted({q for transition in transitions for q in transition})
    q_max = int(levels[-1])
    banks = {}
    total_bytes = 0
    try:
        states_max = np.asarray(graph.states_by_q[q_max])
        cards_max = np.full(width, q_max, dtype=np.int32)
        offsets, sums = core.state_histogram(
            states_max,
            cards_max,
            values,
            n_threads=n_threads,
        )
        expected = width * q_max
        if int(offsets[-1]) != expected:
            return None
        bank_max = np.asarray(sums, dtype=np.float64).reshape(width, q_max, 2)
        banks[q_max] = bank_max
        total_bytes += int(bank_max.nbytes)

        for q in reversed(levels[:-1]):
            q = int(q)
            if q_max % q:
                return None
            ratio = q_max // q
            bank_q = bank_max.reshape(width, q, ratio, 2).sum(axis=2)
            banks[q] = bank_q
            total_bytes += int(bank_q.nbytes)
    except Exception:
        return None

    return banks, int(n_threads), int(total_bytes)


def _resolution_increment_from_stats(
    Gf,
    Hf,
    Gc,
    Hc,
    local_feature: int,
    source_q: int,
    target_q: int,
    raw_feature: int,
) -> DeficitCandidate:
    Gf = np.asarray(Gf, dtype=np.float64)
    Hf = np.asarray(Hf, dtype=np.float64)
    Gc = np.asarray(Gc, dtype=np.float64)
    Hc = np.asarray(Hc, dtype=np.float64)
    if Gf.shape != (int(target_q),) or Hf.shape != Gf.shape:
        raise ValueError("fine resolution statistics have the wrong shape")
    if Gc.shape != (int(source_q),) or Hc.shape != Gc.shape:
        raise ValueError("coarse resolution statistics have the wrong shape")

    vf = Hf > 0.0
    vc = Hc > 0.0
    fine_gain = (
        0.5 * float(np.sum(Gf[vf] ** 2 / Hf[vf]))
        if np.any(vf)
        else 0.0
    )
    coarse_gain = (
        0.5 * float(np.sum(Gc[vc] ** 2 / Hc[vc]))
        if np.any(vc)
        else 0.0
    )
    deficit = max(0.0, fine_gain - coarse_gain)
    df = max(0, int(np.count_nonzero(vf) - np.count_nonzero(vc)))
    p_ref = float(chi2.sf(2.0 * deficit, df)) if df else 1.0
    return DeficitCandidate(
        family="resolution",
        name=f"q{source_q}->q{target_q}:x{raw_feature}",
        p_ref=p_ref,
        deficit=deficit,
        df=df,
        feature=int(local_feature),
        raw_feature=int(raw_feature),
        source_q=int(source_q),
        target_q=int(target_q),
        estimated_bytes=int(target_q) * 8,
    )


def resolution_candidates(
    graph: ProjectedTrainingGraph,
    alpha2: float,
    diagnostics: Optional[List[dict]] = None,
    metadata: Optional[dict] = None,
) -> List[DeficitCandidate]:
    out: List[DeficitCandidate] = []
    native_result = _native_resolution_stat_banks(graph)
    if native_result is None:
        native_banks = None
        native_threads = 0
        native_bytes = 0
        backend = "scalar"
    else:
        native_banks, native_threads, native_bytes = native_result
        backend = "native_state_bank"

    for local, raw in enumerate(graph.selected_raw_features):
        for q0, q1 in ((4, 8), (8, 16)):
            if q1 not in graph.states_by_q:
                continue
            if (
                native_banks is not None
                and q0 in native_banks
                and q1 in native_banks
            ):
                candidate = _resolution_increment_from_stats(
                    native_banks[q1][local, :, 0],
                    native_banks[q1][local, :, 1],
                    native_banks[q0][local, :, 0],
                    native_banks[q0][local, :, 1],
                    local,
                    q0,
                    q1,
                    int(raw),
                )
                Gf, Hf = None, None
            else:
                candidate, Gf, Hf = _resolution_increment(
                    graph.states_by_q[q1],
                    graph.g,
                    graph.h,
                    local,
                    q0,
                    q1,
                    int(raw),
                )
            passed = bool(
                candidate.df > 0 and candidate.p_ref < float(alpha2)
            )
            if diagnostics is not None:
                diagnostics.append(
                    _candidate_diagnostic(candidate, passed=passed)
                )
            if passed:
                if (
                    native_banks is not None
                    and q1 in native_banks
                ):
                    graph.basis_stat_cache[candidate.name] = (
                        np.asarray(
                            native_banks[q1][local, :, 0],
                            dtype=np.float64,
                        ).copy(),
                        np.asarray(
                            native_banks[q1][local, :, 1],
                            dtype=np.float64,
                        ).copy(),
                    )
                else:
                    graph.basis_stat_cache[candidate.name] = (
                        np.asarray(Gf, dtype=np.float64).copy(),
                        np.asarray(Hf, dtype=np.float64).copy(),
                    )
                out.append(candidate)

    if metadata is not None:
        metadata.update(
            {
                "resolution_histogram_backend": backend,
                "resolution_native_threads": int(native_threads),
                "resolution_native_bank_bytes": int(native_bytes),
                "resolution_native_levels": (
                    []
                    if native_banks is None
                    else sorted(int(q) for q in native_banks)
                ),
            }
        )
    out.sort(key=lambda candidate: (
        candidate.p_ref,
        candidate.target_q,
        candidate.raw_feature,
    ))
    return out


def _pair_gain(G, H, axis: int) -> float:
    g = np.sum(G, axis=axis); h = np.sum(H, axis=axis); v = h > 0
    return 0.5 * float(np.sum(g[v] * g[v] / h[v])) if np.any(v) else 0.0


def _pair_sweep_bound(G, H, support, sweeps: int = 2) -> float:
    valid = np.asarray(support, dtype=bool) & (H > 0)
    if not np.any(valid): return 0.0
    sh = np.sqrt(np.where(valid, H, 0.0))
    r = np.zeros_like(G, dtype=np.float64); r[valid] = -G[valid] / sh[valid]
    gains = (_pair_gain(G, H, 2), _pair_gain(G, H, 1), _pair_gain(G, H, 0))
    order = tuple(sorted((2, 1, 0), key=lambda ax: (-gains[{2: 0, 1: 1, 0: 2}[ax]], ax)))
    for _ in range(int(sweeps)):
        for ax in order:
            num = np.sum(sh * r, axis=ax); den = np.sum(np.where(valid, H, 0.0), axis=ax)
            delta = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            if ax == 2: r -= sh * delta[:, :, None]
            elif ax == 1: r -= sh * delta[:, None, :]
            else: r -= sh * delta[None, :, :]
    return 0.5 * float(np.dot(r[valid], r[valid]))



def _pair_sweep_bound_batch(G, H, sweeps: int = 2) -> np.ndarray:
    """Batch the historical sweep projection and preserve scalar final scores."""
    G = np.asarray(G, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if G.shape != H.shape or G.ndim != 4 or G.shape[1:] != (4, 4, 4):
        raise ValueError("batch sweep expects matching (n, 4, 4, 4) G/H arrays")
    if len(G) == 0:
        return np.empty(0, dtype=np.float64)

    valid = H > 0.0
    sh = np.sqrt(np.where(valid, H, 0.0))
    r = np.divide(-G, sh, out=np.zeros_like(G), where=valid)

    pair_gains = {}
    for scalar_axis, batch_axis in ((2, 3), (1, 2), (0, 1)):
        pair_G = np.sum(G, axis=batch_axis)
        pair_H = np.sum(H, axis=batch_axis)
        pair_valid = pair_H > 0.0
        terms = np.divide(
            pair_G * pair_G,
            pair_H,
            out=np.zeros_like(pair_G),
            where=pair_valid,
        )
        pair_gains[scalar_axis] = 0.5 * np.sum(
            terms,
            axis=tuple(range(1, terms.ndim)),
        )

    orders = np.asarray(
        [
            tuple(
                sorted(
                    (2, 1, 0),
                    key=lambda axis: (-pair_gains[axis][index], axis),
                )
            )
            for index in range(len(G))
        ],
        dtype=np.int8,
    )

    for order in sorted(set(map(tuple, orders.tolist()))):
        ids = np.flatnonzero(
            np.all(orders == np.asarray(order, dtype=np.int8), axis=1)
        )
        if not len(ids):
            continue
        group_r = r[ids]
        group_sh = sh[ids]
        group_valid = valid[ids]
        group_H = H[ids]
        for _ in range(int(sweeps)):
            for scalar_axis in order:
                batch_axis = int(scalar_axis) + 1
                numerator = np.sum(group_sh * group_r, axis=batch_axis)
                denominator = np.sum(
                    np.where(group_valid, group_H, 0.0),
                    axis=batch_axis,
                )
                delta = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 0.0,
                )
                if scalar_axis == 2:
                    group_r -= group_sh * delta[:, :, :, None]
                elif scalar_axis == 1:
                    group_r -= group_sh * delta[:, :, None, :]
                else:
                    group_r -= group_sh * delta[:, None, :, :]
        r[ids] = group_r

    output = np.empty(len(G), dtype=np.float64)
    for index in range(len(G)):
        mask = valid[index]
        output[index] = (
            0.5 * float(np.dot(r[index][mask], r[index][mask]))
            if np.any(mask)
            else 0.0
        )
    return output


_TRIAD_PAIR_CACHE_BUDGET_BYTES = 8 * 1024 * 1024
_TRIAD_NATIVE_BANK_BUDGET_BYTES = 8 * 1024 * 1024


def _build_triad_pair_prefix_cache(
    S4: np.ndarray,
    *,
    budget_bytes: int = _TRIAD_PAIR_CACHE_BUDGET_BYTES,
):
    """Cache reusable 16-state pair prefixes for exact triad refinement.

    A sorted triad (a, b, c) uses code 16*a + 4*b + c. The 16*a + 4*b
    prefix is reusable for every c > b. Prefixes are uint8 and admitted by
    reuse count under a strict byte budget.
    """
    states = np.asarray(S4)
    if states.ndim != 2:
        raise ValueError("S4 must be a 2-D state matrix")
    if len(states) == 0 or states.shape[1] < 3 or int(budget_bytes) <= 0:
        return np.empty((0, len(states)), dtype=np.uint8), {}
    if np.any((states < 0) | (states > 3)):
        raise ValueError("S4 must contain only 4-state labels in [0, 3]")
    states = np.ascontiguousarray(states, dtype=np.uint8)

    bytes_per_pair = int(len(states))
    pair_count = sum(
        1
        for a in range(states.shape[1])
        for b in range(a + 1, states.shape[1] - 1)
    )
    capacity = min(
        pair_count,
        int(budget_bytes) // max(bytes_per_pair, 1),
    )
    if capacity <= 0:
        return np.empty((0, len(states)), dtype=np.uint8), {}

    candidates = [
        (-(states.shape[1] - b - 1), a, b)
        for a in range(states.shape[1])
        for b in range(a + 1, states.shape[1] - 1)
    ]
    candidates.sort()
    selected = [(a, b) for _, a, b in candidates[:capacity]]

    bank = np.empty((len(selected), len(states)), dtype=np.uint8)
    temporary = np.empty(len(states), dtype=np.uint8)
    lookup = {}
    for index, (a, b) in enumerate(selected):
        np.left_shift(states[:, a], 4, out=bank[index])
        np.left_shift(states[:, b], 2, out=temporary)
        np.add(bank[index], temporary, out=bank[index], casting="unsafe")
        lookup[(int(a), int(b))] = int(index)
    return bank, lookup


def _triad_cube(
    S4,
    g,
    h,
    triad,
    code_buffer=None,
    pair_prefix_cache=None,
):
    a, b, c = triad
    if code_buffer is None:
        code = np.empty(S4.shape[0], dtype=np.int64)
    else:
        code = np.asarray(code_buffer)
        if code.dtype != np.int64 or code.shape != (S4.shape[0],):
            raise ValueError("triad code buffer must be a row-aligned int64 vector")

    cache_bank = None
    cache_lookup = None
    if pair_prefix_cache is not None:
        cache_bank, cache_lookup = pair_prefix_cache
    cache_index = None if cache_lookup is None else cache_lookup.get((int(a), int(b)))
    if cache_index is None:
        np.multiply(S4[:, a], 4, out=code, casting="unsafe")
        np.add(code, S4[:, b], out=code, casting="unsafe")
        np.multiply(code, 4, out=code)
        np.add(code, S4[:, c], out=code, casting="unsafe")
    else:
        np.add(
            cache_bank[int(cache_index)],
            S4[:, c],
            out=code,
            casting="unsafe",
        )

    G = np.bincount(code, weights=g, minlength=64).reshape(4, 4, 4)
    H = np.bincount(code, weights=h, minlength=64).reshape(4, 4, 4)
    # geom.logistic_gh rejects non-positive row Hessians. Therefore a cell has
    # row support iff its accumulated Hessian is positive, so the historical
    # third count histogram is exactly redundant.
    support = H > 0
    return G, H, support


def _native_triad_histogram_bank(
    S4: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    triads: np.ndarray,
):
    """Return an exact native G/H bank when an additive core is available."""
    triad_array = np.ascontiguousarray(triads, dtype=np.int32).reshape(-1, 3)
    output_bytes = int(len(triad_array)) * 64 * 2 * np.dtype(np.float64).itemsize
    if output_bytes > _TRIAD_NATIVE_BANK_BUDGET_BYTES:
        return None

    try:
        from ._internal.cerm_training_core_runtime import (
            _prebuilt_library_path,
            load_native_training_core,
        )
        already_loaded = load_native_training_core.cache_info().currsize > 0
        prebuilt = _prebuilt_library_path()
        if not already_loaded and not (prebuilt is not None and prebuilt.is_file()):
            return None
        core = load_native_training_core()
    except Exception:
        return None

    if getattr(core, "_triad_histogram", None) is None:
        return None

    values = np.column_stack(
        [
            np.asarray(g, dtype=np.float64),
            np.asarray(h, dtype=np.float64),
        ]
    )
    n_threads = min(
        4,
        max(1, os.cpu_count() or 1),
        max(1, len(triad_array)),
    )
    try:
        bank = core.triad_histogram(
            np.ascontiguousarray(S4, dtype=np.uint8),
            triad_array,
            values,
            n_threads=n_threads,
        )
    except Exception:
        return None
    return np.asarray(bank, dtype=np.float64), int(n_threads), output_bytes



_TRIAD_STAGE1_GUARD_REL = 1e-10


def _native_fused_triad_stage1(
    S4: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    triads: np.ndarray,
    critical: float,
    *,
    guard_rel: float = _TRIAD_STAGE1_GUARD_REL,
):
    """Run native triad histogram construction and guarded Stage-1 together."""
    triad_array = np.ascontiguousarray(triads, dtype=np.int32).reshape(-1, 3)
    full_bank_bytes = (
        int(len(triad_array)) * 64 * 2 * np.dtype(np.float64).itemsize
    )
    if full_bank_bytes > _TRIAD_NATIVE_BANK_BUDGET_BYTES:
        return None

    if (
        getattr(_native_triad_histogram_bank, "__name__", "")
        != "_native_triad_histogram_bank"
        and _native_triad_histogram_bank(S4, g, h, triads) is None
    ):
        return None

    try:
        from ._internal.cerm_training_core_runtime import (
            _prebuilt_library_path,
            load_native_training_core,
        )
        already_loaded = load_native_training_core.cache_info().currsize > 0
        prebuilt = _prebuilt_library_path()
        if not already_loaded and not (
            prebuilt is not None and prebuilt.is_file()
        ):
            return None
        core = load_native_training_core()
    except Exception:
        return None
    if getattr(core, "_triad_fused_stage1", None) is None:
        return None

    values = np.column_stack(
        [
            np.asarray(g, dtype=np.float64),
            np.asarray(h, dtype=np.float64),
        ]
    )
    n_threads = min(
        4,
        max(1, os.cpu_count() or 1),
        max(1, len(triad_array)),
    )
    try:
        flags, cheap, bank = core.triad_fused_stage1(
            np.ascontiguousarray(S4, dtype=np.uint8),
            triad_array,
            values,
            critical=critical,
            guard_rel=guard_rel,
            n_threads=n_threads,
        )
    except Exception:
        return None

    clear_reject = (flags & 1) != 0
    survivor_count = int(np.count_nonzero(~clear_reject))
    output_bytes = survivor_count * 64 * 2 * np.dtype(np.float64).itemsize
    return flags, cheap, bank, int(n_threads), output_bytes


def _vectorized_native_triad_stage1(
    native_bank: np.ndarray,
    critical: float,
    *,
    guard_rel: float = _TRIAD_STAGE1_GUARD_REL,
):
    """Return only numerically clear cheap-screen rejects from a native G/H bank.

    This routine is deliberately one-sided.  It never supplies the cheap value
    used for survivor ordering: candidates that are not clearly rejected are
    recomputed through the historical scalar arithmetic before sweep screening.
    A relative guard around the chi-square boundary sends near-threshold rows to
    that scalar path.
    """
    bank = np.asarray(native_bank, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[1:] != (64, 2):
        raise ValueError("native triad bank must have shape (n_triads, 64, 2)")
    if len(bank) == 0:
        empty = np.empty(0, dtype=bool)
        return empty, empty, np.empty(0, dtype=np.float64)

    G = bank[:, :, 0].reshape(-1, 4, 4, 4)
    H = bank[:, :, 1].reshape(-1, 4, 4, 4)
    valid = H > 0.0

    full_terms = np.divide(
        G * G,
        H,
        out=np.zeros_like(G),
        where=valid,
    )
    full_gain = 0.5 * np.sum(full_terms, axis=(1, 2, 3))

    pair_gains = []
    for axis in (3, 2, 1):
        pair_G = np.sum(G, axis=axis)
        pair_H = np.sum(H, axis=axis)
        pair_valid = pair_H > 0.0
        pair_terms = np.divide(
            pair_G * pair_G,
            pair_H,
            out=np.zeros_like(pair_G),
            where=pair_valid,
        )
        pair_gains.append(0.5 * np.sum(pair_terms, axis=(1, 2)))

    cheap = np.maximum(
        0.0,
        full_gain - np.maximum.reduce(pair_gains),
    )
    threshold = 0.5 * float(critical)
    scale = np.maximum.reduce(
        [
            np.ones_like(cheap),
            np.abs(cheap),
            np.full_like(cheap, abs(threshold)),
            np.abs(full_gain),
            *[np.abs(value) for value in pair_gains],
        ]
    )
    guard = float(guard_rel) * scale
    full_support = np.all(valid, axis=(1, 2, 3))
    ambiguous = np.abs(cheap - threshold) <= guard
    clear_reject = (
        full_support
        & (cheap < threshold - guard)
        & ~ambiguous
    )
    return clear_reject, ambiguous, cheap


def best_triad_candidate(graph: ProjectedTrainingGraph, alpha2: float, solve_cap: int):
    if 4 not in graph.states_by_q or len(graph.selected_raw_features) < 3:
        return None, {
            "n_triads": 0,
            "safe_survivors": 0,
            "exact_solves": 0,
            "stage2_passes": 0,
            "triad_exact_diagnostics": [],
            "min_triad_p_ref": None,
        }
    S4 = graph.states_by_q[4]
    full_df = 27
    critical = float(chi2.ppf(1.0 - float(alpha2), full_df))
    rows = []
    triads = np.asarray(
        list(combinations(range(S4.shape[1]), 3)),
        dtype=np.int32,
    ).reshape(-1, 3)
    n_triads = int(len(triads))
    cheap_reject = 0; sweep_reject = 0
    cache_hits = 0; cache_misses = 0
    survivor_stats = {}
    fused_result = _native_fused_triad_stage1(
        S4, graph.g, graph.h, triads, critical
    )

    if fused_result is not None:
        flags, _cheap_arr, native_bank, native_threads, native_bytes = fused_result
        triad_backend = "native"
        fused_stage1 = True
        cache_bytes = 0
        cache_pairs = 0
        code_buffer = None
        pair_prefix_cache = None
        cache_lookup = {}
        native_clear_reject = (flags & 1) != 0
        native_ambiguous = (flags & 2) != 0
    else:
        fused_stage1 = False
        native_result = _native_triad_histogram_bank(
            S4, graph.g, graph.h, triads
        )
        if native_result is not None:
            native_bank, native_threads, native_bytes = native_result
            triad_backend = "native"
            cache_bytes = 0
            cache_pairs = 0
            code_buffer = None
            pair_prefix_cache = None
            cache_lookup = {}
            (
                native_clear_reject,
                native_ambiguous,
                _native_cheap_approx,
            ) = _vectorized_native_triad_stage1(
                native_bank,
                critical,
            )
        else:
            native_bank = None
            native_threads = 0
            native_bytes = 0
            triad_backend = "pair_prefix_cache"
            code_buffer = np.empty(S4.shape[0], dtype=np.int64)
            pair_prefix_cache = _build_triad_pair_prefix_cache(S4)
            cache_bank, cache_lookup = pair_prefix_cache
            cache_bytes = int(cache_bank.nbytes)
            cache_pairs = int(len(cache_lookup))
            native_clear_reject = None
            native_ambiguous = None
            _native_cheap_approx = None

    vectorized_rejects = 0
    vectorized_ambiguous = 0
    native_sweep_indices = []
    native_sweep_cheap = []
    native_sweep_full_support = []
    for triad_index, tri_array in enumerate(triads):
        tri = tuple(int(value) for value in tri_array)
        if native_bank is not None:
            if bool(native_clear_reject[triad_index]):
                vectorized_rejects += 1
                cheap_reject += 1
                continue
            if bool(native_ambiguous[triad_index]):
                vectorized_ambiguous += 1
            G = native_bank[triad_index, :, 0].reshape(4, 4, 4)
            H = native_bank[triad_index, :, 1].reshape(4, 4, 4)
            support = H > 0
        else:
            if (tri[0], tri[1]) in cache_lookup:
                cache_hits += 1
            else:
                cache_misses += 1
            G, H, support = _triad_cube(
                S4,
                graph.g,
                graph.h,
                tri,
                code_buffer=code_buffer,
                pair_prefix_cache=pair_prefix_cache,
            )

        v = H > 0
        full = 0.5 * float(np.sum(G[v] ** 2 / H[v])) if np.any(v) else 0.0
        cheap = max(0.0, full - max(_pair_gain(G, H, 2), _pair_gain(G, H, 1), _pair_gain(G, H, 0)))
        full_support = bool(np.all(support))
        if full_support and 2.0 * cheap <= critical:
            cheap_reject += 1; continue

        if native_bank is not None:
            native_sweep_indices.append(int(triad_index))
            native_sweep_cheap.append(float(cheap))
            native_sweep_full_support.append(bool(full_support))
            continue

        sweep = _pair_sweep_bound(G, H, support)
        if full_support and 2.0 * sweep <= critical:
            sweep_reject += 1; continue
        rows.append((sweep, cheap, tri))
        survivor_stats[tri] = (
            G.reshape(64).astype(np.float64).copy(),
            H.reshape(64).astype(np.float64).copy(),
        )

    native_sweep_batch_size = 0
    if native_bank is not None and native_sweep_indices:
        sweep_ids = np.asarray(native_sweep_indices, dtype=np.int64)
        sweep_G = native_bank[sweep_ids, :, 0].reshape(-1, 4, 4, 4)
        sweep_H = native_bank[sweep_ids, :, 1].reshape(-1, 4, 4, 4)
        sweep_values = _pair_sweep_bound_batch(sweep_G, sweep_H)
        native_sweep_batch_size = int(len(sweep_ids))
        for position, triad_index in enumerate(sweep_ids):
            sweep = float(sweep_values[position])
            cheap = float(native_sweep_cheap[position])
            full_support = bool(native_sweep_full_support[position])
            if full_support and 2.0 * sweep <= critical:
                sweep_reject += 1
                continue
            tri = tuple(int(value) for value in triads[int(triad_index)])
            rows.append((sweep, cheap, tri))

    rows.sort(key=lambda r: (-r[0], -r[1], r[2]))

    # Retain only the tiny exact G/H payload needed by Stage 2. The complete
    # native bank can be up to 8 MiB, while solve_cap is normally <=16, so this
    # turns the screening artifact into an output-sensitive ~16 KiB cache.
    stage2_stats = {}
    if native_bank is not None and rows:
        triad_index = {
            tuple(int(value) for value in triad): int(index)
            for index, triad in enumerate(triads)
        }
        for _, _, tri in rows[: int(solve_cap)]:
            index = triad_index[tri]
            stage2_stats[tri] = (
                native_bank[index, :, 0].copy(),
                native_bank[index, :, 1].copy(),
            )
    elif native_bank is None and rows:
        for _, _, tri in rows[: int(solve_cap)]:
            if tri in survivor_stats:
                stage2_stats[tri] = survivor_stats[tri]
    stage2_stats_bytes = int(
        sum(G.nbytes + H.nbytes for G, H in stage2_stats.values())
    )

    # Full screening buffers have no role after the selected exact G/H payload
    # is copied. End their lifetime before QR allocations in the exact solve.
    del code_buffer, pair_prefix_cache, native_bank

    solved = []
    for _, _, tri in rows[: int(solve_cap)]:
        if tri in stage2_stats:
            G_state, H_state = stage2_stats[tri]
            df, gain, p_ref = geom.triad_score_from_stats(
                G_state,
                H_state,
                q=4,
            )
        else:
            basis = geom.triad_basis(
                S4,
                graph.y,
                graph.score,
                tri,
                q=4,
                name=f"triad:{tri}",
            )
            df = int(basis.df)
            gain = float(basis.gain)
            p_ref = float(basis.p_ref)
        solved.append((p_ref, -gain, tri, df, gain))
    solved.sort(key=lambda x: (x[0], x[1], x[2]))
    passes = [
        x
        for x in solved
        if x[3] > 0 and np.isfinite(x[0]) and x[0] < float(alpha2)
    ]
    triad_diagnostics = []
    for p_ref, _, tri, df, gain in solved:
        raw_tri = tuple(
            sorted(int(graph.selected_raw_features[index]) for index in tri)
        )
        diagnostic_candidate = DeficitCandidate(
            family="interaction",
            name=f"triad:{raw_tri}",
            p_ref=float(p_ref),
            deficit=float(gain),
            df=int(df),
            triad=tuple(int(index) for index in tri),
            raw_triad=raw_tri,
            source_q=4,
            estimated_bytes=64 * 8,
        )
        triad_diagnostics.append(
            _candidate_diagnostic(
                diagnostic_candidate,
                passed=bool(
                    df > 0
                    and np.isfinite(p_ref)
                    and p_ref < float(alpha2)
                ),
            )
        )
    meta = {
        "n_triads": n_triads, "cheap_safe_reject": cheap_reject, "sweep_safe_reject": sweep_reject,
        "safe_survivors": len(rows), "exact_solves": min(len(rows), int(solve_cap)), "stage2_passes": len(passes),
        "triad_histogram_backend": triad_backend,
        "triad_native_threads": native_threads,
        "triad_native_bank_bytes": native_bytes,
        "triad_pair_cache_bytes": cache_bytes,
        "triad_pair_cache_pairs": cache_pairs,
        "triad_pair_cache_hits": cache_hits,
        "triad_pair_cache_misses": cache_misses,
        "triad_support_from_hessian": True,
        "triad_stage1_backend": (
            "vectorized_guarded" if triad_backend == "native" else "scalar"
        ),
        "triad_stage1_fused": bool(fused_stage1),
        "triad_stage1_vectorized_rejects": int(vectorized_rejects),
        "triad_stage1_ambiguous": int(vectorized_ambiguous),
        "triad_sweep_backend": (
            "vectorized_exact" if triad_backend == "native" else "scalar"
        ),
        "triad_sweep_batch_size": int(native_sweep_batch_size),
        "triad_stage2_stats_reuse": bool(stage2_stats),
        "triad_stage2_stats_count": int(len(stage2_stats)),
        "triad_stage2_stats_bytes": int(stage2_stats_bytes),
        "triad_exact_diagnostics": triad_diagnostics,
        "min_triad_p_ref": (
            min(float(row["p_ref"]) for row in triad_diagnostics)
            if triad_diagnostics
            else None
        ),
    }
    if not passes: return None, meta
    p, _, tri, df, gain = passes[0]
    raw_tri = tuple(sorted(int(graph.selected_raw_features[j]) for j in tri))
    c = DeficitCandidate(
        family="interaction", name=f"triad:{raw_tri}", p_ref=float(p), deficit=float(gain), df=int(df),
        triad=tuple(int(j) for j in tri), raw_triad=raw_tri, source_q=4, estimated_bytes=64 * 8,
    )
    if tri in stage2_stats:
        G_state, H_state = stage2_stats[tri]
        graph.basis_stat_cache[c.name] = (
            np.asarray(G_state, dtype=np.float64).copy(),
            np.asarray(H_state, dtype=np.float64).copy(),
        )
    return c, meta


def _with_candidate_search_diagnostics(
    metadata: dict,
    resolution_diagnostics: Sequence[dict],
) -> dict:
    """Attach complete training-side deficit diagnostics to search metadata."""
    output = dict(metadata)
    resolution_rows = [dict(row) for row in resolution_diagnostics]
    triad_rows = [
        dict(row) for row in output.get("triad_exact_diagnostics", [])
    ]
    output["resolution_diagnostics"] = resolution_rows
    output["resolution_evaluated"] = len(resolution_rows)
    output["stage2_q4_q8_passes"] = sum(
        bool(row["passed_stage2"])
        and row.get("source_q") == 4
        and row.get("target_q") == 8
        for row in resolution_rows
    )
    output["stage2_q8_q16_passes"] = sum(
        bool(row["passed_stage2"])
        and row.get("source_q") == 8
        and row.get("target_q") == 16
        for row in resolution_rows
    )

    def minimum_p(rows: Sequence[dict]) -> float | None:
        values = [
            float(row["p_ref"])
            for row in rows
            if np.isfinite(float(row["p_ref"]))
        ]
        return min(values) if values else None

    q4_q8 = [
        row
        for row in resolution_rows
        if row.get("source_q") == 4 and row.get("target_q") == 8
    ]
    q8_q16 = [
        row
        for row in resolution_rows
        if row.get("source_q") == 8 and row.get("target_q") == 16
    ]
    output["min_resolution_p_ref"] = minimum_p(resolution_rows)
    output["min_q4_q8_p_ref"] = minimum_p(q4_q8)
    output["min_q8_q16_p_ref"] = minimum_p(q8_q16)
    output["min_triad_p_ref"] = minimum_p(triad_rows)

    all_rows = resolution_rows + triad_rows
    finite_rows = [
        row for row in all_rows if np.isfinite(float(row["p_ref"]))
    ]
    strongest = (
        min(
            finite_rows,
            key=lambda row: (
                float(row["p_ref"]),
                -float(row["deficit"]),
                str(row["name"]),
            ),
        )
        if finite_rows
        else None
    )
    output["strongest_candidate"] = (
        None if strongest is None else dict(strongest)
    )
    output["best_stage2_reference_p_ref"] = (
        None if strongest is None else float(strongest["p_ref"])
    )
    return output


def build_candidate_bank(graph: ProjectedTrainingGraph, config: UNDRConfig):
    config.validate()
    resolution_diagnostics: List[dict] = []
    resolution_meta: dict = {}
    res = resolution_candidates(
        graph,
        config.alpha2,
        diagnostics=resolution_diagnostics,
        metadata=resolution_meta,
    )
    tri, tri_meta = best_triad_candidate(
        graph, config.alpha2, config.maximum_stage2_triads
    )
    tri_meta.update(resolution_meta)
    tri_meta = _with_candidate_search_diagnostics(
        tri_meta,
        resolution_diagnostics,
    )
    tri_meta["selection_stat_cache_count"] = int(
        len(graph.basis_stat_cache)
    )
    tri_meta["selection_stat_cache_bytes"] = int(
        sum(
            np.asarray(G).nbytes + np.asarray(H).nbytes
            for G, H in graph.basis_stat_cache.values()
        )
    )
    bank = list(res) + ([] if tri is None else [tri])
    family_trials = {
        "resolution": len(graph.selected_raw_features) * sum(q in graph.states_by_q for q in (8, 16)),
        "interaction": comb(len(graph.selected_raw_features), 3) if len(graph.selected_raw_features) >= 3 else 0,
    }
    return bank, family_trials, tri_meta


def _basis(candidate: DeficitCandidate, graph: ProjectedTrainingGraph):
    retained = graph.basis_stat_cache.get(candidate.name)
    if candidate.family == "resolution":
        if retained is not None:
            G_state, H_state = retained
            return geom.resolution_basis_from_stats(
                graph.states_by_q[candidate.target_q],
                candidate.feature,
                candidate.source_q,
                candidate.target_q,
                G_state,
                H_state,
                name=candidate.name,
            )
        return geom.resolution_basis(
            graph.states_by_q[candidate.target_q], graph.y, graph.score, candidate.feature,
            candidate.source_q, candidate.target_q, name=candidate.name,
        )
    if candidate.family == "interaction":
        if retained is not None:
            G_state, H_state = retained
            return geom.triad_basis_from_stats(
                graph.states_by_q[4],
                candidate.triad,
                G_state,
                H_state,
                q=4,
                name=candidate.name,
            )
        return geom.triad_basis(
            graph.states_by_q[4],
            graph.y,
            graph.score,
            candidate.triad,
            q=4,
            name=candidate.name,
        )
    raise KeyError(candidate.family)


def _surprise(p: float) -> float:
    return -log(max(float(p), np.finfo(float).tiny))



_CONDITIONAL_STATE_BANK_BUDGET_BYTES = 64 * 1024 * 1024


def _native_conditional_cross_grams(
    selected: Sequence[geom.StateQuotientBasis],
    candidates: Sequence[geom.StateQuotientBasis],
    h: np.ndarray,
):
    """Return selected->candidate cross-Gram blocks from one native pair bank.

    The existing ABI-v2 pair histogram kernel is reused; no new native symbol is
    required.  A packed uint8 state matrix is ephemeral to this selection round.
    """
    if not selected or not candidates:
        return []
    n_rows = len(np.asarray(h))
    bases = list(selected) + list(candidates)
    packed_bytes = int(n_rows) * int(len(bases))
    if packed_bytes > int(_CONDITIONAL_STATE_BANK_BUDGET_BYTES):
        return None
    if any(len(b.codes) != n_rows for b in bases):
        raise ValueError("basis row mismatch")
    if any(int(b.nstates) <= 0 or int(b.nstates) > 255 for b in bases):
        return None

    try:
        from ._internal.cerm_training_core_runtime import (
            _prebuilt_library_path,
            load_native_training_core,
        )
        already_loaded = load_native_training_core.cache_info().currsize > 0
        prebuilt = _prebuilt_library_path()
        if not already_loaded and not (prebuilt is not None and prebuilt.is_file()):
            return None
        core = load_native_training_core()
    except Exception:
        return None

    state_columns = []
    for basis in bases:
        codes = np.asarray(basis.codes)
        if codes.ndim != 1 or len(codes) != n_rows:
            return None
        if len(codes) and (
            np.min(codes, initial=0) < 0
            or np.max(codes, initial=0) >= int(basis.nstates)
        ):
            return None
        state_columns.append(np.asarray(codes, dtype=np.uint8))
    states = np.ascontiguousarray(np.column_stack(state_columns), dtype=np.uint8)
    cards = np.asarray([int(b.nstates) for b in bases], dtype=np.int32)

    n_selected = len(selected)
    pairs = np.asarray(
        [
            (left, n_selected + right)
            for left in range(n_selected)
            for right in range(len(candidates))
        ],
        dtype=np.int32,
    )
    values = np.ascontiguousarray(np.asarray(h, dtype=np.float64)[:, None])
    pair_count = len(pairs)
    if pair_count == 0:
        return []
    # Thread launch overhead dominates tiny banks; larger rounds benefit from
    # pair-level parallelism in the existing native kernel.
    work = int(n_rows) * int(pair_count)
    n_threads = 1 if work < 250_000 else min(
        4,
        max(1, os.cpu_count() or 1),
        pair_count,
    )
    try:
        offsets, sums = core.pair_histogram(
            states,
            cards,
            pairs,
            values,
            n_threads=n_threads,
        )
    except Exception:
        return None

    cross = [
        [None for _ in selected]
        for _ in candidates
    ]
    pair_index = 0
    for left_index, left in enumerate(selected):
        for right_index, right in enumerate(candidates):
            begin = int(offsets[pair_index])
            end = int(offsets[pair_index + 1])
            expected = int(left.nstates) * int(right.nstates)
            if end - begin != expected:
                return None
            joint = np.asarray(
                sums[begin:end, 0],
                dtype=np.float64,
            ).reshape(int(left.nstates), int(right.nstates))
            cross[right_index][left_index] = (
                left.A.T @ joint @ right.A
            )
            pair_index += 1
    return [
        np.vstack(blocks)
        for blocks in cross
    ]


def _conditional_gains_batched(
    selected: Sequence[geom.StateQuotientBasis],
    candidates: Sequence[geom.StateQuotientBasis],
    h: np.ndarray,
):
    """Evaluate one conditional-selection round with a shared selected Gram."""
    if not candidates:
        return [], "none"

    cross_blocks = _native_conditional_cross_grams(
        selected,
        candidates,
        h,
    )
    if cross_blocks is None:
        return [
            geom.conditional_gain(selected, candidate, h)
            for candidate in candidates
        ], "scalar"

    K0, t0 = geom.gram_and_target(selected, h)
    g0, r0, _ = geom.span_gain_from_gram(K0, t0)
    output = []
    selected_dim = int(K0.shape[0])
    for candidate, X in zip(candidates, cross_blocks):
        candidate_dim = int(candidate.df)
        K1 = np.zeros(
            (selected_dim + candidate_dim, selected_dim + candidate_dim),
            dtype=np.float64,
        )
        K1[:selected_dim, :selected_dim] = K0
        K1[:selected_dim, selected_dim:] = X
        K1[selected_dim:, :selected_dim] = X.T
        K1[selected_dim:, selected_dim:] = np.eye(candidate_dim)
        t1 = np.concatenate([t0, candidate.t])
        g1, r1, _ = geom.span_gain_from_gram(K1, t1)
        dg = max(0.0, float(g1) - float(g0))
        dr = max(0, int(r1) - int(r0))
        p = float(chi2.sf(2.0 * dg, dr)) if dr else 1.0
        output.append((dg, dr, p))
    return output, "native_pair_bank"



@dataclass
class _ConditionalPairWorkspace:
    """One-time packed state bank for repeated conditional-selection rounds."""

    core: object
    states: np.ndarray
    cards: np.ndarray
    values: np.ndarray
    column_by_name: Dict[str, int]

    @classmethod
    def build(
        cls,
        candidates: Sequence[DeficitCandidate],
        bases: Sequence[geom.StateQuotientBasis],
        h: np.ndarray,
    ):
        if len(candidates) != len(bases) or len(candidates) < 1:
            return None
        names = [str(candidate.name) for candidate in candidates]
        if len(set(names)) != len(names):
            return None
        n_rows = len(np.asarray(h))
        packed_bytes = int(n_rows) * int(len(bases))
        if packed_bytes > int(_CONDITIONAL_STATE_BANK_BUDGET_BYTES):
            return None
        if any(len(basis.codes) != n_rows for basis in bases):
            return None
        if any(
            int(basis.nstates) <= 0 or int(basis.nstates) > 255
            for basis in bases
        ):
            return None

        try:
            from ._internal.cerm_training_core_runtime import (
                _prebuilt_library_path,
                load_native_training_core,
            )
            already_loaded = load_native_training_core.cache_info().currsize > 0
            prebuilt = _prebuilt_library_path()
            if not already_loaded and not (
                prebuilt is not None and prebuilt.is_file()
            ):
                return None
            core = load_native_training_core()
        except Exception:
            return None

        columns = []
        for basis in bases:
            codes = np.asarray(basis.codes)
            if codes.ndim != 1 or len(codes) != n_rows:
                return None
            if len(codes) and (
                np.min(codes, initial=0) < 0
                or np.max(codes, initial=0) >= int(basis.nstates)
            ):
                return None
            columns.append(np.asarray(codes, dtype=np.uint8))

        states = np.ascontiguousarray(
            np.column_stack(columns),
            dtype=np.uint8,
        )
        cards = np.asarray(
            [int(basis.nstates) for basis in bases],
            dtype=np.int32,
        )
        values = np.asarray(h, dtype=np.float64)
        if values.ndim != 1 or len(values) != n_rows:
            return None
        values = np.ascontiguousarray(values[:, None])
        return cls(
            core=core,
            states=states,
            cards=cards,
            values=values,
            column_by_name={
                name: index for index, name in enumerate(names)
            },
        )

    @property
    def packed_bytes(self) -> int:
        return int(self.states.nbytes)

    def cross_grams(
        self,
        selected_candidates: Sequence[DeficitCandidate],
        selected_bases: Sequence[geom.StateQuotientBasis],
        remaining_candidates: Sequence[DeficitCandidate],
        remaining_bases: Sequence[geom.StateQuotientBasis],
    ):
        if (
            len(selected_candidates) != len(selected_bases)
            or len(remaining_candidates) != len(remaining_bases)
            or not selected_candidates
            or not remaining_candidates
        ):
            return None
        try:
            left_slots = [
                int(self.column_by_name[candidate.name])
                for candidate in selected_candidates
            ]
            right_slots = [
                int(self.column_by_name[candidate.name])
                for candidate in remaining_candidates
            ]
        except KeyError:
            return None

        pairs = np.asarray(
            [
                (left, right)
                for left in left_slots
                for right in right_slots
            ],
            dtype=np.int32,
        )
        pair_count = int(len(pairs))
        if pair_count == 0:
            return []
        work = int(len(self.states)) * pair_count
        n_threads = 1 if work < 250_000 else min(
            4,
            max(1, os.cpu_count() or 1),
            pair_count,
        )
        try:
            offsets, sums = self.core.pair_histogram(
                self.states,
                self.cards,
                pairs,
                self.values,
                n_threads=n_threads,
            )
        except Exception:
            return None

        cross = [
            [None for _ in selected_bases]
            for _ in remaining_bases
        ]
        pair_index = 0
        for left_index, left in enumerate(selected_bases):
            for right_index, right in enumerate(remaining_bases):
                begin = int(offsets[pair_index])
                end = int(offsets[pair_index + 1])
                expected = int(left.nstates) * int(right.nstates)
                if end - begin != expected:
                    return None
                joint = np.asarray(
                    sums[begin:end, 0],
                    dtype=np.float64,
                ).reshape(int(left.nstates), int(right.nstates))
                cross[right_index][left_index] = (
                    left.A.T @ joint @ right.A
                )
                pair_index += 1
        return [np.vstack(blocks) for blocks in cross]


def _conditional_gains_from_gram(
    K0: np.ndarray,
    t0: np.ndarray,
    candidates: Sequence[geom.StateQuotientBasis],
    cross_blocks: Sequence[np.ndarray],
):
    """Evaluate conditional gains from an already-frozen selected Gram."""
    if len(candidates) != len(cross_blocks):
        raise ValueError("candidate/cross-block length mismatch")
    g0, r0, _ = geom.span_gain_from_gram(K0, t0)
    selected_dim = int(K0.shape[0])
    output = []
    for candidate, X in zip(candidates, cross_blocks):
        candidate_dim = int(candidate.df)
        K1 = np.zeros(
            (selected_dim + candidate_dim, selected_dim + candidate_dim),
            dtype=np.float64,
        )
        K1[:selected_dim, :selected_dim] = K0
        K1[:selected_dim, selected_dim:] = X
        K1[selected_dim:, :selected_dim] = X.T
        K1[selected_dim:, selected_dim:] = np.eye(candidate_dim)
        t1 = np.concatenate([t0, candidate.t])
        g1, r1, _ = geom.span_gain_from_gram(K1, t1)
        dg = max(0.0, float(g1) - float(g0))
        dr = max(0, int(r1) - int(r0))
        p = float(chi2.sf(2.0 * dg, dr)) if dr else 1.0
        output.append((dg, dr, p, K1, t1))
    return output


def select_budgeted(bank: Sequence[DeficitCandidate], graph: ProjectedTrainingGraph, config: UNDRConfig,
                    family_trials: Dict[str, int]) -> Selection:
    pool = [c for c in bank if c.df > 0 and c.p_ref < float(config.alpha2)]
    if config.budget <= 0 or not pool:
        return Selection([], [], [])
    first_rows = []
    for c in pool:
        m = max(1, int(family_trials.get(c.family, 1)))
        score = _surprise(c.p_ref) - float(config.family_volume_rho) * log(m)
        first_rows.append((-score, c.p_ref, c.name, c))
    first_rows.sort(key=lambda r: (r[0], r[1], r[2]))
    first_score = -first_rows[0][0]
    first = first_rows[0][3]
    basis_cache: Dict[str, geom.StateQuotientBasis] = {}

    def cached_basis(candidate: DeficitCandidate):
        basis = basis_cache.get(candidate.name)
        if basis is None:
            basis = _basis(candidate, graph)
            basis_cache[candidate.name] = basis
        return basis

    selected = [first]; bases = [cached_basis(first)]
    trace = [{"stage": "first", "candidate": first.name, "family": first.family,
              "p_ref": first.p_ref, "adjusted_surprise": first_score,
              "family_trials": int(family_trials.get(first.family, 1)), "kept": True}]
    remaining = [c for c in pool if c is not first]
    threshold = _surprise(config.alpha2)

    workspace = None
    K_selected = None
    t_selected = None
    basis_by_name = None
    workspace_backend = "none"
    workspace_packed_bytes = 0
    if len(remaining) >= 1 and int(config.budget) > 1:
        pool_bases = [cached_basis(candidate) for candidate in pool]
        workspace = _ConditionalPairWorkspace.build(
            pool,
            pool_bases,
            graph.h,
        )
        if workspace is not None:
            workspace_backend = "native_pair_workspace"
            workspace_packed_bytes = int(getattr(workspace, "packed_bytes", 0))
            basis_by_name = {
                candidate.name: basis
                for candidate, basis in zip(pool, pool_bases)
            }
            first_basis = basis_by_name[first.name]
            K_selected = np.eye(first_basis.df, dtype=np.float64)
            t_selected = np.asarray(first_basis.t, dtype=np.float64).copy()

    while remaining and len(selected) < int(config.budget):
        rows = []
        if basis_by_name is None:
            remaining_bases = [cached_basis(c) for c in remaining]
        else:
            remaining_bases = [basis_by_name[c.name] for c in remaining]

        cross_blocks = None
        incremental_values = None
        if workspace is not None and len(remaining) >= 1:
            cross_blocks = workspace.cross_grams(
                selected,
                bases,
                remaining,
                remaining_bases,
            )
            if cross_blocks is not None:
                incremental_values = _conditional_gains_from_gram(
                    K_selected,
                    t_selected,
                    remaining_bases,
                    cross_blocks,
                )

        if incremental_values is None:
            conditional_values, conditional_backend = _conditional_gains_batched(
                bases,
                remaining_bases,
                graph.h,
            )
            row_values = [
                (dg, ddf, p, None, None)
                for dg, ddf, p in conditional_values
            ]
        else:
            row_values = incremental_values

        for c, b, (dg, ddf, p, K1, t1) in zip(
            remaining,
            remaining_bases,
            row_values,
        ):
            cost_ratio = max(1.0, float(c.estimated_bytes) / float(config.byte_unit))
            adjusted = _surprise(p) - float(config.second_cost_eta) * log(cost_ratio)
            rows.append(
                (
                    -adjusted, p, -dg, c.name, c, b, dg, ddf,
                    adjusted, cost_ratio, K1, t1,
                )
            )
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
        (
            _, p, _, _, c, b, dg, ddf, adjusted, cost_ratio,
            chosen_K, chosen_t,
        ) = rows[0]
        keep = bool(ddf > 0 and p < float(config.alpha2) and adjusted > threshold)
        trace.append({"stage": "conditional", "candidate": c.name, "family": c.family,
                      "conditional_gain": float(dg), "conditional_df": int(ddf), "conditional_p": float(p),
                      "estimated_bytes": int(c.estimated_bytes), "cost_ratio": float(cost_ratio),
                      "adjusted_surprise": float(adjusted), "threshold": float(threshold), "kept": keep})
        if not keep: break
        selected.append(c)
        bases.append(b)
        if workspace is not None and chosen_K is not None:
            K_selected = chosen_K
            t_selected = chosen_t
        else:
            workspace = None
            K_selected = None
            t_selected = None
        remaining = [x for x in remaining if x is not c]

    final_K = K_selected
    final_t = t_selected
    if len(bases) == 1 and (final_K is None or final_t is None):
        final_K = np.eye(bases[0].df, dtype=np.float64)
        final_t = np.asarray(bases[0].t, dtype=np.float64).copy()
    if final_K is not None and final_t is not None:
        graph.selection_gram_names = tuple(
            candidate.name for candidate in selected
        )
        graph.selection_gram_cache = (
            np.asarray(final_K, dtype=np.float64).copy(),
            np.asarray(final_t, dtype=np.float64).copy(),
        )
    return Selection(
        selected,
        bases,
        trace,
        workspace_backend=workspace_backend,
        workspace_packed_bytes=workspace_packed_bytes,
    )


def fit_joint(selection: Selection, graph: ProjectedTrainingGraph, config: UNDRConfig):
    selected_names = tuple(candidate.name for candidate in selection.candidates)
    retained = graph.selection_gram_cache
    if (
        retained is not None
        and graph.selection_gram_names == selected_names
    ):
        K_selected, t_selected = retained
        return geom.fit_joint_quotient_lookups_from_gram(
            selection.bases,
            K_selected,
            t_selected,
            l2=config.l2,
            lr=config.learning_rate,
        )
    return geom.fit_joint_quotient_lookups(
        selection.bases,
        graph.h,
        l2=config.l2,
        lr=config.learning_rate,
    )


def _codes_for_candidate(candidate: DeficitCandidate, states_by_q: Dict[int, np.ndarray]):
    if candidate.family == "resolution":
        return np.asarray(states_by_q[candidate.target_q][:, candidate.feature], dtype=np.int64)
    S = states_by_q[4]; a, b, c = candidate.triad
    return ((S[:, a].astype(np.int64) * 4 + S[:, b]) * 4 + S[:, c]).astype(np.int64)


def apply_frozen(score, selection: Selection, lookups: Sequence[np.ndarray], states_by_q: Dict[int, np.ndarray]):
    out = np.asarray(score, dtype=np.float64).copy()
    for c, lu in zip(selection.candidates, lookups):
        out += np.asarray(lu, dtype=np.float64)[_codes_for_candidate(c, states_by_q)]
    return out


def _logloss_each(y, score):
    y = np.asarray(y, dtype=np.float64); score = np.asarray(score, dtype=np.float64)
    return np.logaddexp(0.0, score) - y * score


def _empirical_bernstein_lcb(d, range_bound: float, delta: float):
    d = np.asarray(d, dtype=np.float64); n = len(d)
    if n < 2: return float(np.mean(d)) if n else 0.0, float("inf"), float("-inf")
    lt = float(np.log(2.0 / float(delta))); mean = float(np.mean(d)); var = float(np.var(d, ddof=1))
    rad = float(np.sqrt(2.0 * var * lt / n) + 7.0 * float(range_bound) * lt / (3.0 * (n - 1)))
    return mean, rad, mean - rad


class FrozenUNDRProgram:
    """Frozen opt-in correction around a fitted numeric binary CERM estimator."""
    def __init__(self, estimator, selected_raw_features, selection, lookups, accepted, audit,
                 state_encoder=None, state_encoder_projected=False):
        self.estimator = estimator
        self.selected_raw_features = np.asarray(selected_raw_features, dtype=np.int64)
        # Quotient bases contain training-row state codes and are required only
        # through joint fitting.  Freeze only the semantic candidate list and
        # trace; prediction reconstructs fresh state codes from X.
        self.selection = Selection(
            candidates=list(selection.candidates),
            bases=[],
            trace=[dict(row) for row in selection.trace],
        )
        self.lookups = [np.asarray(x, dtype=np.float64) for x in lookups]
        self.accepted = bool(accepted)
        self.audit = audit
        self.state_encoder = state_encoder
        self.state_encoder_projected = bool(state_encoder_projected)

    def _states(self, X):
        X = _numeric_matrix(X); _, base = _require_binary_fitted(self.estimator)
        encoder = base.encoder_ if self.state_encoder is None else self.state_encoder
        projected = False if self.state_encoder is None else self.state_encoder_projected
        need = sorted({int(c.target_q) if c.family == "resolution" else 4 for c in self.selection.candidates})
        states, _ = _state_transforms(
            encoder,
            projected,
            X,
            need,
            self.selected_raw_features,
        )
        return states

    def decision_function(self, X):
        X = _numeric_matrix(X); model, _ = _require_binary_fitted(self.estimator)
        score = _model_score(model, X)
        if not self.accepted or not self.selection.candidates: return score
        return apply_frozen(score, self.selection, self.lookups, self._states(X))

    def predict_proba(self, X):
        s = self.decision_function(X); p = 1.0 / (1.0 + np.exp(-np.clip(s, -50.0, 50.0)))
        return np.column_stack([1.0 - p, p])

    @property
    def model_bytes_(self):
        return int(sum(x.nbytes for x in self.lookups)) if self.accepted else 0


def fit_frozen_undr(estimator, X_train, y_train, X_holdout, y_holdout, *, config: UNDRConfig = UNDRConfig()):
    """Fit UNDR on training rows, freeze it, then use a fresh holdout once.

    The function does *not* refit ``estimator``. The caller is responsible for
    fitting the base CERM only on ``X_train, y_train`` before calling this API.
    """
    config.validate()
    train = extract_projected_training_graph(estimator, X_train, y_train, maximum_features=config.maximum_features)
    bank, trials, tri_meta = build_candidate_bank(train, config)
    selection = select_budgeted(bank, train, config, trials)
    lookups = fit_joint(selection, train, config) if selection.candidates else []

    Xh = _numeric_matrix(X_holdout); yh = np.asarray(y_holdout, dtype=np.int32)
    model, base = _require_binary_fitted(estimator)
    base_h = _model_score(model, Xh)
    hold_states = {}
    state_encoder = train.state_encoder if train.state_encoder is not None else base.encoder_
    state_projected = bool(train.state_encoder_projected)
    for q in sorted({int(c.target_q) if c.family == "resolution" else 4 for c in selection.candidates}):
        hold_states[q], _ = _state_transform(
            state_encoder, state_projected, Xh, q, train.selected_raw_features
        )
    new_h = apply_frozen(base_h, selection, lookups, hold_states) if selection.candidates else base_h.copy()
    if selection.candidates:
        d = _logloss_each(yh, base_h) - _logloss_each(yh, new_h)
        B = 2.0 * sum(float(np.max(np.abs(lu), initial=0.0)) for lu in lookups)
        gain, radius, lcb = _empirical_bernstein_lcb(d, B, config.holdout_delta)
        accepted = bool(lcb > 0.0)
    else:
        gain, radius, lcb, accepted = 0.0, float("nan"), float("nan"), False
    search_metadata = dict(tri_meta)
    search_metadata["conditional_workspace_backend"] = str(selection.workspace_backend)
    search_metadata["conditional_workspace_packed_bytes"] = int(selection.workspace_packed_bytes)
    search_metadata["transform_calls"] = {int(k): str(v) for k, v in train.transform_calls.items()}
    search_metadata["state_levels"] = tuple(sorted(int(k) for k in train.states_by_q))
    search_metadata["selected_raw_features"] = tuple(int(x) for x in train.selected_raw_features)
    search_metadata["state_bank_bytes"] = int(
        sum(np.asarray(state).nbytes for state in train.states_by_q.values())
    )
    retained_gram = train.selection_gram_cache
    search_metadata["selection_gram_reuse"] = bool(retained_gram is not None)
    search_metadata["selection_gram_cache_bytes"] = int(
        0
        if retained_gram is None
        else np.asarray(retained_gram[0]).nbytes
        + np.asarray(retained_gram[1]).nbytes
    )
    search_metadata["selection_gram_dimension"] = int(
        0 if retained_gram is None else np.asarray(retained_gram[1]).size
    )
    search_metadata["training_graph_bytes"] = int(
        train.selected_raw_features.nbytes
        + sum(np.asarray(state).nbytes for state in train.states_by_q.values())
        + train.y.nbytes
        + train.score.nbytes
        + train.g.nbytes
        + train.h.nbytes
    )
    search_metadata["training_basis_bytes_dropped_at_freeze"] = int(
        sum(
            np.asarray(basis.codes).nbytes
            + np.asarray(basis.A).nbytes
            + np.asarray(basis.t).nbytes
            for basis in selection.bases
        )
    )
    audit = UNDRFitAudit(
        selected_names=tuple(c.name for c in selection.candidates),
        selected_families=tuple(c.family for c in selection.candidates),
        stage2_resolution=sum(c.family == "resolution" for c in bank),
        stage2_triad=int(tri_meta.get("stage2_passes", 0)),
        triad_exact_solves=int(tri_meta.get("exact_solves", 0)),
        first_family_trials=trials,
        holdout_gain=float(gain), holdout_radius=float(radius), holdout_lcb=float(lcb), accepted=accepted,
        operator_bytes=int(sum(lu.nbytes for lu in lookups)) if accepted else 0,
        search_metadata=search_metadata, selection_trace=selection.trace,
    )
    return FrozenUNDRProgram(
        estimator, train.selected_raw_features, selection, lookups, accepted, audit,
        state_encoder=state_encoder, state_encoder_projected=state_projected,
    )
