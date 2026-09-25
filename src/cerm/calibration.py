from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    accepted: bool
    scale: float
    offset: float
    raw_logloss: float
    calibrated_logloss: float
    improvement: float
    improvement_se: float
    signal_to_noise: float
    folds: int
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float) and not np.isfinite(value):
                payload[key] = None
        return payload


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(logits)
    positive = logits >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    out[~positive] = exp_logits / (1.0 + exp_logits)
    return out


def fit_affine_calibrator(
    logits: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "affine",
    l2: float = 1e-3,
    min_improvement: float = 1e-4,
    min_signal: float = 1.0,
    scale_bounds: tuple[float, float] = (0.25, 4.0),
    offset_bound: float = 5.0,
    folds: int = 0,
) -> CalibrationResult:
    """Fit a positive-scale affine logit calibrator.

    ``method="intercept"`` optimizes only an offset. ``method="affine"``
    optimizes a positive scale and an offset.  Acceptance is based on
    unregularized cross-fitted log loss, while the optimization includes a
    small penalty towards the identity map.
    """

    if method not in {"intercept", "affine"}:
        raise ValueError("method must be 'intercept' or 'affine'")
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    if len(logits) != len(y):
        raise ValueError("logits and y must have the same length")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if l2 < 0 or min_improvement < 0 or min_signal < 0:
        raise ValueError(
            "l2, min_improvement, and min_signal must be non-negative"
        )

    raw_probability = np.clip(_sigmoid(logits), 1e-12, 1.0 - 1e-12)
    raw_loss = float(log_loss(y, raw_probability, labels=[0, 1]))

    lower, upper = map(float, scale_bounds)
    if not (0 < lower <= 1.0 <= upper):
        raise ValueError("scale_bounds must be positive and contain 1")

    if method == "intercept":
        def objective(parameters):
            offset = float(parameters[0])
            p = np.clip(_sigmoid(logits + offset), 1e-12, 1.0 - 1e-12)
            return float(log_loss(y, p, labels=[0, 1]) + l2 * offset * offset)

        result = minimize(
            objective,
            x0=np.asarray([0.0]),
            method="L-BFGS-B",
            bounds=[(-float(offset_bound), float(offset_bound))],
        )
        scale = 1.0
        offset = float(result.x[0])
    else:
        log_lower, log_upper = math.log(lower), math.log(upper)

        def objective(parameters):
            log_scale, offset = map(float, parameters)
            scale_value = math.exp(log_scale)
            p = np.clip(
                _sigmoid(scale_value * logits + offset),
                1e-12,
                1.0 - 1e-12,
            )
            penalty = l2 * (log_scale * log_scale + offset * offset)
            return float(log_loss(y, p, labels=[0, 1]) + penalty)

        result = minimize(
            objective,
            x0=np.asarray([0.0, 0.0]),
            method="L-BFGS-B",
            bounds=[
                (log_lower, log_upper),
                (-float(offset_bound), float(offset_bound)),
            ],
        )
        scale = float(math.exp(float(result.x[0])))
        offset = float(result.x[1])

    calibrated_probability = np.clip(
        _sigmoid(scale * logits + offset),
        1e-12,
        1.0 - 1e-12,
    )
    calibrated_loss = float(
        log_loss(y, calibrated_probability, labels=[0, 1])
    )
    raw_point_loss = -(
        y * np.log(raw_probability)
        + (1 - y) * np.log1p(-raw_probability)
    )
    calibrated_point_loss = -(
        y * np.log(calibrated_probability)
        + (1 - y) * np.log1p(-calibrated_probability)
    )
    point_improvement = raw_point_loss - calibrated_point_loss
    improvement = float(point_improvement.mean())
    improvement_se = float(
        point_improvement.std(ddof=1) / np.sqrt(len(point_improvement))
    ) if len(point_improvement) > 1 else 0.0
    signal_to_noise = (
        improvement / improvement_se
        if improvement_se > 0
        else (float("inf") if improvement > 0 else 0.0)
    )
    accepted = bool(
        result.success
        and np.isfinite(calibrated_loss)
        and improvement >= float(min_improvement)
        and improvement >= float(min_signal) * improvement_se
    )
    if not accepted:
        scale, offset, calibrated_loss, improvement = 1.0, 0.0, raw_loss, 0.0

    return CalibrationResult(
        method=method,
        accepted=accepted,
        scale=float(scale),
        offset=float(offset),
        raw_logloss=raw_loss,
        calibrated_logloss=float(calibrated_loss),
        improvement=float(improvement),
        improvement_se=float(improvement_se),
        signal_to_noise=float(signal_to_noise),
        folds=int(folds),
        converged=bool(result.success),
    )


def apply_affine_calibration(model, scale: float, offset: float, X: np.ndarray) -> None:
    """Apply an affine logit map directly to a fitted Hybrid program.

    The semantic and compiled programs remain equivalent because all final
    coefficients and the intercept are transformed before block tables and the
    replacement plan are rebuilt.
    """

    scale = float(scale)
    offset = float(offset)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("calibration scale must be finite and positive")
    if not np.isfinite(offset):
        raise ValueError("calibration offset must be finite")

    model.base_coef_ = np.asarray(model.base_coef_, dtype=np.float64) * scale
    model.block_coef_ = np.asarray(model.block_coef_, dtype=np.float64) * scale
    model.intercept_ = scale * float(model.intercept_) + offset
    model.clf_.coef_ = np.asarray(model.clf_.coef_, dtype=np.float64) * scale
    model.clf_.intercept_ = (
        np.asarray(model.clf_.intercept_, dtype=np.float64) * scale + offset
    )
    model._compile_block_tables()
    states = model._states(np.asarray(X, dtype=np.float64))
    model._build_replacement_plan_v2(states)
    model.model_bytes_estimate_ = int(
        model.base_.model_bytes_estimate_ + model.replacement_bytes_
    )


def cross_fitted_logits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learner_factory,
    selected_config,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Generate OOF logits using one already-selected semantic architecture."""

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    folds = min(int(n_splits), int(np.bincount(y).min()))
    folds = max(2, folds)
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(random_state),
    )
    logits = np.empty(len(y), dtype=np.float64)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        learner = learner_factory(int(random_state) + 104729 * (fold + 1))
        learner._fit_final(X[train_idx], y[train_idx], selected_config)
        logits[valid_idx] = learner.decision_function(X[valid_idx])
    return logits, folds
