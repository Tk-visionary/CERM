from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import (
    GammaRegressor,
    HuberRegressor,
    PoissonRegressor,
    QuantileRegressor,
    TweedieRegressor,
)


@dataclass(frozen=True)
class ObjectiveFitPlan:
    """Validated target transformations for a generalized objective.

    ``representation_target`` is used by the finite-state structure selector.
    ``target`` and ``sample_weight`` are passed to the loss-specific solver.
    ``log_scale`` records the fixed prediction offset for diagnostics.
    """

    representation_target: np.ndarray
    target: np.ndarray
    sample_weight: np.ndarray | None
    log_scale: np.ndarray | None


class ObjectiveBackend:
    """Internal protocol shared by generalized regression objectives."""

    name: str
    output_kind: str = "single"

    def validate_target(self, y: np.ndarray) -> None:
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinity")

    def representation_target(
        self,
        y: np.ndarray,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> np.ndarray:
        del exposure
        return y if offset is None else y - offset

    def prepare_fit(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> ObjectiveFitPlan:
        if exposure is not None:
            raise ValueError(f"exposure is supported only for Poisson regression, not {self.name}")
        target = y if offset is None else y - offset
        return ObjectiveFitPlan(
            representation_target=target,
            target=target,
            sample_weight=sample_weight,
            log_scale=offset,
        )

    def make_solvers(self):
        raise NotImplementedError

    def inverse_link(
        self,
        raw: np.ndarray,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> np.ndarray:
        if exposure is not None:
            raise ValueError(f"exposure is supported only for Poisson regression, not {self.name}")
        return raw if offset is None else raw + offset.reshape((-1,) + (1,) * (raw.ndim - 1))


class HuberBackend(ObjectiveBackend):
    name = "huber"

    def __init__(self, *, epsilon: float, alpha: float, max_iter: int, tol: float):
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def make_solvers(self):
        return [
            HuberRegressor(
                epsilon=self.epsilon,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=True,
            )
        ]


class QuantileBackend(ObjectiveBackend):
    name = "quantile"

    def __init__(self, *, quantiles: Sequence[float], alpha: float):
        self.quantiles = tuple(float(value) for value in quantiles)
        self.alpha = float(alpha)
        self.output_kind = "single" if len(self.quantiles) == 1 else "multi_quantile"

    def make_solvers(self):
        return [
            QuantileRegressor(
                quantile=quantile,
                alpha=self.alpha,
                fit_intercept=True,
                solver="highs",
            )
            for quantile in self.quantiles
        ]


class PositiveTweedieBackend(ObjectiveBackend):
    """Log-link positive-response backend with exact scale/offset handling.

    For Tweedie variance power ``p``, scaling both target and mean by ``s``
    multiplies the unit deviance by ``s**(2-p)``.  Therefore fitting ``y/s``
    with sample weights multiplied by ``s**(2-p)`` is exactly equivalent to
    fitting a model with fixed log-offset ``log(s)``.
    """

    def __init__(
        self,
        *,
        name: str,
        power: float,
        alpha: float,
        max_iter: int,
        tol: float,
    ):
        self.name = str(name)
        self.power = float(power)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def validate_target(self, y: np.ndarray) -> None:
        super().validate_target(y)
        if self.power >= 2.0:
            if np.any(y <= 0):
                raise ValueError(f"{self.name} regression requires strictly positive targets")
        elif self.power > 1.0:
            if np.any(y < 0):
                raise ValueError(f"{self.name} regression requires non-negative targets")

    @staticmethod
    def _scale(
        offset: np.ndarray | None, exposure: np.ndarray | None
    ) -> np.ndarray | None:
        if offset is None and exposure is None:
            return None
        if exposure is not None and np.any(exposure <= 0):
            raise ValueError("exposure must be strictly positive")
        if offset is None:
            scale = exposure.copy()
        else:
            scale = np.exp(offset)
            if exposure is not None:
                scale = scale * exposure
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("combined offset/exposure scale must be finite and positive")
        return scale

    def representation_target(
        self,
        y: np.ndarray,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> np.ndarray:
        scale = self._scale(offset, exposure)
        rate = y if scale is None else y / scale
        return np.log1p(rate)

    def prepare_fit(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> ObjectiveFitPlan:
        scale = self._scale(offset, exposure)
        if scale is None:
            return ObjectiveFitPlan(
                representation_target=np.log1p(y),
                target=y,
                sample_weight=sample_weight,
                log_scale=None,
            )
        target = y / scale
        multiplier = np.power(scale, 2.0 - self.power)
        weights = multiplier if sample_weight is None else sample_weight * multiplier
        if not np.isfinite(target).all() or not np.isfinite(weights).all():
            raise ValueError("offset/exposure produced non-finite transformed data")
        return ObjectiveFitPlan(
            representation_target=np.log1p(target),
            target=target,
            sample_weight=weights,
            log_scale=np.log(scale),
        )

    def make_solvers(self):
        if self.name == "gamma":
            solver = GammaRegressor(
                alpha=self.alpha,
                fit_intercept=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            solver = TweedieRegressor(
                power=self.power,
                alpha=self.alpha,
                link="log",
                fit_intercept=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        return [solver]

    def inverse_link(
        self,
        raw: np.ndarray,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> np.ndarray:
        scale = self._scale(offset, exposure)
        prediction = np.exp(raw)
        if scale is not None:
            prediction = prediction * scale.reshape((-1,) + (1,) * (raw.ndim - 1))
        return prediction


class PoissonBackend(ObjectiveBackend):
    name = "poisson"

    def __init__(self, *, alpha: float, max_iter: int, tol: float):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def validate_target(self, y: np.ndarray) -> None:
        super().validate_target(y)
        if np.any(y < 0):
            raise ValueError("Poisson regression requires non-negative targets")

    @staticmethod
    def _scale(
        n_samples: int,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> np.ndarray | None:
        if offset is None and exposure is None:
            return None
        if exposure is not None and np.any(exposure <= 0):
            raise ValueError("exposure must be strictly positive")
        if offset is None:
            return exposure.copy()
        offset_scale = np.exp(offset)
        if not np.isfinite(offset_scale).all():
            raise ValueError("Poisson offset is too large")
        scale = offset_scale if exposure is None else offset_scale * exposure
        if not np.isfinite(scale).all():
            raise ValueError("combined Poisson offset/exposure is not finite")
        return scale

    def representation_target(
        self,
        y: np.ndarray,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> np.ndarray:
        scale = self._scale(len(y), offset, exposure)
        rate = y if scale is None else y / scale
        return np.log1p(rate)

    def prepare_fit(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> ObjectiveFitPlan:
        scale = self._scale(len(y), offset, exposure)
        if scale is None:
            return ObjectiveFitPlan(
                representation_target=np.log1p(y),
                target=y,
                sample_weight=sample_weight,
                log_scale=None,
            )
        target = y / scale
        weights = scale if sample_weight is None else sample_weight * scale
        if not np.isfinite(target).all() or not np.isfinite(weights).all():
            raise ValueError("Poisson offset/exposure produced non-finite transformed data")
        log_scale = np.log(scale)
        return ObjectiveFitPlan(
            representation_target=np.log1p(target),
            target=target,
            sample_weight=weights,
            log_scale=log_scale,
        )

    def make_solvers(self):
        return [
            PoissonRegressor(
                alpha=self.alpha,
                fit_intercept=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        ]

    def inverse_link(
        self,
        raw: np.ndarray,
        *,
        offset: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> np.ndarray:
        scale = self._scale(len(raw), offset, exposure)
        prediction = np.exp(raw)
        if scale is not None:
            prediction = prediction * scale.reshape((-1,) + (1,) * (raw.ndim - 1))
        return prediction


def make_objective_backend(
    *,
    loss: str,
    quantile: float,
    quantiles: Sequence[float] | None,
    huber_epsilon: float,
    head_alpha: float,
    max_iter: int,
    tol: float,
    tweedie_power: float = 1.5,
) -> ObjectiveBackend:
    if loss == "huber":
        return HuberBackend(
            epsilon=huber_epsilon,
            alpha=head_alpha,
            max_iter=max_iter,
            tol=tol,
        )
    if loss == "quantile":
        return QuantileBackend(quantiles=(float(quantile),), alpha=head_alpha)
    if loss == "multi_quantile":
        if quantiles is None:
            values = (0.1, 0.5, 0.9)
        else:
            values = tuple(float(value) for value in quantiles)
        return QuantileBackend(quantiles=values, alpha=head_alpha)
    if loss == "poisson":
        return PoissonBackend(alpha=head_alpha, max_iter=max_iter, tol=tol)
    if loss == "gamma":
        return PositiveTweedieBackend(
            name="gamma", power=2.0, alpha=head_alpha, max_iter=max_iter, tol=tol
        )
    if loss == "tweedie":
        return PositiveTweedieBackend(
            name="tweedie", power=float(tweedie_power),
            alpha=head_alpha, max_iter=max_iter, tol=tol
        )
    raise ValueError(f"unsupported generalized regression loss: {loss!r}")
