from __future__ import annotations

"""Public regression estimator identities.

The ordinary top-level ``CERMRegressor`` now uses the fused residual V2
procedure.  The historical finite-state Ridge estimator remains available as
``CERMRidgeRegressor``.  Internal modules may continue importing the historical
class from ``cerm.regression`` where it is used as a stable baseline component.
"""

from .fused_regression import CERMFusedRegressor
from .regression import CERMRegressor as _HistoricalCERMRegressor


class CERMRegressor(CERMFusedRegressor):
    """Default CERM regressor using fused residual V2.

    Fused V2 contains its own finite-state mean baseline and promotes nonlinear
    residual correction only when its fitted selection gate chooses it.  The
    constructor therefore keeps the validated fused capacity surface rather
    than reusing historical Ridge-only parameters.
    """


class CERMRidgeRegressor(_HistoricalCERMRegressor):
    """Historical finite-state Ridge regression estimator.

    This class preserves the previous public ``CERMRegressor`` implementation,
    constructor semantics, semantic export, persistence, and native runtime for
    explicit compatibility and controlled comparisons.
    """


__all__ = ["CERMRegressor", "CERMRidgeRegressor"]
