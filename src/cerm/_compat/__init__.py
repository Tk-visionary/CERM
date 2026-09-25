"""Compatibility helpers for optional or version-sensitive dependencies.

Private dependency APIs must be imported only from modules in this package.
The rest of CERM should use the stable wrappers exposed here.
"""

from .sklearn import (
    fit_ridge_lsqr_alpha_path_exact,
    num_samples,
    safe_indexing,
    train_binary_liblinear,
)

__all__ = [
    "fit_ridge_lsqr_alpha_path_exact",
    "num_samples",
    "safe_indexing",
    "train_binary_liblinear",
]
