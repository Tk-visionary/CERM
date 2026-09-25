import numpy as np
from scipy import sparse

from cerm._compat.sklearn import fit_ridge_lsqr_alpha_path_exact
from cerm.training_graph import solve_binary_logistic_path
from cerm._internal.cerm_hierarchical_residual import _rank_pairs_newton
from cerm import CERMGeneralizedRegressor


def test_liblinear_c_path_parallel_is_bitwise_equal():
    rng = np.random.default_rng(101)
    X = sparse.random(1200, 80, density=0.08, random_state=3, format="csr")
    y = rng.integers(0, 2, size=1200)
    train, valid = X[:900], X[900:]
    target = y[:900]
    Cs = [0.2, 1.0, 5.0]
    serial = solve_binary_logistic_path(
        train, target, valid, Cs, random_state=77, max_iter=500, n_jobs=1
    )
    parallel = solve_binary_logistic_path(
        train, target, valid, Cs, random_state=77, max_iter=500, n_jobs=3
    )
    for C in Cs:
        a, b = serial[float(C)], parallel[float(C)]
        assert np.array_equal(a.coefficient, b.coefficient)
        assert a.intercept == b.intercept
        assert np.array_equal(a.valid_probability, b.valid_probability)
        assert a.n_iter == b.n_iter


def test_ridge_alpha_path_parallel_is_bitwise_equal():
    rng = np.random.default_rng(102)
    X = sparse.random(1400, 120, density=0.06, random_state=4, format="csr")
    y = rng.normal(size=1400)
    alphas = np.asarray([0.1, 1.0, 10.0])
    serial = fit_ridge_lsqr_alpha_path_exact(X, y, alphas, n_jobs=1)
    parallel = fit_ridge_lsqr_alpha_path_exact(X, y, alphas, n_jobs=3)
    for a, b in zip(serial, parallel):
        assert np.array_equal(a, b)


def test_newton_pair_ranking_parallel_is_identical():
    rng = np.random.default_rng(103)
    states = np.asfortranarray(
        rng.integers(0, 16, size=(50_000, 16), dtype=np.int16)
    )
    y = rng.integers(0, 2, size=len(states))
    serial = _rank_pairs_newton(states, y, 20, feature_limit=16, n_jobs=1)
    parallel = _rank_pairs_newton(states, y, 20, feature_limit=16, n_jobs=4)
    assert serial == parallel


def test_multi_quantile_parallel_heads_are_bitwise_equal():
    rng = np.random.default_rng(104)
    X = rng.normal(size=(180, 8))
    y = 1.4 * X[:, 0] - 0.6 * X[:, 1] + rng.normal(scale=0.5, size=len(X))
    kwargs = dict(
        loss="multi_quantile",
        quantiles=(0.1, 0.5, 0.9),
        representation_mode="linear",
        max_features=8,
        max_interaction_features=4,
        random_state=91,
    )
    serial = CERMGeneralizedRegressor(n_jobs=1, **kwargs).fit(X, y)
    parallel = CERMGeneralizedRegressor(n_jobs=3, **kwargs).fit(X, y)
    assert np.array_equal(serial.predict(X), parallel.predict(X))
    assert np.array_equal(serial.linear_coef_, parallel.linear_coef_)
    assert np.array_equal(np.asarray(serial.intercept_), np.asarray(parallel.intercept_))
    assert np.array_equal(np.asarray(serial.n_iter_), np.asarray(parallel.n_iter_))
