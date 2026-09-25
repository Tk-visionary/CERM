import numpy as np
from sklearn.datasets import load_diabetes, make_friedman1, make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from cerm._internal.cerm_fused_regression import FusedResidualCERMRegressor


def test_fused_large_n_selects_shared_and_improves_friedman():
    X, y = make_friedman1(n_samples=2500, n_features=10, noise=1.0, random_state=11)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=11)
    model = FusedResidualCERMRegressor(random_state=11).fit(Xtr, ytr)
    prediction = model.predict(Xte)
    baseline = model.baseline_.predict(Xte)
    assert model.selected_kind_ == "fused"
    assert model.fit_diagnostics_["threshold_binary_solves"] == 0
    assert r2_score(yte, prediction) > r2_score(yte, baseline) + 0.02


def test_fused_large_n_preserves_linear_noop():
    X, y = make_regression(
        n_samples=2500,
        n_features=12,
        n_informative=8,
        noise=12.0,
        random_state=11,
    )
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=11)
    model = FusedResidualCERMRegressor(random_state=11).fit(Xtr, ytr)
    assert model.selected_kind_ == "current_mean"
    assert np.array_equal(model.predict(Xte), model.baseline_.predict(Xte))


def test_small_n_crossfit_uses_raw_branch_on_diabetes():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=47)
    model = FusedResidualCERMRegressor(random_state=47).fit(Xtr, ytr)
    prediction = model.predict(Xte)
    assert model.fit_diagnostics_["selection_mode"] == "small_n_crossfit_raw"
    assert model.selected_kind_ == "raw_fused"
    assert model.fit_diagnostics_["threshold_binary_solves"] == 0
    assert r2_score(yte, prediction) > 0.46
