import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.datasets import make_friedman1, make_regression

from cerm._internal.cerm_fused_regression import (
    FusedResidualCERMRegressor,
    FusedThresholdHead,
)


def _fast_large_n_model(seed=17):
    return FusedResidualCERMRegressor(
        n_bins=6,
        max_features=6,
        max_interaction_features=4,
        max_pairs=2,
        max_fine_pairs=0,
        max_bins=8,
        spline_knots=3,
        small_n_threshold=0,
        raw_max_pairs=1,
        random_state=seed,
        max_iter=80,
    )


def test_none_and_uniform_weights_are_exactly_equivalent():
    X, y = make_friedman1(
        n_samples=160,
        n_features=6,
        noise=0.8,
        random_state=17,
    )
    reference = _fast_large_n_model(17).fit(X, y)
    explicit_none = _fast_large_n_model(17).fit(X, y, sample_weight=None)
    uniform = _fast_large_n_model(17).fit(X, y, sample_weight=np.ones(len(y)))

    reference_prediction = reference.predict(X)
    assert np.array_equal(reference_prediction, explicit_none.predict(X))
    assert np.array_equal(reference_prediction, uniform.predict(X))
    assert reference.selected_kind_ == explicit_none.selected_kind_ == uniform.selected_kind_
    assert uniform.fit_diagnostics_["sample_weighted"] is True


def test_zero_weight_rows_do_not_change_fit_or_prediction():
    X, y = make_regression(
        n_samples=140,
        n_features=6,
        n_informative=5,
        noise=5.0,
        random_state=23,
    )
    rng = np.random.default_rng(23)
    X_ignored = rng.normal(loc=1e5, scale=1e3, size=(5, X.shape[1]))
    y_ignored = np.linspace(-1e8, 1e8, len(X_ignored))
    X_augmented = np.vstack([X, X_ignored])
    y_augmented = np.r_[y, y_ignored]
    weights = np.r_[np.ones(len(y)), np.zeros(len(y_ignored))]

    reference = _fast_large_n_model(23).fit(X, y)
    weighted = _fast_large_n_model(23).fit(
        X_augmented,
        y_augmented,
        sample_weight=weights,
    )

    assert np.array_equal(reference.predict(X), weighted.predict(X))
    assert reference.selected_kind_ == weighted.selected_kind_
    assert weighted.fit_diagnostics_["input_samples"] == len(y_augmented)
    assert weighted.fit_diagnostics_["effective_samples"] == len(y)


def test_integer_weights_match_row_duplication_for_fused_threshold_objective():
    rng = np.random.default_rng(31)
    design = sparse.csr_matrix(rng.normal(size=(28, 5)))
    thresholds = np.asarray([-0.6, 0.0, 0.7])
    latent_target = rng.normal(size=design.shape[0])
    labels = (latent_target[:, None] > thresholds[None, :]).astype(float)
    weights = rng.integers(1, 4, size=len(labels)).astype(float)

    weighted = FusedThresholdHead(C=0.04, spline_knots=3, max_iter=200).fit(
        design,
        labels,
        thresholds,
        -1.5,
        1.5,
        sample_weight=weights,
    )
    repeated_rows = np.repeat(np.arange(len(labels)), weights.astype(int))
    duplicated = FusedThresholdHead(C=0.04, spline_knots=3, max_iter=200).fit(
        design[repeated_rows],
        labels[repeated_rows],
        thresholds,
        -1.5,
        1.5,
    )

    probe = sparse.csr_matrix(rng.normal(size=(17, design.shape[1])))
    assert np.allclose(
        weighted.predict_survival(probe),
        duplicated.predict_survival(probe),
        rtol=0.0,
        atol=1e-12,
    )


def test_imbalanced_weights_produce_finite_predictions():
    X, y = make_friedman1(
        n_samples=180,
        n_features=6,
        noise=1.0,
        random_state=37,
    )
    weights = np.geomspace(1e-2, 1e2, len(y))
    model = _fast_large_n_model(37).fit(X, y, sample_weight=weights)
    prediction = model.predict(X[:25])

    assert np.isfinite(prediction).all()
    assert np.isfinite(model.baseline_validation_rmse_)
    assert np.isfinite(model.selected_validation_rmse_)
    assert model.fit_diagnostics_["sample_weight_sum"] == pytest.approx(weights.sum())


def test_mixed_pandas_categorical_and_missing_reuses_regression_adapter():
    rng = np.random.default_rng(41)
    n = 150
    frame = pd.DataFrame(
        {
            "numeric": rng.normal(size=n),
            "category": np.asarray(["a", "b", "c", "d", "e"])[
                rng.integers(0, 5, size=n)
            ],
            "with_missing": rng.normal(size=n),
        }
    )
    frame.loc[::11, "with_missing"] = np.nan
    category_effect = frame["category"].map(
        {"a": -2.0, "b": -1.0, "c": 0.0, "d": 1.0, "e": 2.0}
    ).to_numpy()
    y = 2.0 * frame["numeric"].to_numpy() + category_effect + rng.normal(
        scale=0.2, size=n
    )
    weights = np.linspace(0.5, 2.0, n)

    model = _fast_large_n_model(41)
    model.max_identity_categories = 3
    model.fit(frame, y, sample_weight=weights)

    assert model.adapter_ is not None
    assert model.fit_diagnostics_["typed_adapter"] is True
    assert model.n_features_in_ == frame.shape[1]
    assert any(kind == "categorical_quotient" for kind in model._core_feature_kinds_)
    assert any(kind == "missing_state" for kind in model._core_feature_kinds_)
    prediction = model.predict(frame.iloc[:20])
    reordered = model.predict(frame.iloc[:20][list(reversed(frame.columns))])
    assert np.isfinite(prediction).all()
    assert np.array_equal(prediction, reordered)


def test_weighted_small_n_crossfit_uses_weighted_gate_without_failure():
    X, y = make_friedman1(
        n_samples=90,
        n_features=6,
        noise=1.0,
        random_state=43,
    )
    weights = np.linspace(0.25, 3.0, len(y))
    model = FusedResidualCERMRegressor(
        n_bins=6,
        max_features=6,
        max_interaction_features=4,
        max_pairs=1,
        max_bins=8,
        spline_knots=3,
        small_n_threshold=1000,
        small_n_folds=3,
        raw_max_pairs=1,
        random_state=43,
        max_iter=60,
    ).fit(X, y, sample_weight=weights)

    assert model.fit_diagnostics_["selection_mode"] == "small_n_crossfit_raw"
    assert np.isfinite(model.fit_diagnostics_["baseline_validation_rmse"])
    assert np.isfinite(model.fit_diagnostics_["selected_validation_rmse"])
    assert np.isfinite(np.asarray(model.fit_diagnostics_["fold_gammas"])).all()
    assert np.isfinite(model.predict(X[:12])).all()


@pytest.mark.parametrize(
    "sample_weight, match",
    [
        (np.ones(19), "inconsistent lengths"),
        (np.r_[np.ones(19), -1.0], "negative"),
        (np.r_[np.ones(19), np.nan], "NaN or infinity"),
        (np.zeros(20), "all zero"),
    ],
)
def test_invalid_sample_weight_is_rejected(sample_weight, match):
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.linspace(-1.0, 1.0, len(X))
    with pytest.raises(ValueError, match=match):
        FusedResidualCERMRegressor().fit(X, y, sample_weight=sample_weight)
