import numpy as np
import pytest
from sklearn import config_context
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cerm import CERMClassifier, CERMGeneralizedRegressor, CERMMultiLabelClassifier


def _small_classifier(**kwargs):
    params = dict(
        preset="balanced",
        max_features=8,
        max_interaction_features=6,
        max_interactions=4,
        random_state=31,
    )
    params.update(kwargs)
    return CERMClassifier(**params)


def test_gamma_and_tweedie_positive_predictions():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(220, 6))
    mean = np.exp(0.35 * X[:, 0] - 0.25 * X[:, 1])
    y = rng.gamma(shape=2.5, scale=mean / 2.5)
    for loss in ("gamma", "tweedie"):
        model = CERMGeneralizedRegressor(
            loss=loss,
            preset="balanced",
            max_features=6,
            max_interaction_features=6,
            max_interactions=4,
            random_state=9,
        ).fit(X, y)
        prediction = model.predict(X[:20])
        assert prediction.shape == (20,)
        assert np.isfinite(prediction).all()
        assert np.all(prediction > 0)


def test_tweedie_power_validation_and_target_contracts():
    X = np.arange(60, dtype=float).reshape(20, 3)
    with pytest.raises(ValueError, match="strictly between 1 and 2"):
        CERMGeneralizedRegressor(loss="tweedie", tweedie_power=2.0).fit(
            X, np.ones(20)
        )
    with pytest.raises(ValueError, match="strictly positive"):
        CERMGeneralizedRegressor(loss="gamma").fit(X, np.r_[np.zeros(1), np.ones(19)])
    with pytest.raises(ValueError, match="non-negative"):
        CERMGeneralizedRegressor(loss="tweedie").fit(X, np.r_[-1.0, np.ones(19)])


def test_gamma_offset_exposure_exact_prediction_scaling():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(180, 5))
    exposure = rng.uniform(0.5, 2.0, size=len(X))
    offset = rng.normal(scale=0.15, size=len(X))
    scale = exposure * np.exp(offset)
    rate = np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1])
    y = scale * rng.gamma(shape=3.0, scale=rate / 3.0)
    model = CERMGeneralizedRegressor(
        loss="gamma",
        preset="balanced",
        max_features=5,
        max_interaction_features=5,
        max_interactions=3,
        random_state=4,
    ).fit(X, y, exposure=exposure, offset=offset)
    base = model.predict(X[:25])
    doubled = model.predict(X[:25], exposure=np.full(25, 2.0))
    unit = model.predict(X[:25], exposure=np.ones(25))
    assert np.allclose(doubled, 2.0 * unit, rtol=0, atol=0)
    assert np.isfinite(base).all()


def test_binary_all_one_weights_are_bitwise_equivalent():
    X, y = make_classification(
        n_samples=180, n_features=8, n_informative=5, random_state=3
    )
    plain = _small_classifier().fit(X, y)
    weighted = _small_classifier().fit(X, y, sample_weight=np.ones(len(y)))
    assert np.array_equal(plain.predict_proba(X), weighted.predict_proba(X))
    assert plain.model_.selected_hybrid_config_ == weighted.model_.selected_hybrid_config_


def test_ovr_and_independent_multilabel_accept_sample_weight():
    X, y = make_classification(
        n_samples=240,
        n_features=9,
        n_informative=6,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=5,
    )
    weight = np.where(y == 2, 2.5, 1.0)
    multiclass = _small_classifier(max_features=9).fit(X, y, sample_weight=weight)
    assert multiclass.predict_proba(X[:8]).shape == (8, 3)
    labels = np.column_stack([y == 0, y == 1]).astype(int)
    multilabel = CERMMultiLabelClassifier(
        estimator=_small_classifier(max_features=9),
        representation_strategy="independent",
    ).fit(X, labels, sample_weight=weight)
    assert multilabel.predict_proba(X[:8]).shape == (8, 2)


def test_shared_weighted_paths_fail_explicitly():
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=12,
    )
    with pytest.raises(ValueError, match="not shared multiclass"):
        _small_classifier(multiclass_strategy="shared").fit(
            X, y, sample_weight=np.ones(len(y))
        )
    labels = np.column_stack([y == 0, y == 1]).astype(int)
    with pytest.raises(ValueError, match="representation_strategy='independent'"):
        CERMMultiLabelClassifier(
            estimator=_small_classifier(), representation_strategy="shared"
        ).fit(X, labels, sample_weight=np.ones(len(y)))


def test_classifier_metadata_routing_through_pipeline():
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=4, random_state=15
    )
    weight = np.where(y == 1, 2.0, 1.0)
    with config_context(enable_metadata_routing=True):
        scaler = StandardScaler().set_fit_request(sample_weight=False)
        estimator = _small_classifier(
            max_features=6, max_interaction_features=4, max_interactions=2
        ).set_fit_request(sample_weight=True)
        pipeline = Pipeline([("scale", scaler), ("model", estimator)])
        pipeline.fit(X, y, sample_weight=weight)
        assert pipeline.predict(X[:5]).shape == (5,)
