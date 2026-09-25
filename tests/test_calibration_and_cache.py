from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, roc_auc_score

from cerm import CERMClassifier, NewtonHistogramCache
from cerm.calibration import fit_affine_calibrator


def test_affine_calibrator_improves_overconfident_logits():
    rng = np.random.default_rng(211)
    latent = rng.normal(size=800)
    true_probability = 1.0 / (1.0 + np.exp(-latent))
    y = rng.binomial(1, true_probability)
    overconfident_logits = 2.7 * latent + 0.35
    result = fit_affine_calibrator(
        overconfident_logits,
        y,
        method="affine",
        l2=1e-4,
        min_improvement=1e-4,
        folds=5,
    )
    assert result.accepted
    assert 0 < result.scale < 1.0
    assert result.calibrated_logloss < result.raw_logloss
    assert result.improvement > 0


def test_estimator_cross_fitted_calibration_preserves_ranking_and_program_exactness():
    X, y = make_classification(
        n_samples=180,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        class_sep=0.8,
        flip_y=0.08,
        random_state=223,
    )
    common = dict(
        max_features=7,
        pair_feature_limit=7,
        random_state=223,
        prediction_backend="optimized",
    )
    raw = CERMClassifier(**common).fit(X, y)
    calibrated = CERMClassifier(
        **common,
        calibration="affine",
        calibration_folds=3,
        calibration_l2=1e-3,
        calibration_min_improvement=0.0,
    ).fit(X, y)

    raw_score = raw.decision_function(X)
    calibrated_score = calibrated.decision_function(X)
    result = calibrated.calibration_result_
    np.testing.assert_allclose(
        calibrated_score,
        result.scale * raw_score + result.offset,
        atol=2e-10,
        rtol=0,
    )
    assert result.scale > 0
    assert roc_auc_score(y, raw_score) == roc_auc_score(y, calibrated_score)
    np.testing.assert_allclose(
        calibrated.program_.predict_proba(X),
        calibrated.optimize("balanced").predict_proba(X),
        atol=2e-12,
        rtol=0,
    )


def test_newton_histogram_cache_roundtrip_and_exact_ranking(tmp_path):
    X, y = make_classification(
        n_samples=220,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=227,
    )
    model = CERMClassifier(
        max_features=8,
        pair_feature_limit=8,
        random_state=227,
    ).fit(X, y)
    cache = model.build_newton_cache(
        X,
        y,
        prediction="base",
        feature_limit=8,
    )
    rankings = cache.sweep_pair_rankings([0.5, 5.0, 20.0], max_pairs=6)
    assert set(rankings) == {0.5, 5.0, 20.0}
    assert all(len(value) == 6 for value in rankings.values())
    assert cache.nbytes > 0
    assert model.training_cache_ is cache


    from cerm._internal.cerm_hierarchical_residual import _rank_pairs_newton

    state_matrix = cache.states.array[:, : cache.feature_limit]
    expected_pairs = _rank_pairs_newton(
        state_matrix,
        model._label_encoder_.transform(y),
        6,
        feature_limit=cache.feature_limit,
        l2=5.0,
        p=model.model_.base_.predict_proba(model.program_._matrix(X))[:, 1],
    )
    actual_pairs = [
        (row["left"], row["right"]) for row in cache.rank_pairs(6, 5.0)
    ]
    assert actual_pairs == expected_pairs

    path = cache.save(tmp_path / "cache.npz")
    restored = NewtonHistogramCache.load(path)
    assert restored.feature_names == cache.feature_names
    for l2 in (0.5, 5.0, 20.0):
        assert restored.rank_pairs(6, l2) == cache.rank_pairs(6, l2)

    model.clear_training_cache()
    assert not hasattr(model, "training_cache_")


def test_fit_time_training_cache_is_optional_and_reusable():
    X, y = make_classification(
        n_samples=180,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        random_state=229,
    )
    model = CERMClassifier(
        max_features=7,
        pair_feature_limit=7,
        random_state=229,
        cache_training_statistics=True,
    ).fit(X, y)
    assert hasattr(model, "training_cache_")
    assert model.training_cache_.nbytes > 0
    assert model.fit_diagnostics_.training_cache_bytes == model.training_cache_.nbytes
    first = model.training_cache_.rank_pairs(5, 5.0)
    second = model.build_newton_cache(
        X, y, prediction="base", feature_limit=7, store=False
    ).rank_pairs(5, 5.0)
    assert first == second


def test_compiled_program_can_be_saved_and_loaded_without_recompilation(tmp_path):
    from cerm import CompiledProgram

    X, y = make_classification(
        n_samples=180,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        random_state=233,
    )
    labels = np.where(y == 1, "positive", "negative")
    model = CERMClassifier(
        max_features=7,
        pair_feature_limit=7,
        random_state=233,
    ).fit(X, labels)
    compiled = model.optimize("balanced").compile(
        tmp_path / "compiled_source",
        dtype="float64",
    )
    expected_probability = compiled.predict_proba(X[:40])
    expected_label = compiled.predict(X[:40])
    manifest = compiled.save(tmp_path / "artifact", include_source=True)
    restored = CompiledProgram.load(manifest)
    np.testing.assert_allclose(
        restored.predict_proba(X[:40]),
        expected_probability,
        atol=0,
        rtol=0,
    )
    np.testing.assert_array_equal(restored.predict(X[:40]), expected_label)
    assert restored.metadata["reloaded"] is True
