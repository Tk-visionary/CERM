from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cerm import CERMClassifier, CompiledProgram, verify_export
from cerm._internal.cerm_hierarchical_residual import HierarchicalResidualCERM
from cerm._internal.cerm_nested_representation import NestedRepresentationMixin


def _small_classifier(**kwargs):
    params = dict(
        preset="balanced",
        max_bins=4,
        max_features=4,
        max_interaction_features=4,
        max_interactions=0,
        resource_policy="ignore",
        random_state=917,
    )
    params.update(kwargs)
    return CERMClassifier(**params)


def _binary_data(seed=0, n=120, d=6):
    return make_classification(
        n_samples=n,
        n_features=d,
        n_informative=min(4, d),
        n_redundant=0,
        random_state=seed,
    )


@pytest.mark.parametrize(
    "labels",
    [
        lambda y: y.astype(bool),
        lambda y: y.astype(np.int64),
        lambda y: np.where(y == 1, "positive", "negative"),
    ],
)
def test_binary_target_label_types(labels):
    X, y = _binary_data(seed=1)
    target = labels(y)
    model = _small_classifier().fit(X, target)
    prediction = model.predict(X[:7])
    assert prediction.shape == (7,)
    assert set(np.unique(prediction)).issubset(set(np.unique(target)))
    assert np.isfinite(model.predict_proba(X[:7])).all()


def test_binary_rare_positive_boundary_and_invalid_targets():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 5))
    y = np.zeros(40, dtype=int)
    y[:4] = 1
    model = _small_classifier().fit(X, y)
    assert model.predict_proba(X[:3]).shape == (3, 2)

    with pytest.raises(ValueError, match="Only binary classification"):
        _small_classifier().fit(X, np.zeros(40, dtype=int))
    too_rare = np.zeros(40, dtype=int)
    too_rare[:3] = 1
    with pytest.raises(ValueError, match="at least 4 samples"):
        _small_classifier().fit(X, too_rare)


def test_dataframe_mixed_missing_unseen_and_schema_contract():
    X_num, y = _binary_data(seed=3, n=100, d=3)
    frame = pd.DataFrame(X_num, columns=["a", "b", "c"])
    frame["all_missing"] = np.nan
    frame["category"] = np.where(np.arange(len(frame)) % 3 == 0, "red", "blue")
    frame.loc[::11, "b"] = np.nan

    model = _small_classifier().fit(frame, y)
    reference = model.predict_proba(frame.iloc[:12])
    reordered = frame[["category", "c", "a", "all_missing", "b"]].iloc[:12]
    assert np.array_equal(reference, model.predict_proba(reordered))

    unseen = frame.iloc[:4].copy()
    unseen["category"] = "unseen-at-predict"
    assert np.isfinite(model.predict_proba(unseen)).all()

    with pytest.raises(ValueError, match=r"missing=\['a'\]"):
        model.predict(frame.drop(columns=["a"]).iloc[:4])
    with pytest.raises(ValueError, match=r"extra=\['extra'\]"):
        model.predict(frame.assign(extra=1.0).iloc[:4])


def test_feature_degeneracy_and_shape_regimes():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 5))
    X[:, 0] = 1.0
    X[:, 1] = (np.arange(len(X)) == 0).astype(float)
    X[:, 4] = X[:, 3]
    y = (rng.normal(size=len(X)) > 0).astype(int)
    model = _small_classifier().fit(X, y)
    assert np.isfinite(model.predict_proba(X[:8])).all()

    X_wide = rng.normal(size=(40, 80))
    y_wide = np.tile([0, 1], 20)
    wide = _small_classifier().fit(X_wide, y_wide)
    assert wide.predict(X_wide[:2]).shape == (2,)

    X_tall = rng.normal(size=(600, 3))
    y_tall = (X_tall[:, 0] + 0.1 * rng.normal(size=600) > 0).astype(int)
    tall = _small_classifier().fit(X_tall, y_tall)
    assert tall.predict_proba(X_tall[:2]).shape == (2, 2)


def test_high_cardinality_categorical_unweighted_and_weighted_rejection():
    rng = np.random.default_rng(5)
    n = 100
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "category": [f"cat-{i % 25}" for i in range(n)],
        }
    )
    y = np.tile([0, 1], n // 2)
    model = _small_classifier(max_identity_categories=8).fit(frame, y)
    assert np.isfinite(model.predict_proba(frame.iloc[:5])).all()

    with pytest.raises(ValueError, match="target-aware typed preprocessing"):
        _small_classifier(max_identity_categories=8).fit(
            frame, y, sample_weight=np.ones(n)
        )

    identity = _small_classifier(
        category_policy="identity", max_identity_categories=8
    ).fit(frame, y, sample_weight=np.ones(n))
    assert identity.predict_proba(frame.iloc[:5]).shape == (5, 2)


def test_zero_weight_rows_are_equivalent_to_removal():
    X, y = _binary_data(seed=6, n=100, d=6)
    rng = np.random.default_rng(7)
    ignored_X = 1e6 * rng.normal(size=(12, X.shape[1]))
    ignored_y = np.tile([0, 1], 6)
    X_augmented = np.vstack([X, ignored_X])
    y_augmented = np.r_[y, ignored_y]
    weights = np.r_[np.ones(len(y)), np.zeros(len(ignored_y))]

    plain = _small_classifier().fit(X, y)
    weighted = _small_classifier().fit(
        X_augmented, y_augmented, sample_weight=weights
    )
    assert np.array_equal(plain.predict_proba(X), weighted.predict_proba(X))
    assert plain.model_.selected_hybrid_config_ == weighted.model_.selected_hybrid_config_


def test_extreme_and_invalid_sample_weights():
    X, y = _binary_data(seed=8, n=120, d=6)
    weight = np.geomspace(1e-4, 1e4, num=len(y))
    model = _small_classifier().fit(X, y, sample_weight=weight)
    assert np.isfinite(model.predict_proba(X[:10])).all()

    bad = [
        np.ones(len(y) - 1),
        np.r_[np.ones(len(y) - 1), -1.0],
        np.r_[np.ones(len(y) - 1), np.nan],
        np.zeros(len(y)),
    ]
    for candidate in bad:
        with pytest.raises(ValueError):
            _small_classifier().fit(X, y, sample_weight=candidate)


def test_semantic_alias_set_params_is_clone_safe():
    model = CERMClassifier(resource_policy="ignore")
    model.set_params(state_detail="coarse")
    assert model.max_bins == 4
    cloned = clone(model)
    assert cloned.max_bins == 4
    assert cloned.state_detail == "coarse"
    assert cloned.get_params(deep=False)["max_bins"] == 4


def test_bulk_set_params_preserves_sklearn_parameter_identity():
    model = CERMClassifier(resource_policy="ignore")
    params = model.get_params(deep=False)
    model.set_params(**params)
    roundtrip = model.get_params(deep=False)
    assert set(roundtrip) == set(params)
    for name, value in roundtrip.items():
        assert value is params[name]


def test_grid_search_and_pipeline_smoke():
    X, y = _binary_data(seed=9, n=80, d=5)
    search = GridSearchCV(
        _small_classifier(max_features=5, max_interaction_features=5),
        {"max_bins": [4, 8]},
        cv=2,
        n_jobs=1,
    ).fit(X, y)
    assert search.best_estimator_.predict(X[:3]).shape == (3,)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", _small_classifier(max_features=5, max_interaction_features=5)),
        ]
    ).fit(X, y)
    assert pipeline.predict_proba(X[:3]).shape == (3, 2)


def test_shared_nested_engine_matches_legacy_code_layout_exactly():
    X, y = _binary_data(seed=10, n=140, d=7)
    model = _small_classifier(
        max_bins=16,
        max_features=6,
        max_interaction_features=6,
        max_interactions=4,
    ).fit(X, y)
    base = model.model_.base_
    assert NestedRepresentationMixin in type(base).__mro__
    states = base.encoder_.transform_columns(X[:32], base.feature_idx_)
    shared_codes = base._build_codes_from_states(states)
    legacy_codes = HierarchicalResidualCERM._build_codes_from_states(base, states)
    assert np.array_equal(shared_codes, legacy_codes)


def test_semantic_optimized_execution_and_batch_contracts():
    X, y = _binary_data(seed=11, n=120, d=6)
    model = _small_classifier(max_features=6, max_interaction_features=6).fit(X, y)
    semantic = model.program_
    optimized = model.optimize("balanced")
    for batch in (X[:1], X[:17]):
        expected = semantic.predict_proba(batch)
        assert np.max(np.abs(expected - optimized.predict_proba(batch))) <= 2e-12
        assert np.max(np.abs(expected - model.predict_proba(batch))) <= 2e-12
        assert model.decision_function(batch).shape == (len(batch),)
        assert model.predict(batch).shape == (len(batch),)

    with pytest.raises(ValueError):
        model.predict(np.empty((0, X.shape[1])))
    with pytest.raises(ValueError, match="expecting"):
        model.predict(X[:, :-1])


def test_estimator_save_load_export_and_model_size(tmp_path):
    X, y = _binary_data(seed=12, n=100, d=5)
    model = _small_classifier(max_features=5, max_interaction_features=5).fit(X, y)
    expected = model.predict_proba(X[:20])

    path = model.save(tmp_path / "model.joblib")
    loaded = CERMClassifier.load(path)
    assert np.array_equal(expected, loaded.predict_proba(X[:20]))

    manifest = model.export(tmp_path / "semantic_package")
    verified = verify_export(manifest)
    assert verified["format"] == "cerm-python-package-v2"
    assert model.model_bytes_estimate_ > 0
    assert model.program_.model_bytes_estimate > 0


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is required for native contract")
def test_native_compile_save_load_parity(tmp_path):
    X, y = _binary_data(seed=13, n=120, d=6)
    model = _small_classifier(max_features=6, max_interaction_features=6).fit(X, y)
    compiled = model.compile_native(tmp_path / "binary_native")
    expected = model.program_.predict_proba(X[:25])
    assert np.max(np.abs(expected - compiled.predict_proba(X[:25]))) <= 2e-12
    assert compiled.predict_proba(X[:1]).shape == (1, 2)

    manifest = compiled.save(tmp_path / "compiled_package")
    verify_export(manifest)
    loaded = CompiledProgram.load(tmp_path / "compiled_package")
    assert np.max(np.abs(expected - loaded.predict_proba(X[:25]))) <= 2e-12
    assert compiled.model_bytes_estimate > 0
