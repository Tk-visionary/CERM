from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, make_classification

from cerm import CERMClassifier


def small_numeric():
    X, y = make_classification(
        n_samples=140,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=13,
    )
    return X, np.where(y == 1, "yes", "no")


def test_sklearn_clone_and_string_labels():
    X, y = small_numeric()
    model = CERMClassifier(
        max_features=6,
        pair_feature_limit=6,
        random_state=13,
    )
    cloned = clone(model)
    assert cloned.get_params()["max_features"] == 6
    model.fit(X, y)
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert set(model.predict(X[:10])) <= {"yes", "no"}
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_dataframe_typed_schema_and_feature_names():
    n = 150
    frame = pd.DataFrame(
        {
            "x": np.linspace(-2, 2, n),
            "category": [f"c{i % 18}" for i in range(n)],
            "missing": np.where(np.arange(n) % 7 == 0, np.nan, np.arange(n) / n),
        }
    )
    y = np.where(frame["x"].to_numpy() + (np.arange(n) % 3 == 0) > 0, "up", "down")
    model = CERMClassifier(
        max_features=10,
        pair_feature_limit=8,
        categorical_features="auto",
        max_identity_categories=8,
        category_bins=4,
        random_state=17,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(frame, y)
        p1 = model.predict_proba(frame.iloc[:20])
        p2 = model.predict_proba(frame[["missing", "category", "x"]].iloc[:20])
    assert not [w for w in caught if "unknown categories" in str(w.message)]
    np.testing.assert_allclose(p1, p2, atol=0.0, rtol=0.0)
    assert tuple(model.feature_names_in_) == tuple(frame.columns)
    assert len(model.get_feature_names_out()) == model.n_adapted_features_
    assert model.fit_diagnostics_.adapted_features == model.n_adapted_features_


def test_save_load_and_export(tmp_path):
    X, y = small_numeric()
    model = CERMClassifier(
        max_features=6,
        pair_feature_limit=6,
        random_state=19,
    ).fit(X, y)
    path = model.save(tmp_path / "model.joblib")
    restored = CERMClassifier.load(path)
    np.testing.assert_allclose(
        model.predict_proba(X[:25]),
        restored.predict_proba(X[:25]),
        atol=0.0,
        rtol=0.0,
    )
    manifest_path = model.export(tmp_path / "package")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "cerm-python-package-v2"
    assert len(manifest["model"]["npz"]["sha256"]) == 64
    assert manifest["model"]["npz"]["bytes"] > 0
    assert manifest["library_version"] == model.VERSION
    assert manifest["classes"] == ["no", "yes"]
    assert manifest["positive_class"] == "yes"


def test_semantic_and_optimized_program_are_equivalent():
    X, y = small_numeric()
    model = CERMClassifier(
        max_features=6,
        pair_feature_limit=6,
        random_state=23,
    ).fit(X, y)
    optimized = model.optimize("balanced")
    np.testing.assert_allclose(
        model.predict_proba(X[:40]),
        optimized.predict_proba(X[:40]),
        atol=2e-12,
        rtol=0.0,
    )


def test_native_compile_equivalence(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:220], y[:220]
    model = CERMClassifier(
        max_features=12,
        pair_feature_limit=10,
        random_state=29,
    ).fit(X, y)
    compiled = model.compile_native(tmp_path / "native")
    np.testing.assert_allclose(
        model.predict_proba(X[:60]),
        compiled.predict_proba(X[:60]),
        atol=2e-12,
        rtol=0.0,
    )


def test_numeric_dataframe_reorders_columns_and_rejects_schema_changes():
    frame = pd.DataFrame(
        {
            "a": np.linspace(-1, 1, 120),
            "b": np.sin(np.linspace(0, 4, 120)),
            "c": np.cos(np.linspace(0, 3, 120)),
        }
    )
    y = np.where(frame["a"] + frame["b"] > 0, "p", "n")
    model = CERMClassifier(
        max_features=3,
        pair_feature_limit=3,
        categorical_features=(),
        random_state=41,
    ).fit(frame, y)
    expected = model.predict_proba(frame.iloc[:20])
    actual = model.predict_proba(frame[["c", "a", "b"]].iloc[:20])
    np.testing.assert_allclose(expected, actual, atol=0.0, rtol=0.0)
    with np.testing.assert_raises(ValueError):
        model.predict_proba(frame.assign(extra=1).iloc[:20])


def test_typed_direct_states_optimize_and_compile(tmp_path):
    n = 160
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1.5, 1.5, n),
            "cat": [f"k{i % 5}" for i in range(n)],
        }
    )
    y = np.where(frame["x"].to_numpy() + (np.arange(n) % 5 == 4) * 0.8 > 0, "yes", "no")
    model = CERMClassifier(
        max_features=6,
        pair_feature_limit=5,
        categorical_features="auto",
        category_policy="identity",
        random_state=43,
    ).fit(frame, y)
    optimized = model.optimize("balanced")
    np.testing.assert_allclose(
        model.predict_proba(frame.iloc[:50]),
        optimized.predict_proba(frame.iloc[:50]),
        atol=2e-12,
        rtol=0.0,
    )
    compiled = optimized.compile(tmp_path / "typed_graph")
    np.testing.assert_allclose(
        model.predict_proba(frame.iloc[:50]),
        compiled.predict_proba(frame.iloc[:50]),
        atol=2e-12,
        rtol=0.0,
    )


def test_practical_search_profile_exposes_reduced_training_graph():
    X, y = load_breast_cancer(return_X_y=True)
    model = CERMClassifier(
        max_features=24,
        pair_feature_limit=20,
        search_profile="practical",
        random_state=101,
        resource_policy="ignore",
    ).fit(X, y)
    diagnostics = model.fit_diagnostics_
    assert diagnostics.search_profile == "practical"
    assert diagnostics.requested_pair_feature_limit == 20
    assert diagnostics.effective_pair_feature_limit == 20
    assert diagnostics.training_graph_candidate_count <= 9
    assert diagnostics.training_graph_design_count <= 7
