from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_friedman1, make_multilabel_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from cerm import (
    CERMClassifier,
    CERMMultiLabelClassifier,
    CERMRidgeRegressor as CERMRegressor,
    ProgramBundle,
    verify_export,
)


def _small_classifier(**kwargs):
    params = dict(
        preset="balanced",
        max_features=6,
        max_interaction_features=6,
        max_interactions=4,
        pair_feature_limit=6,
        selection_strategy="two_holdout",
        random_state=17,
    )
    params.update(kwargs)
    return CERMClassifier(**params)


def test_multiclass_ovr_probability_contract_and_accuracy(tmp_path):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=3, stratify=y
    )
    model = _small_classifier().fit(X_train, y_train)
    probability = model.predict_proba(X_test)
    assert model.task_type_ == "multiclass"
    assert isinstance(model.program_, ProgramBundle)
    assert probability.shape == (len(X_test), 3)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-12)
    assert accuracy_score(y_test, model.predict(X_test)) >= 0.80

    manifest = model.export(tmp_path / "multiclass")
    payload = json.loads(manifest.read_text())
    assert payload["format"] in {"cerm-program-bundle-v1", "cerm-program-bundle-v2"}
    assert payload["task_type"] == "multiclass"
    assert payload["n_outputs"] == 3
    assert verify_export(manifest)["task_type"] == "multiclass"


def test_multilabel_independent_heads_and_constant_label(tmp_path):
    X, y = make_multilabel_classification(
        n_samples=90,
        n_features=8,
        n_classes=3,
        n_labels=2,
        random_state=4,
        allow_unlabeled=True,
    )
    y[:, 2] = 1
    estimator = _small_classifier(max_features=8, max_interaction_features=8)
    model = CERMMultiLabelClassifier(estimator=estimator, threshold=0.45).fit(X, y)
    probability = model.predict_proba(X[:7])
    prediction = model.predict(X[:7])
    assert probability.shape == (7, 3)
    assert prediction.shape == (7, 3)
    assert np.all(prediction[:, 2] == 1)
    assert model.fit_diagnostics_["constant_outputs"] == 1

    manifest = model.export(tmp_path / "multilabel")
    payload = json.loads(manifest.read_text())
    assert payload["task_type"] == "multilabel"
    assert payload["n_outputs"] == 3
    assert verify_export(manifest)["task_type"] == "multilabel"


def test_regression_selects_linear_or_finite_state_structure():
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=8,
        noise=5.0,
        random_state=5,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    linear_model = CERMRegressor(
        preset="balanced",
        max_features=10,
        max_interactions=6,
        random_state=7,
    ).fit(X_train, y_train)
    assert r2_score(y_test, linear_model.predict(X_test)) > 0.95

    X, y = make_friedman1(n_samples=600, n_features=10, noise=1.0, random_state=6)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    nonlinear_model = CERMRegressor(
        preset="balanced",
        max_features=10,
        max_interactions=8,
        random_state=7,
    ).fit(X_train, y_train)
    assert r2_score(y_test, nonlinear_model.predict(X_test)) > 0.75
    assert nonlinear_model.fit_diagnostics_["selected_config"]["max_main_level"] in {0, 4, 8, 16}


def test_regression_dataframe_categories_export_and_roundtrip(tmp_path):
    rng = np.random.default_rng(11)
    n = 120
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "category": np.asarray([f"c{i % 25}" for i in range(n)], dtype=object),
        }
    )
    y = 2.5 * frame["x"].to_numpy() + np.asarray([int(value[1:]) % 5 for value in frame["category"]])
    model = CERMRegressor(
        max_features=8,
        max_interactions=4,
        category_policy="auto",
        random_state=11,
    ).fit(frame, y)
    before = model.predict(frame.iloc[:10])
    path = model.save(tmp_path / "regressor.joblib")
    restored = CERMRegressor.load(path)
    np.testing.assert_allclose(restored.predict(frame.iloc[:10]), before)

    manifest = model.export(tmp_path / "regression_export")
    payload = json.loads(manifest.read_text())
    assert payload["format"] == "cerm-python-package-v3"
    assert payload["task_type"] == "regression"
    assert payload["n_outputs"] == 1
    assert payload["adapter"]["portable"] is True
    assert payload["adapter"]["format"] == "cerm-regression-adapter-ir-v1"
    assert not (tmp_path / "regression_export" / "adapter.joblib").exists()
    assert verify_export(manifest)["task_type"] == "regression"


def test_multiclass_bundle_evaluates_each_head_once():
    class CountingHead:
        def __init__(self, offset):
            self.offset = offset
            self.calls = 0

        def decision_function(self, X):
            self.calls += 1
            return np.asarray(X)[:, 0] + self.offset

        def predict_proba(self, X):
            raise AssertionError("multiclass path must not request probabilities")

    heads = tuple(CountingHead(i) for i in range(3))
    bundle = ProgramBundle(programs=heads, task_type="multiclass", classes=(0, 1, 2))
    probability = bundle.predict_proba(np.arange(5.0).reshape(-1, 1))
    assert probability.shape == (5, 3)
    assert [head.calls for head in heads] == [1, 1, 1]


def test_multilabel_auto_thresholds_are_per_output():
    X, y = make_multilabel_classification(
        n_samples=160, n_features=10, n_classes=4, n_labels=2, random_state=19
    )
    estimator = _small_classifier(max_features=10, max_interaction_features=8)
    model = CERMMultiLabelClassifier(estimator=estimator, threshold="auto").fit(X, y)
    assert model.thresholds_.shape == (4,)
    assert np.all((model.thresholds_ > 0.0) & (model.thresholds_ < 1.0))
    np.testing.assert_allclose(np.asarray(model.program_.threshold), model.thresholds_)


def test_regression_native_compiler_numeric_and_roundtrip(tmp_path):
    from cerm import CompiledRegressionProgram

    X, y = make_friedman1(n_samples=240, n_features=8, noise=0.5, random_state=23)
    model = CERMRegressor(
        preset="balanced",
        max_features=8,
        max_interactions=6,
        random_state=23,
    ).fit(X, y)
    compiled = model.compile_native(tmp_path / "regression_native")
    expected = model.predict(X[:37])
    np.testing.assert_allclose(compiled.predict(X[:37]), expected, rtol=1e-12, atol=1e-12)

    manifest = compiled.save(tmp_path / "compiled_regression", include_source=True)
    restored = CompiledRegressionProgram.load(manifest)
    np.testing.assert_allclose(restored.predict(X[:37]), expected, rtol=1e-12, atol=1e-12)


def test_regression_native_compiler_with_dataframe_adapter(tmp_path):
    rng = np.random.default_rng(29)
    n = 180
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "group": np.asarray([f"g{i % 7}" for i in range(n)], dtype=object),
        }
    )
    y = 1.7 * frame["x"].to_numpy() + np.asarray([int(value[1:]) for value in frame["group"]])
    model = CERMRegressor(
        category_policy="identity",
        max_features=8,
        max_interactions=4,
        random_state=29,
    ).fit(frame, y)
    compiled = model.compile_native(tmp_path / "regression_frame_native")
    np.testing.assert_allclose(
        compiled.predict(frame.iloc[:41]), model.predict(frame.iloc[:41]), rtol=1e-12, atol=1e-12
    )


def test_multiclass_and_multilabel_share_target_neutral_preprocessing():
    X, y = load_iris(return_X_y=True)
    classifier = _small_classifier().fit(X, y)
    assert classifier.fit_diagnostics_["shared_preprocessing"] is True
    programs = classifier.program_.programs
    assert all(program.adapter is programs[0].adapter for program in programs)

    X_multi, y_multi = make_multilabel_classification(
        n_samples=100, n_features=7, n_classes=3, n_labels=2, random_state=41
    )
    multilabel = CERMMultiLabelClassifier(estimator=_small_classifier()).fit(X_multi, y_multi)
    assert multilabel.fit_diagnostics_["shared_preprocessing"] is True
    active = [program for program in multilabel.program_.programs if hasattr(program, "adapter")]
    assert all(program.adapter is active[0].adapter for program in active)


def test_target_aware_categories_keep_independent_head_preprocessing():
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 90),
            "category": [f"category_{index}" for index in range(90)],
        }
    )
    y = np.asarray([0, 1, 2] * 30)
    model = _small_classifier(
        categorical_features=["category"],
        category_policy="auto",
        max_identity_categories=4,
    ).fit(frame, y)
    assert model.fit_diagnostics_["shared_preprocessing"] is False
    adapters = [program.adapter for program in model.program_.programs]
    assert len({id(adapter) for adapter in adapters}) == 3


def test_shared_adapter_is_exported_once_for_multiclass(tmp_path):
    rng = np.random.default_rng(53)
    n = 150
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "category": np.asarray([f"c{i % 6}" for i in range(n)], dtype=object),
        }
    )
    y = ((frame["x"].to_numpy() > 0).astype(int) + np.arange(n) % 3) % 3
    model = _small_classifier(
        categorical_features=["category"], category_policy="identity"
    ).fit(frame, y)
    manifest = model.export(tmp_path / "shared_bundle")
    payload = json.loads(manifest.read_text())
    assert payload["format"] == "cerm-program-bundle-v2"
    assert payload["shared_adapter"] is not None
    assert all(head["uses_shared_adapter"] for head in payload["heads"])
    assert not list((tmp_path / "shared_bundle").glob("head_*/adapter.*"))
    verify_export(manifest)


def test_regression_pair_predictions_are_batch_invariant():
    from sklearn.datasets import make_friedman1

    X, y = make_friedman1(n_samples=180, n_features=8, noise=0.2, random_state=19)
    model = CERMRegressor(
        preset="accurate",
        max_interaction_features=8,
        max_interactions=8,
        random_state=19,
    ).fit(X, y)
    batch = model.predict(X[:24])
    individual = np.asarray([model.predict(X[index:index + 1])[0] for index in range(24)])
    np.testing.assert_allclose(batch, individual, rtol=0.0, atol=1e-12)


def test_portable_regression_export_roundtrip_numeric_and_pair_batch_invariance(tmp_path):
    from sklearn.datasets import make_friedman1
    from cerm import PortableRegressionProgram

    X, y = make_friedman1(n_samples=260, n_features=8, noise=0.15, random_state=29)
    model = CERMRegressor(
        max_interaction_features=8,
        max_interactions=8,
        include_linear=False,
        random_state=29,
    ).fit(X, y)
    manifest = model.export(tmp_path / "portable_numeric")
    portable = PortableRegressionProgram.load(manifest.parent)
    expected = model.predict(X[:37])
    np.testing.assert_allclose(portable.predict(X[:37]), expected, rtol=0.0, atol=1e-12)
    individual = np.asarray([portable.predict(X[index:index + 1])[0] for index in range(37)])
    np.testing.assert_allclose(individual, expected, rtol=0.0, atol=1e-12)


def test_portable_regression_export_roundtrip_dataframe_identity_and_quotient(tmp_path):
    from cerm import PortableRegressionProgram

    rng = np.random.default_rng(41)
    n = 180
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "small": rng.choice(["a", "b", "c"], size=n),
            "large": [f"category-{value}" for value in rng.integers(0, 30, size=n)],
        }
    )
    frame.loc[::17, "x"] = np.nan
    target = (
        frame["x"].fillna(0).to_numpy() * 1.8
        + (frame["small"] == "b").to_numpy() * 2.0
        + np.asarray([int(value.split("-")[1]) % 5 for value in frame["large"]])
        + rng.normal(scale=0.1, size=n)
    )
    model = CERMRegressor(
        categorical_features=["small", "large"],
        category_policy="auto",
        max_identity_categories=5,
        random_state=41,
    ).fit(frame, target)
    manifest = model.export(tmp_path / "portable_frame")
    portable = PortableRegressionProgram.load(manifest.parent)
    np.testing.assert_allclose(
        portable.predict(frame.iloc[:40]),
        model.predict(frame.iloc[:40]),
        rtol=0.0,
        atol=1e-12,
    )


def test_compiled_regression_v2_save_is_pickle_free_and_dataframe_portable(tmp_path):
    from cerm import CompiledRegressionProgram, verify_export

    rng = np.random.default_rng(53)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=160),
            "category": rng.choice(["red", "green", "blue"], size=160),
        }
    )
    frame.loc[::19, "x"] = np.nan
    y = frame["x"].fillna(0).to_numpy() * 2 + (frame["category"] == "green").to_numpy()
    model = CERMRegressor(
        categorical_features=["category"],
        category_policy="identity",
        random_state=53,
    ).fit(frame, y)
    compiled = model.compile_native(tmp_path / "compiled_source")
    manifest_path = compiled.save(tmp_path / "compiled_saved", include_source=True)
    manifest = verify_export(manifest_path)
    assert manifest["format"] == "cerm-compiled-regression-v2"
    assert not list(manifest_path.parent.glob("*.joblib"))
    reloaded = CompiledRegressionProgram.load(manifest_path.parent)
    np.testing.assert_allclose(
        reloaded.predict(frame.iloc[:35]),
        model.predict(frame.iloc[:35]),
        rtol=0.0,
        atol=1e-12,
    )


def test_compiled_binary_v2_is_pickle_free_for_portable_typed_adapter(tmp_path):
    from cerm import CompiledProgram, verify_export

    rng = np.random.default_rng(67)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=180),
            "category": [f"category-{value}" for value in rng.integers(0, 24, size=180)],
        }
    )
    frame.loc[::23, "x"] = np.nan
    y = (
        frame["x"].fillna(0).to_numpy()
        + np.asarray([int(value.split("-")[1]) < 8 for value in frame["category"]])
        > 0.5
    ).astype(int)
    model = CERMClassifier(
        categorical_features=["category"],
        category_policy="ordered",
        random_state=67,
    ).fit(frame, y)
    compiled = model.compile_native(tmp_path / "binary_source")
    manifest_path = compiled.save(tmp_path / "binary_saved", include_source=True)
    manifest = verify_export(manifest_path)
    assert manifest["format"] == "cerm-compiled-program-v2"
    assert not list(manifest_path.parent.glob("*.joblib"))
    restored = CompiledProgram.load(manifest_path.parent)
    np.testing.assert_allclose(
        restored.predict_proba(frame.iloc[:40]),
        model.predict_proba(frame.iloc[:40]),
        rtol=0.0,
        atol=1e-12,
    )


def test_compiled_multiclass_bundle_v2_deduplicates_shared_adapter(tmp_path):
    from cerm import CompiledProgramBundle, verify_export

    rng = np.random.default_rng(71)
    n = 210
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "category": rng.choice(["a", "b", "c", "d"], size=n),
        }
    )
    y = np.where(
        frame["x"].to_numpy() > 0.7,
        "high",
        np.where(frame["category"].isin(["a", "b"]), "group", "base"),
    )
    model = CERMClassifier(
        categorical_features=["category"],
        category_policy="identity",
        preset="balanced",
        random_state=71,
    ).fit(frame, y)
    assert model.fit_diagnostics_["shared_preprocessing"] is True
    compiled = model.compile_native(tmp_path / "bundle_source")
    manifest_path = compiled.save(tmp_path / "bundle_saved", include_source=False)
    manifest = verify_export(manifest_path)
    assert manifest["format"] == "cerm-compiled-bundle-v2"
    assert manifest["shared_adapter"] is not None
    assert len(list(manifest_path.parent.glob("shared_adapter.*"))) == 2
    assert not list(manifest_path.parent.rglob("*.joblib"))
    restored = CompiledProgramBundle.load(manifest_path.parent)
    np.testing.assert_allclose(
        restored.predict_proba(frame.iloc[:45]),
        model.predict_proba(frame.iloc[:45]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        restored.predict(frame.iloc[:45]),
        model.predict(frame.iloc[:45]),
    )


def test_compiled_multilabel_bundle_v2_roundtrip_with_constant_head(tmp_path):
    from cerm import CompiledProgramBundle, verify_export

    rng = np.random.default_rng(73)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=180),
            "category": rng.choice(["left", "right"], size=180),
        }
    )
    y = np.column_stack(
        [
            (frame["x"].to_numpy() > 0).astype(int),
            (frame["category"] == "right").astype(int).to_numpy(),
            np.ones(len(frame), dtype=int),
        ]
    )
    base = CERMClassifier(
        categorical_features=["category"],
        category_policy="identity",
        preset="balanced",
        random_state=73,
    )
    model = CERMMultiLabelClassifier(estimator=base).fit(frame, y)
    compiled = model.compile_native(tmp_path / "multilabel_source")
    manifest_path = compiled.save(tmp_path / "multilabel_saved")
    verify_export(manifest_path)
    restored = CompiledProgramBundle.load(manifest_path.parent)
    np.testing.assert_allclose(
        restored.predict_proba(frame.iloc[:35]),
        model.predict_proba(frame.iloc[:35]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        restored.predict(frame.iloc[:35]),
        model.predict(frame.iloc[:35]),
    )


def test_portable_typed_adapter_compact_mapping_and_legacy_compatibility(tmp_path):
    import json
    from cerm import CERMClassifier, PortableTypedAdapter
    from cerm._internal.cerm_typed_quotient_adapters_v4 import export_typed_adapter_ir

    rng = np.random.default_rng(79)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=220),
            "category": [f"category-{value:03d}" for value in rng.integers(0, 50, size=220)],
        }
    )
    y = ((frame["x"].to_numpy() > 0) ^ (frame["category"].str[-3:].astype(int).to_numpy() < 12)).astype(int)
    model = CERMClassifier(
        categorical_features=["category"],
        category_policy="identity",
        preset="balanced",
        random_state=79,
    ).fit(frame, y)
    npz_path, json_path = export_typed_adapter_ir(model.adapter_, tmp_path / "compact")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    mapping = metadata["categorical"][0]["mapping"]
    assert metadata["schema_version"] == 3
    assert mapping["encoding"] == "parallel-json-v1"
    assert mapping["key_type"] == "str"
    assert len(mapping["keys"]) == len(mapping["states"])

    compact = PortableTypedAdapter.load(json_path, npz_path)
    expected = model.adapter_.transform(frame.iloc[:50]).matrix
    np.testing.assert_array_equal(compact.transform(frame.iloc[:50]).matrix, expected)

    # Version-1 row-wise mapping remains readable for existing artifacts.
    legacy = dict(metadata)
    legacy["schema_version"] = 2
    legacy_entry = dict(legacy["categorical"][0])
    legacy_entry["mapping"] = [
        {"key": {"type": "str", "value": key}, "state": state}
        for key, state in zip(mapping["keys"], mapping["states"])
    ]
    legacy["categorical"] = [legacy_entry]
    legacy_json = tmp_path / "legacy.json"
    legacy_json.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
    restored_legacy = PortableTypedAdapter.load(legacy_json, npz_path)
    np.testing.assert_array_equal(
        restored_legacy.transform(frame.iloc[:50]).matrix,
        expected,
    )
