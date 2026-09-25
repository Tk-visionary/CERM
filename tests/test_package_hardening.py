from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.utils import get_tags

from cerm import CERMClassifier, CERMConfig, TypedAdapterConfig


def binary_data(n_samples=120, n_features=5, seed=73):
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(1, min(n_features, 3)),
        n_redundant=0,
        random_state=seed,
    )


def test_config_roundtrip_preserves_selection_controls():
    config = CERMConfig(
        max_features=7,
        pair_feature_limit=5,
        prediction_backend="semantic",
        selection_strategy="cross_fitted",
        selection_folds=4,
        selection_near_tie=0.002,
        selection_min_improvement=0.0007,
        calibration="affine",
        calibration_folds=4,
        calibration_l2=0.002,
        calibration_min_improvement=0.0003,
        calibration_min_signal=1.25,
        cache_training_statistics=True,
        adapter=TypedAdapterConfig(categorical_features=("cat",)),
    )
    estimator = CERMClassifier.from_config(config)
    restored = estimator.to_config()
    assert restored == config


def test_one_feature_dataset_is_supported():
    X = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)
    model = CERMClassifier(
        max_features=1,
        pair_feature_limit=1,
        random_state=79,
    ).fit(X, y)
    assert model.predict_proba(X[:8]).shape == (8, 2)


def test_input_validation_is_explicit():
    X, y = binary_data()
    with pytest.raises(TypeError, match="sparse input is not supported"):
        CERMClassifier(max_features=5, pair_feature_limit=5).fit(sparse.csr_matrix(X), y)

    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN|infinity|finite"):
        CERMClassifier(max_features=5, pair_feature_limit=5).fit(bad, y)

    continuous = np.linspace(0.0, 1.0, len(y))
    with pytest.raises(ValueError, match="continuous"):
        CERMClassifier(max_features=5, pair_feature_limit=5).fit(X, continuous)

    one_class = np.zeros(len(y), dtype=int)
    with pytest.raises(ValueError, match="binary|2 classes"):
        CERMClassifier(max_features=5, pair_feature_limit=5).fit(X, one_class)


def test_multiclass_sklearn_tag_is_declared():
    tags = get_tags(CERMClassifier())
    assert tags.estimator_type == "classifier"
    assert tags.classifier_tags.multi_class is True
    assert tags.input_tags.sparse is False


def test_optimized_program_is_cached_and_int16_compiles(tmp_path):
    X, y = binary_data(n_samples=180, n_features=6, seed=83)
    model = CERMClassifier(
        max_features=6,
        pair_feature_limit=6,
        random_state=83,
    ).fit(X, y)
    first = model.optimize("memory")
    second = model.optimize("memory")
    assert first is second

    compiled = first.compile(tmp_path / "int16_model", dtype="int16")
    expected = model.predict_proba(X[:60])[:, 1]
    actual = compiled.predict_proba(X[:60])[:, 1]
    error = float(np.max(np.abs(expected - actual)))
    assert error <= compiled.metadata["probability_error_bound"] + 1e-12
    assert compiled.metadata["compression_ratio"] < 1.0


def test_export_manifest_checksums_match_files(tmp_path):
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 120),
            "cat": [f"c{i % 5}" for i in range(120)],
        }
    )
    y = (frame["x"].to_numpy() + (np.arange(120) % 5 == 4) > 0).astype(int)
    model = CERMClassifier(
        max_features=5,
        pair_feature_limit=5,
        categorical_features="auto",
        random_state=89,
    ).fit(frame, y)
    manifest_path = model.export(tmp_path / "export")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "cerm-python-package-v2"
    for section in ("model", "adapter"):
        for record in manifest[section].values():
            payload = (manifest_path.parent / record["file"]).read_bytes()
            import hashlib

            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
            assert len(payload) == record["bytes"]


def test_embedding_adapter_lazy_dependencies_are_loaded_on_demand():
    rng = np.random.default_rng(97)
    matrix = rng.normal(size=(100, 8))
    frame = pd.DataFrame(matrix, columns=[f"e{i}" for i in range(8)])
    y = (matrix[:, 0] - matrix[:, 1] > 0).astype(int)
    model = CERMClassifier(
        max_features=10,
        pair_feature_limit=8,
        categorical_features=(),
        embedding_features=tuple(frame.columns),
        embedding_mode="basic",
        embedding_pca=3,
        embedding_bins=4,
        random_state=97,
    ).fit(frame, y)
    assert model.predict_proba(frame.iloc[:12]).shape == (12, 2)


def test_identity_category_missing_policy_always_reserves_unknown_state():
    from cerm.adapters import TypedQuotientAdapter

    frame = pd.DataFrame({"cat": ["a", "b", "a", "b"] * 10})
    y = np.asarray([0, 1, 0, 1] * 10)
    adapter = TypedQuotientAdapter(
        categorical_columns=["cat"],
        category_policy="identity",
        missing_policy="always",
    )
    trained = adapter.fit_transform(frame, y)
    transformed = adapter.transform(pd.DataFrame({"cat": ["a", "unknown", None]}))
    assert trained.cardinalities[0] == 3
    assert transformed.matrix[1, 0] == transformed.matrix[2, 0] == 2


def test_public_prediction_backend_is_exactly_optimized():
    X, y = binary_data(n_samples=180, n_features=6, seed=107)
    optimized = CERMClassifier(
        max_features=6,
        pair_feature_limit=6,
        prediction_backend="optimized",
        random_state=107,
    ).fit(X, y)
    semantic = CERMClassifier(
        max_features=6,
        pair_feature_limit=6,
        prediction_backend="semantic",
        random_state=107,
    ).fit(X, y)
    np.testing.assert_allclose(
        optimized.predict_proba(X), semantic.predict_proba(X), atol=2e-12, rtol=0
    )
    assert optimized._prediction_program_ is optimized.optimize("balanced")


def test_verify_export_detects_tampering(tmp_path):
    from cerm import verify_export

    X, y = binary_data(n_samples=120, n_features=5, seed=109)
    model = CERMClassifier(max_features=5, pair_feature_limit=5, random_state=109).fit(X, y)
    manifest_path = model.export(tmp_path / "verified")
    manifest = verify_export(manifest_path)
    model_file = manifest_path.parent / manifest["model"]["json"]["file"]
    model_file.write_bytes(model_file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="byte-count|SHA-256"):
        verify_export(manifest_path)


def test_config_roundtrip_preserves_auto_categorical_policy():
    estimator = CERMClassifier(categorical_features="auto")
    restored = CERMClassifier.from_config(estimator.to_config())
    assert restored.categorical_features == "auto"


def test_graph_state_width_is_lossless_for_high_cardinality_direct_states(tmp_path):
    from cerm._internal.cerm_graph_int16_codegen import compile_graph_int16_native
    from cerm._internal.cerm_graph_native_codegen import compile_graph_native
    from cerm._internal.cerm_graph_native_codegen_v2 import compile_graph_native_v2
    from cerm._internal.cerm_graph_optimizer import CERMGraphProgram
    from cerm._internal.cerm_graph_optimizer_v2 import CERMGraphProgramV2
    from cerm._internal.cerm_graph_quantization import quantize_graph_int16

    X = np.asarray([[299.0], [43.0], [0.0]])
    table = np.arange(300, dtype=np.float64) * 0.001
    kwargs = dict(
        feature_idx=np.asarray([0]),
        thresholds=[np.empty(0, dtype=np.float64)],
        unary_tables={0: table},
        pair_programs=[],
        intercept=0.0,
        direct_state_mask=np.asarray([True]),
        direct_state_cardinalities=np.asarray([300]),
    )
    v1 = CERMGraphProgram(**kwargs)
    v2 = CERMGraphProgramV2(**kwargs)
    expected_states = np.asarray([299, 43, 0], dtype=np.uint16)

    for program in (v1, v2):
        assert program.states(X).dtype == np.dtype("uint16")
        np.testing.assert_array_equal(program.states(X).ravel(), expected_states)
        np.testing.assert_allclose(program.decision_function(X), table[expected_states])

    expected_probability = 1.0 / (1.0 + np.exp(-table[expected_states]))
    native_v1, _, _ = compile_graph_native(v1, tmp_path / "graph_v1")
    native_v2, _, _ = compile_graph_native_v2(v2, tmp_path / "graph_v2")
    np.testing.assert_allclose(native_v1(X), expected_probability, atol=1e-14, rtol=0)
    np.testing.assert_allclose(native_v2(X), expected_probability, atol=1e-14, rtol=0)

    quantized = quantize_graph_int16(v2)
    assert quantized.states(X).dtype == np.dtype("uint16")
    np.testing.assert_array_equal(quantized.states(X).ravel(), expected_states)
    expected_quantized_probability = 1.0 / (1.0 + np.exp(-quantized.decision_function(X)))
    native_int16, _, _ = compile_graph_int16_native(
        quantized, tmp_path / "graph_int16"
    )
    np.testing.assert_allclose(
        native_int16(X), expected_quantized_probability, atol=1e-12, rtol=0
    )
