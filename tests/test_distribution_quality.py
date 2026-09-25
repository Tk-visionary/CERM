from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

import cerm
from cerm import CERMClassifier, show_versions
from cerm._compat import num_samples, safe_indexing
from cerm._compat.sklearn import PrivateSklearnAPIUnavailable
from cerm.training_graph import solve_binary_logistic_path


ROOT = Path(__file__).resolve().parents[1]


def test_version_is_single_sourced():
    assert cerm.__version__ == CERMClassifier.VERSION
    assert cerm.__version__ == "1.0.0a1"
    assert "dynamic = [\"version\"]" in (ROOT / "pyproject.toml").read_text()


def test_repository_tests_can_import_src_layout_directly():
    assert (ROOT / "src" / "cerm" / "__init__.py").is_file()
    assert cerm.__file__ is not None


def test_compat_num_samples_supports_common_inputs():
    assert num_samples(np.zeros((5, 2))) == 5
    assert num_samples(pd.DataFrame({"x": range(4)})) == 4
    assert num_samples([1, 2, 3]) == 3


def test_compat_num_samples_uses_cerm_specific_scalar_error():
    with np.testing.assert_raises_regex(TypeError, "CERM requires an input with a sample dimension"):
        num_samples(np.asarray(1.0))


def test_compat_safe_indexing_preserves_supported_container_types():
    indices = np.array([2, 0])
    frame = pd.DataFrame({"x": [10, 20, 30]})
    matrix = sparse.csr_matrix(np.arange(6).reshape(3, 2))
    array = np.arange(6).reshape(3, 2)

    assert safe_indexing(frame, indices)["x"].tolist() == [30, 10]
    assert sparse.issparse(safe_indexing(matrix, indices))
    np.testing.assert_array_equal(safe_indexing(array, indices), array[indices])
    np.testing.assert_array_equal(safe_indexing([10, 20, 30], indices), [30, 10])


def test_private_liblinear_failure_uses_public_fallback(monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] - 0.4 * X[:, 1] > 0).astype(int)
    valid = rng.normal(size=(20, 5))

    def unavailable(*args, **kwargs):
        raise PrivateSklearnAPIUnavailable("simulated ABI change")

    monkeypatch.setattr("cerm.training_graph.train_binary_liblinear", unavailable)
    solution = solve_binary_logistic_path(
        X,
        y,
        valid,
        [0.2, 1.0],
        random_state=7,
        max_iter=1000,
    )

    for C in (0.2, 1.0):
        reference = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=1000,
            random_state=7,
        ).fit(X, y)
        np.testing.assert_allclose(
            solution[C].valid_probability,
            reference.predict_proba(valid)[:, 1],
            rtol=0,
            atol=0,
        )


def test_show_versions_reports_reproducibility_context():
    versions = show_versions()
    assert versions["cerm"] == cerm.__version__
    for key in ("python", "platform", "numpy", "scipy", "pandas", "scikit-learn", "joblib"):
        assert versions[key]


def test_open_source_release_metadata_and_provenance_files_are_present():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in pyproject
    assert 'authors = [{name = "Taishi Kawahara"}]' in pyproject
    assert license_text.startswith("Apache License")
    assert "Version 2.0, January 2004" in license_text
    for name in ("AI_ASSISTANCE.md", "PROVENANCE.md", "CITATION.cff"):
        assert (ROOT / name).is_file()


def test_research_scale_assets_are_split_from_library_repo():
    assert not (ROOT / "benchmarks").exists()
    assert not (ROOT / "undr").exists()
    assert not (ROOT / "docs" / "research").exists()


def test_historical_vendor_package_was_renamed_to_internal():
    assert not (ROOT / "src" / "cerm" / "_vendor").exists()
    assert (ROOT / "src" / "cerm" / "_internal" / "__init__.py").is_file()
