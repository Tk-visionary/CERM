import gc
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification

from cerm import CERMClassifier


def _data():
    return make_classification(
        n_samples=320,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        random_state=600,
    )


@pytest.mark.parametrize("max_bins,levels", [(4, (4,)), (8, (4, 8)), (16, (4, 8, 16))])
def test_max_bins_changes_real_state_hierarchy(max_bins, levels):
    X, y = _data()
    model = CERMClassifier(
        max_bins=max_bins,
        max_features=14,
        max_interaction_features=10,
        max_interactions=8,
        resource_policy="ignore",
        random_state=601,
    ).fit(X, y)
    assert model.model_.base_.levels == levels
    assert model.model_.block_level == max_bins
    assert model.fit_diagnostics_.max_bins == max_bins
    assert set(model.model_.base_.encoder_.maps_) == set(levels)


def test_lower_max_bins_reduces_or_matches_design_dimension():
    X, y = load_breast_cancer(return_X_y=True)
    common = dict(
        max_features=16,
        max_interaction_features=12,
        max_interactions=8,
        resource_policy="ignore",
        random_state=603,
    )
    model4 = CERMClassifier(max_bins=4, **common).fit(X, y)
    model8 = CERMClassifier(max_bins=8, **common).fit(X, y)
    model16 = CERMClassifier(max_bins=16, **common).fit(X, y)
    assert model4.model_.design_dim_ <= model8.model_.design_dim_
    assert model8.model_.design_dim_ <= model16.model_.design_dim_


def test_subsample_reduces_selection_rows_but_predicts_on_full_data():
    X, y = _data()
    model = CERMClassifier(
        subsample=0.5,
        max_features=12,
        max_interaction_features=10,
        max_interactions=8,
        resource_policy="ignore",
        random_state=607,
    ).fit(X, y)
    assert 0 < model.model_.selection_rows_ < len(X)
    assert model.model_.base_.selection_rows_ < len(X)
    assert model.predict_proba(X).shape == (len(X), 2)
    assert model.fit_diagnostics_.subsample == 0.5


def test_colsample_is_persisted_across_semantic_and_native_prediction():
    X, y = _data()
    model = CERMClassifier(
        colsample=0.5,
        max_bins=8,
        max_features=12,
        max_interaction_features=10,
        max_interactions=8,
        resource_policy="ignore",
        random_state=611,
    ).fit(X, y)
    assert model.n_adapted_features_ == 10
    assert model.n_full_adapted_features_ == 20
    reference = model.predict_proba(X[:40])
    np.testing.assert_allclose(
        model.program_.predict_proba(X[:40]), reference, atol=1e-15, rtol=0.0
    )
    # Windows cannot unlink a ctypes-loaded library until process exit. The
    # predictor is released below; tolerate only that platform cleanup delay.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=(os.name == "nt")) as directory:
        compiled = model.compile_native(Path(directory) / "model")
        np.testing.assert_allclose(
            compiled.predict_proba(X[:40]), reference, atol=1e-12, rtol=0.0
        )
        package = Path(directory) / "compiled"
        compiled.save(package)
        restored = type(compiled).load(package)
        np.testing.assert_allclose(
            restored.predict_proba(X[:40]), reference, atol=1e-12, rtol=0.0
        )
        # Windows locks loaded DLLs; release both ctypes-backed predictors
        # before TemporaryDirectory removes the compiled package.
        del restored, compiled
        gc.collect()


def test_parallel_cross_fitted_selection_is_deterministic():
    X, y = load_breast_cancer(return_X_y=True)
    common = dict(
        selection_strategy="cross_fitted",
        selection_folds=3,
        max_features=12,
        max_interaction_features=10,
        max_interactions=8,
        resource_policy="ignore",
        random_state=613,
    )
    serial = CERMClassifier(n_jobs=1, **common).fit(X, y)
    parallel = CERMClassifier(n_jobs=2, **common).fit(X, y)
    assert serial.model_.selected_hybrid_config_ == parallel.model_.selected_hybrid_config_
    np.testing.assert_allclose(
        serial.predict_proba(X), parallel.predict_proba(X), atol=0.0, rtol=0.0
    )


def test_resource_plan_reflects_sampling_and_state_resolution():
    X, _ = _data()
    plan = CERMClassifier(
        max_bins=8,
        subsample=0.5,
        colsample=0.5,
        resource_policy="ignore",
    ).estimate_fit_resources(X)
    assert plan.max_bins == 8
    assert plan.effective_selection_rows == 160
    assert plan.estimated_full_adapted_features == 20
    assert plan.estimated_adapted_features == 10
    assert plan.subsample == 0.5
    assert plan.colsample == 0.5


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"max_bins": 12}, "max_bins"),
        ({"subsample": 0.0}, "subsample"),
        ({"colsample": 1.1}, "colsample"),
        ({"n_jobs": 0}, "n_jobs"),
    ],
)
def test_scaling_parameter_validation(parameters, message):
    with pytest.raises(ValueError, match=message):
        CERMClassifier(**parameters).resolve_params()


def test_resolved_semantics_expose_active_reductions():
    resolved = CERMClassifier(
        preset="balanced",
        max_bins=8,
        subsample=0.75,
        colsample=0.8,
        max_interactions=16,
        reg_lambda=5.0,
    ).resolve_params()
    assert resolved.search_semantics != "full"
    assert set(resolved.active_reductions) >= {
        "reduced_state_resolution",
        "selection_row_subsample",
        "feature_subspace",
        "reduced_candidate_family",
        "interaction_cap",
        "fixed_regularization",
    }


def test_default_resolved_semantics_are_full():
    resolved = CERMClassifier().resolve_params()
    assert resolved.search_semantics == "full"
    assert resolved.active_reductions == ()
