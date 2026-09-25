from __future__ import annotations

import numpy as np
import pandas as pd

from cerm import CERMClassifier, CERMRegressor
from cerm._internal.cerm_typed_quotient_adapters_v4 import TypedQuotientAdapter
from cerm.regression import RegressionTypedAdapter
from cerm.validation import validate_dataframe_schema


def _typed_frame(seed=7, n=240):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat0": rng.choice(["a", "b", "c", None], size=n),
            "cat1": rng.choice(["u", "v", "w", None], size=n),
        }
    )
    return frame


def test_dataframe_schema_exact_order_is_zero_copy():
    frame = _typed_frame()
    validated = validate_dataframe_schema(frame, tuple(frame.columns))
    assert validated is frame
    reordered = frame.loc[:, list(reversed(frame.columns))]
    restored = validate_dataframe_schema(reordered, tuple(frame.columns))
    assert list(restored.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(restored, frame, check_exact=True)


def test_typed_adapter_projected_columns_match_full_transform_bitwise():
    frame = _typed_frame(8, 300)
    y = (frame["x0"].to_numpy() > 0).astype(int)
    adapter = TypedQuotientAdapter(
        categorical_columns=["cat0", "cat1"],
        category_policy="identity",
        missing_policy="always",
        random_state=8,
    )
    adapter.fit_transform(frame, y)
    probe = _typed_frame(9, 180)
    probe.loc[:5, "cat0"] = "unseen"
    full = adapter.transform(probe).matrix
    selected = np.asarray([0, 2, full.shape[1] - 1], dtype=np.int64)
    projected = adapter.transform_columns(probe, selected).matrix
    assert np.array_equal(projected, full[:, selected])


def test_regression_adapter_projected_columns_match_full_transform_bitwise():
    frame = _typed_frame(10, 320)
    rng = np.random.default_rng(10)
    y = 1.2 * frame["x0"].to_numpy() - 0.4 * frame["x1"].to_numpy() + rng.normal(
        scale=0.1, size=len(frame)
    )
    adapter = RegressionTypedAdapter(
        categorical_columns=["cat0", "cat1"],
        category_policy="identity",
        missing_policy="always",
        random_state=10,
    )
    adapter.fit_transform(frame, y)
    probe = _typed_frame(11, 170)
    full = adapter.transform(probe).matrix
    selected = np.asarray([1, 3, full.shape[1] - 1], dtype=np.int64)
    projected = adapter.transform_columns(probe, selected).matrix
    assert np.array_equal(projected, full[:, selected])


def test_state_stream_matches_legacy_code_stream_with_unobserved_joint_states():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    model = CERMClassifier(
        max_features=6,
        max_interaction_features=6,
        max_interactions=4,
        random_state=0,
    ).fit(X, y)
    base = getattr(model.model_, "base_", model.model_)
    # This seed deliberately leaves at least one theoretically possible code
    # outside the observed one-hot cardinality, exercising the masked fallback.
    assert np.any(~base._lookup_full_domain_flags())
    probe = np.vstack(
        [rng.normal(size=(1200, 6)), np.full((1, 6), 10.0), np.full((1, 6), -10.0)]
    )
    states = base._states_for_X(probe)
    legacy = base._decision_from_codes(base._build_codes_from_states(states))
    streamed = base._decision_from_states(states)
    assert np.array_equal(streamed, legacy)


def test_hybrid_sparse_base_sum_and_streamed_lookup_are_bitwise_equal():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(520, 10))
    y = ((X[:, 0] * X[:, 1] + 0.7 * X[:, 2]) > 0).astype(int)
    model = CERMClassifier(
        max_features=10,
        max_interaction_features=8,
        max_interactions=5,
        random_state=12,
    ).fit(X, y)
    hybrid = model.model_
    if not hasattr(hybrid, "base_coef_"):
        return
    probe = rng.normal(size=(900, 10))
    states = hybrid.base_._states_for_X(probe)
    codes = hybrid.base_._build_codes_from_states(states)
    sparse_design = hybrid.base_.oh_.transform(codes)
    legacy = np.asarray(sparse_design @ hybrid.base_coef_).ravel() + hybrid.intercept_
    streamed = hybrid.base_._decision_from_states_with_lookup(
        states, hybrid._base_lookup_tables(), hybrid.intercept_, intercept_last=True
    )
    assert np.array_equal(streamed, legacy)


def test_typed_native_compile_projects_to_core_feature_count(tmp_path):
    frame = _typed_frame(13, 500)
    y = (
        frame["x0"].to_numpy()
        + 0.5 * frame["x1"].to_numpy()
        + (frame["cat0"] == "b").astype(float).to_numpy()
        > 0
    ).astype(int)
    model = CERMClassifier(
        categorical_features=["cat0", "cat1"],
        category_policy="identity",
        max_features=4,
        max_interaction_features=4,
        max_interactions=2,
        random_state=13,
    ).fit(frame, y)
    optimized = model.program_.optimize("latency")
    compiled = optimized.compile(tmp_path / "typed")
    assert len(compiled.feature_indices) == len(optimized.graph.feature_idx)
    probe = _typed_frame(14, 80)
    np.testing.assert_allclose(
        compiled.predict_proba(probe), optimized.predict_proba(probe), rtol=0.0, atol=2e-12
    )
