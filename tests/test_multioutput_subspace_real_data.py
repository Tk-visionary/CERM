import numpy as np
from sklearn.datasets import load_linnerud

from cerm import CERMMultiOutputRegressor


def test_shared_subspace_linnerud_real_multioutput_smoke():
    X, y = load_linnerud(return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    model = CERMMultiOutputRegressor(
        representation_strategy="shared_subspace",
        n_jobs=1,
    ).fit(X, y)
    prediction = model.predict(X)

    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()
    assert model.model_bytes_estimate_ > 0
    assert model.fit_diagnostics_["task_type"] == "multioutput_regression"
    assert model.fit_diagnostics_["strategy"] in {
        "shared_subspace_finite_state",
        "shared_subspace_fallback_independent",
    }
