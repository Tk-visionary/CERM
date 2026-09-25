import numpy as np
from cerm import CERMGeneralizedRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 5))
exposure = rng.uniform(0.5, 3.0, size=len(X))
y = rng.poisson(exposure * np.exp(0.2 + 0.5 * X[:, 0]))
model = CERMGeneralizedRegressor(loss="poisson", preset="balanced").fit(
    X, y, exposure=exposure
)
print(model.predict(X[:5], exposure=exposure[:5]))
