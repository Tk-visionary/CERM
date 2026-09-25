import numpy as np
from cerm import CERMGeneralizedRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 8))
exposure = rng.uniform(0.5, 2.0, size=len(X))
rate = np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1])
y = exposure * rng.gamma(shape=3.0, scale=rate / 3.0)

gamma = CERMGeneralizedRegressor(loss="gamma").fit(
    X, y, exposure=exposure
)
prediction = gamma.predict(X[:10], exposure=exposure[:10])

# Compound Poisson-Gamma-style non-negative targets, including zeros.
tweedie = CERMGeneralizedRegressor(
    loss="tweedie", tweedie_power=1.5
).fit(X, np.maximum(y - np.quantile(y, 0.35), 0.0))
