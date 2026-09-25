import numpy as np
from cerm import CERMGeneralizedRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 6))
y = X[:, 0] + rng.normal(scale=0.5 + np.abs(X[:, 1]))
model = CERMGeneralizedRegressor(
    loss="multi_quantile", quantiles=(0.1, 0.5, 0.9), preset="balanced"
).fit(X, y)
print(model.predict(X[:5]))
