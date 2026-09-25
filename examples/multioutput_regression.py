import numpy as np
from cerm import CERMMultiOutputRegressor, CERMRidgeRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 6))
y = np.column_stack((X[:, 0] + X[:, 1], X[:, 0] - X[:, 2]))
model = CERMMultiOutputRegressor(
    estimator=CERMRidgeRegressor(preset="balanced"),
    representation_strategy="shared",
).fit(X, y)
print(model.predict(X[:5]))
