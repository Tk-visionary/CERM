"""Huber, quantile, and Poisson regression with one CERM API."""

import numpy as np
from sklearn.model_selection import train_test_split

from cerm import CERMGeneralizedRegressor

rng = np.random.default_rng(7)
X = rng.normal(size=(800, 6))
y = 1.2 * X[:, 0] - 0.6 * X[:, 1] + rng.normal(scale=0.5 + np.abs(X[:, 2]))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

median = CERMGeneralizedRegressor(
    loss="quantile",
    quantile=0.5,
    preset="balanced",
    max_features=6,
    max_interactions=4,
).fit(X_train, y_train)

upper = CERMGeneralizedRegressor(
    loss="quantile",
    quantile=0.9,
    preset="balanced",
    max_features=6,
    max_interactions=4,
).fit(X_train, y_train)

print("median predictions:", median.predict(X_test[:5]))
print("90% upper predictions:", upper.predict(X_test[:5]))
