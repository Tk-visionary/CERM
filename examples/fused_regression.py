"""Explicit compatibility-name example for fused residual regression V2."""

from sklearn.datasets import make_friedman1
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from cerm import CERMFusedRegressor


X, y = make_friedman1(
    n_samples=160,
    n_features=8,
    noise=1.0,
    random_state=17,
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
)

model = CERMFusedRegressor(
    n_bins=6,
    max_features=8,
    max_bins=8,
    max_interaction_features=6,
    max_pairs=2,
    random_state=17,
).fit(X_train, y_train)

prediction = model.predict(X_test)
rmse = root_mean_squared_error(y_test, prediction)
print(f"selected_kind={model.selected_kind_} rmse={rmse:.4f}")
