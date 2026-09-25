from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

from cerm import CERMRegressor

X, y = make_friedman1(n_samples=800, noise=1.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = CERMRegressor(random_state=42).fit(X_train, y_train)
print(model.predict(X_test[:5]))
print(model.score(X_test, y_test))
print(model.fit_diagnostics_["engine"], model.selected_kind_)
