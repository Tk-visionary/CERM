from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from cerm import CERMClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
model = CERMClassifier(preset="balanced", random_state=42).fit(X_train, y_train)
print(model.predict_proba(X_test[:5]))
print(model.score(X_test, y_test))
