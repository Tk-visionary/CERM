from sklearn.datasets import make_multilabel_classification

from cerm import CERMClassifier, CERMMultiLabelClassifier

X, y = make_multilabel_classification(
    n_samples=200, n_features=12, n_classes=4, random_state=42
)
model = CERMMultiLabelClassifier(
    estimator=CERMClassifier(preset="balanced", random_state=42)
).fit(X, y)
print(model.predict_proba(X[:5]))
print(model.predict(X[:5]))
