from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from cerm import CERMClassifier, model_summary


X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42,
)

# Start with the ordinary sklearn-style default estimator. Tune only after a
# task-level validation shows that a different search budget is worthwhile.
model = CERMClassifier(random_state=42).fit(X_train, y_train)

probability = model.predict_proba(X_test)[:, 1]
print("log loss:", log_loss(y_test, probability))
print("ROC AUC:", roc_auc_score(y_test, probability))
print(model_summary(model))
