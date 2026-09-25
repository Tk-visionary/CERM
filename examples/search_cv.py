from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from cerm import CERMClassifier, CERMSearchCV

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

search = CERMSearchCV(
    CERMClassifier(
        max_features=12,
        ranking_kind="newton",
        random_state=42,
    ),
    {
        "ranking_l2": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        "max_interaction_features": [2, 4, 8, 12],
    },
    cv=3,
    prefilter_top_k=8,
).fit(X_train, y_train)

print(search.best_params_)
print(search.search_diagnostics_)
print(search.score(X_test, y_test))
