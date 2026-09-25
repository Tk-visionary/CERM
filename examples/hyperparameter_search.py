from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split

from cerm import (
    CERMClassifier,
    get_resource_params,
    get_search_params,
    get_tunable_params,
)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

base = CERMClassifier(random_state=42, n_jobs=1)
print("model HPO:", get_tunable_params(base))
print("search controls:", get_search_params(base))
print("resources:", get_resource_params(base))

# Keep the tutorial grid intentionally small. Larger studies can expand these
# same direct numeric/structural parameters or use an adaptive HPO library.
search = GridSearchCV(
    base,
    {
        "max_bins": [8, 16],
        "max_features": [32, 64],
        "max_interaction_features": [16, 24],
        "max_interactions": [8, 16],
        "reg_lambda": [0.3, 1.0],
    },
    scoring="neg_log_loss",
    cv=3,
    n_jobs=1,
).fit(X_train, y_train)

print(search.best_params_)
print(search.score(X_test, y_test))
