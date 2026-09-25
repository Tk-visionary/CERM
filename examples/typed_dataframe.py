import numpy as np
import pandas as pd

from cerm import CERMClassifier

n = 300
X = pd.DataFrame(
    {
        "income": np.linspace(0, 1, n),
        "region": [f"r{i % 12}" for i in range(n)],
        "usage": np.where(np.arange(n) % 11 == 0, np.nan, np.sin(np.arange(n) / 20)),
    }
)
y = np.where(X["income"].to_numpy() + (np.arange(n) % 12 > 8) * 0.4 > 0.7, "high", "low")

model = CERMClassifier(
    categorical_features="auto",
    category_policy="auto",
    max_identity_categories=8,
    missing_policy="observed",
    max_features=16,
    max_interaction_features=12,
).fit(X, y)

print(model.predict_proba(X.head()))
print(model.get_feature_names_out())
