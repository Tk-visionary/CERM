from pathlib import Path
from sklearn.datasets import load_breast_cancer

from cerm import CERMClassifier

X, y = load_breast_cancer(return_X_y=True)
model = CERMClassifier(max_features=20, max_interaction_features=14).fit(X, y)

semantic_probability = model.predict_proba(X[:100])
optimized = model.optimize(target="balanced")
optimized_probability = optimized.predict_proba(X[:100])

runtime = optimized.compile(Path("cerm_native_model"), dtype="float64")
native_probability = runtime.predict_proba(X[:100])

print(abs(semantic_probability - optimized_probability).max())
print(abs(semantic_probability - native_probability).max())
