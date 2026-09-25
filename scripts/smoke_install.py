"""Minimal installed-package smoke test used by release workflows."""

from __future__ import annotations

from sklearn.datasets import load_breast_cancer

from cerm import CERMClassifier, __version__


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = CERMClassifier(
        preset="balanced",
        max_features=8,
        max_interaction_features=6,
        max_interactions=8,
        random_state=0,
    ).fit(X[:240], y[:240])
    probability = model.predict_proba(X[240:260])
    if probability.shape != (20, 2):
        raise RuntimeError(f"unexpected probability shape: {probability.shape}")
    print(f"installed CERM {__version__} smoke test passed")


if __name__ == "__main__":
    main()
