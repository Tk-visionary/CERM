import numpy as np
from cerm import CERMClassifier

# Raise the training cost of positive examples without discarding any rows.
weights = np.where(y_train == 1, 5.0, 1.0)
model = CERMClassifier(preset="balanced").fit(
    X_train, y_train, sample_weight=weights
)
