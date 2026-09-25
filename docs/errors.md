# Validation errors

CERM uses a small public exception taxonomy for errors that users can reasonably
recover from in ordinary Python workflows. The classes remain subclasses of the
historical built-in exceptions, so existing `except ValueError` and
`except RuntimeError` handlers continue to work.

```python
from cerm import (
    CERMDataSchemaError,
    CERMParameterError,
    CERMAliasConflictError,
    CERMResourceLimitError,
)
```

## DataFrame schema mismatch

A model fitted with a DataFrame remembers its input columns. Prediction accepts
the same columns in a different order, but missing or unexpected columns raise
`CERMDataSchemaError` with both lists and a recovery hint.

## Categorical specification

Invalid categorical/embedding public parameter values use the public parameter
validation path. String-vs-sequence type mistakes remain `TypeError`; invalid
value combinations are validation errors. A column cannot simultaneously be
configured as categorical and embedding.

## Semantic versus historical parameters

CERM keeps historical constructor names for scikit-learn cloning and
reproducibility. New code should prefer semantic names such as `state_detail`
and `feature_budget`. If both naming styles are supplied and disagree,
`CERMAliasConflictError` is raised rather than silently choosing one.

For example, do not combine:

```python
CERMClassifier(state_detail="coarse", max_bins=16)
```

Use one naming style instead.

## Resource budgets

`CERMResourceLimitError` is raised before an expensive fit when
`resource_policy="raise"` and an explicit structural budget would be exceeded.
The message identifies every violated budget and points to:

```python
plan = model.estimate_fit_resources(X)
print(plan.to_dict())
```

`resource_policy="warn"` emits `CERMResourceWarning` instead. Resource estimates
are structural safety estimates, not measured peak-RSS guarantees.
