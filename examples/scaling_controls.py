from cerm import CERMClassifier

# Defaults preserve the full 0.5.9 semantics.
full = CERMClassifier(preset="balanced")

# Explicit resource reduction. Validate on the deployment split before use.
reduced = CERMClassifier(
    preset="balanced",
    max_bins=8,
    subsample=0.75,
    colsample=0.85,
    n_jobs=2,
)

# After fitting:
# print(reduced.fit_diagnostics_.active_reductions)
# print(reduced.fit_diagnostics_.search_semantics)
