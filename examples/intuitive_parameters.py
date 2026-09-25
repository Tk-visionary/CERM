from cerm import CERMClassifier

# Recommended user-facing configuration.
model = CERMClassifier(
    search_effort="balanced",
    state_detail="fine",
    feature_budget=64,
    interaction_search_features=24,
    interaction_budget=16,
    interaction_order=2,
    l2_regularization="auto",
    memory_limit_mb=4096,
    resource_policy="raise",
    random_state=42,
)

print(model.parameter_summary())
plan = model.estimate_fit_resources(X_train)
print(plan.to_dict())
model.fit(X_train, y_train)
print(model.get_user_params())
print(model.resolved_params_)
print(model.fit_diagnostics_.to_dict())
