"""Run scikit-learn estimator compliance checks against public CERM estimators."""

from __future__ import annotations

import warnings
from unittest import SkipTest

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.estimator_checks import estimator_checks_generator

from cerm import (
    CERMClassifier,
    CERMGeneralizedRegressor,
    CERMMultiOutputRegressor,
    CERMRegressor,
    CERMRidgeRegressor,
)


KNOWN_XFAILS = {
    (
        "CERMClassifier",
        "check_sample_weight_equivalence_on_dense_data",
    ): (
        "Classification has weighted nested representation semantics, but full "
        "adaptive training is not defined to equal materialized row replication."
    ),
    (
        "CERMRegressor",
        "check_sample_weight_equivalence_on_dense_data",
    ): (
        "Fused regression matches integer-frequency weighting at fixed "
        "representation/objective boundaries, but adaptive holdout/cross-fit "
        "splits and regularized objectives are not whole-estimator row-replication invariants."
    ),
    (
        "CERMGeneralizedRegressor",
        "check_sample_weight_equivalence_on_dense_data",
    ): (
        "The generalized head inherits finite-state feature-distribution semantics, "
        "so weighted fitting is not required to equal literal row replication."
    ),
    (
        "CERMMultiOutputRegressor",
        "check_sample_weight_equivalence_on_dense_data",
    ): (
        "The shared/independent multi-output estimators retain their historical "
        "finite-state weighting contract rather than literal row replication."
    ),
}


def _check_name(check) -> str:
    function = getattr(check, "func", check)
    return getattr(function, "__name__", repr(check))


def run(estimator, name: str) -> tuple[int, int, int]:
    passed = 0
    xfailed = 0
    skipped = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for candidate, check in estimator_checks_generator(estimator, legacy=True):
            check_name = _check_name(check)
            try:
                check(candidate)
            except SkipTest as exc:
                skipped += 1
                print(f"{name} skipped: {check_name}: {exc}")
            except Exception:
                reason = KNOWN_XFAILS.get((name, check_name))
                if reason is None:
                    raise
                xfailed += 1
                print(f"{name} expected xfail: {check_name}: {reason}")
            else:
                passed += 1
    print(
        f"{name} estimator checks passed: {passed}; expected xfails: {xfailed}; "
        f"environmental skips: {skipped}"
    )
    return passed, xfailed, skipped


def main() -> None:
    classifier = CERMClassifier(
        max_features=6,
        max_interaction_features=4,
        max_interactions=4,
        resource_policy="ignore",
        random_state=0,
    )
    regressor = CERMRegressor(
        n_bins=5,
        max_features=6,
        max_bins=4,
        max_interaction_features=4,
        max_pairs=1,
        random_state=0,
    )
    generalized = CERMGeneralizedRegressor(
        loss="huber",
        preset="balanced",
        max_features=4,
        max_bins=4,
        max_interaction_features=4,
        max_interactions=2,
        max_iter=200,
        random_state=0,
    )
    multioutput = CERMMultiOutputRegressor(
        estimator=CERMRidgeRegressor(
            preset="balanced",
            max_features=4,
            max_bins=4,
            max_interaction_features=4,
            max_interactions=0,
            random_state=0,
        ),
        representation_strategy="shared",
    )
    results = [
        run(classifier, "CERMClassifier"),
        run(regressor, "CERMRegressor"),
        run(generalized, "CERMGeneralizedRegressor"),
        run(multioutput, "CERMMultiOutputRegressor"),
    ]
    total_passed = sum(row[0] for row in results)
    total_xfailed = sum(row[1] for row in results)
    total_skipped = sum(row[2] for row in results)
    print(
        "total scikit-learn estimator checks passed: "
        f"{total_passed}; expected xfails: {total_xfailed}; "
        f"environmental skips: {total_skipped}"
    )


if __name__ == "__main__":
    main()
