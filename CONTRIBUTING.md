# Contributing

CERM is distributed under the Apache License 2.0. Contributions are welcome.

For local development:

```bash
python -m pip install -e ".[test]"
python -m pytest
PYTHONPATH=src python scripts/run_estimator_checks.py
python scripts/check_package.py
python scripts/check_runtime_dependencies.py
```

## Contribution expectations

Changes to statistical behavior must include task-level tests and must state
whether they are an exact rewrite, a bounded-exact search change, or an
approximation. Changes to optimized or native backends must compare predictions
against the semantic backend.

Public API changes require a changelog entry and, after the API stabilizes, a
deprecation path. New private scikit-learn access is allowed only inside
`cerm._compat` and must have a public fallback.

Contributors must not submit code copied from a source whose license is
incompatible or unknown. Any adapted third-party code must retain required
copyright and license notices and must be recorded in
`THIRD_PARTY_NOTICES.md`. AI-assisted contributions must be reviewed and tested
by the human contributor before submission.

Unless explicitly stated otherwise, contributions intentionally submitted for
inclusion in CERM are provided under the Apache License 2.0, consistent with
Section 5 of that license.
