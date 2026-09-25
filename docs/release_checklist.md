# Release checklist

1. Confirm the release version in `src/cerm/_version.py`, `CHANGELOG.md`, and
   `CITATION.cff`.
2. Confirm that the distribution name in `pyproject.toml` is `CERM` and that the
   import package remains `cerm`.
3. Confirm that `LICENSE`, `pyproject.toml`, and `CITATION.cff` all declare
   Apache License 2.0 / `Apache-2.0`.
4. Confirm that `PROVENANCE.md`, `AI_ASSISTANCE.md`, and
   `THIRD_PARTY_NOTICES.md` remain accurate.
5. Confirm that research-only `benchmarks/`, `undr/`, and `docs/research/` trees
   are absent from the library repository and maintained in the separate
   research workspace.
6. Run `python scripts/check_public_api.py` to verify live constructor signatures,
   semantic aliases, `[project.urls]`, citation metadata, and the current-version
   changelog heading.
7. Run `python scripts/check_documentation.py` and
   `python -m mkdocs build --strict`.
8. Run `python -m pytest --cov=cerm`.
9. Run `python scripts/run_estimator_checks.py`.
10. Run `python scripts/check_runtime_dependencies.py`.
11. Run `python scripts/check_package.py`, then remove generated caches before
    packaging.
12. Build both sdist and wheel with `python -m build` in a clean environment.
13. Run `python scripts/verify_distributions.py dist/*` and
    `python -m twine check dist/*`.
14. Install each artifact into a fresh environment, run
    `python scripts/smoke_install.py`, and run `python -m pip check`.
15. The default pull-request CI intentionally runs one Ubuntu/Python 3.12 core
    validation job only. Before a release, manually dispatch the `CI` workflow
    with `full=true` to run Python 3.10/3.13, macOS, Windows, estimator checks,
    distribution verification, and the fresh-wheel smoke test.
16. Confirm the PyPI pending/trusted publisher exactly matches:
    `CERM`, `Tk-visionary/CERM`, `release-build.yml`, environment `pypi`.
17. Confirm the GitHub environment is named `pypi`. Add deployment protection or
    required-review rules if the repository visibility and GitHub plan support them.
18. Optionally run the `Build and publish release` workflow manually. Manual
    dispatch builds and verifies artifacts but intentionally does not publish them.
19. Create and push the exact version tag `v<version>` from the source commit to
    publish. The workflow rejects a tag that does not match `src/cerm/_version.py`.
20. The publish job must receive only `id-token: write`, download the validated
    build artifact, and publish through PyPI Trusted Publishing without an API token.
21. After publication, verify `python -m pip install CERM==<version>` in a fresh
    environment and confirm the rendered PyPI metadata, project URLs, license,
    and attestations.
