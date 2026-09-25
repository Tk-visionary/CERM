# Security notes

Do not load untrusted joblib files, training caches, compiled packages, shared
libraries, or export manifests. Native compilation executes a local compiler
and loads its output into the Python process.

Checksum verification detects corruption but does not make an artifact safe.
See the repository-level `SECURITY.md` for the reporting and support policy.
