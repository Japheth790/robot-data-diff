## What does this change?

<!-- One logical change per PR. Link the issue if there is one. -->

## How was it tested?

- [ ] `ruff check .` passes
- [ ] `pytest` passes, and a new test fails without this change (for bug fixes)
- [ ] `pytest -m network` (if the loader or Hub handling changed)

## Checklist

- [ ] Default CLI behaviour is unchanged, or the change is behind a flag
- [ ] No new heavy dependencies; no video downloads; still CPU-only
- [ ] JSON output: keys only added (or `SCHEMA_VERSION` bumped and `docs/json-schema.md` updated)
- [ ] `CHANGELOG.md` updated under "Unreleased"
