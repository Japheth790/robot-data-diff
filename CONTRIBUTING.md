# Contributing to robot-data-diff

Thanks for helping! Bug reports, docs fixes and PRs are all welcome.

## Ground rules

- **CPU-only, pure Python.** Runtime dependencies are `pyarrow`, `pandas`,
  `numpy` and `huggingface_hub`; please don't add heavy ones.
- **Never download or decode videos**, and never require a GPU or an account.
- **Don't break the default CLI behaviour.** New behaviour goes behind a flag.
- The JSON output (`docs/json-schema.md`) is an interface: adding keys is
  fine; renaming or removing one means bumping `SCHEMA_VERSION`.
- Small, reviewable PRs with one logical change each.

## Development setup

```bash
git clone https://github.com/Japheth790/robot-data-diff
cd robot-data-diff
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running checks

```bash
ruff check .                     # lint (CI runs this)
pytest                           # unit tests, no network (CI runs this)
pytest -m network                # also hits the Hugging Face Hub (lerobot/pusht)
python benchmarks/bench_load.py  # load-time benchmark, 5,000 synthetic episodes
```

Tests build tiny synthetic LeRobot datasets in `tmp_path` with the helpers
in [`tests/lerobot_fixtures.py`](tests/lerobot_fixtures.py) (`make_episodes`,
`write_v21`, `write_v30`). The `write` fixture in `tests/conftest.py` runs a
test once per format. A new test usually looks like this:

```python
def test_something(write, episodes):
    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 2
    r = diff_datasets(load_dataset(str(write("a", episodes))),
                      load_dataset(str(write("b", b))))
    assert [d.index for d in r.changed] == [0]
```

For a bug fix, please add a test that fails without the fix.

## Code map

| File | What's in it |
|---|---|
| `src/robot_data_diff/loader.py` | resolve local path / HF id, read v2.1 + v3.0 metadata and frame data |
| `src/robot_data_diff/stats.py` | `action_quality` heuristics and the episode fingerprint |
| `src/robot_data_diff/differ.py` | `diff_datasets`, `DiffResult`, `to_dict()` |
| `src/robot_data_diff/gate.py` | `--fail-on` rules |
| `src/robot_data_diff/filters.py` | `--episodes` / `--task` |
| `src/robot_data_diff/report.py` | text and Markdown renderers |
| `src/robot_data_diff/html_report.py` | the self-contained HTML report |
| `src/robot_data_diff/cli.py` | argument parsing, exit codes |

## Reporting bugs

Please include the dataset id(s) and revisions (or how to build a small
dataset that shows the problem), your robot-data-diff version
(`robot-data-diff --version`), and the exact command. Add `--verbose` to get
the full traceback. The issue template asks for all of this.

## Good first issues

Ideas that are well-scoped and don't need deep knowledge of the code:

- **Index ranges in text output**: print `removed: 200-205` instead of
  `[200, 201, 202, 203, 204, 205]` (behind a flag, or in Markdown only).
- **`--fail-on frames>5%`**: fail when the total frame count changed by
  more than a threshold.
- **Per-feature summary**: a "most-changed features" table (how many
  episodes changed per feature) in the text and HTML reports.
- **`--ignore-feature NAME`**: leave a feature out of change detection
  (e.g. a noisy `next.reward`).
- **Quantile stats**: newer LeRobot versions write `q01`/`q99` per-episode
  stats; compare them when both sides have them.
- **Markdown file output**: `--markdown OUT`, like `--html` and `--json`.
- **Progress for large downloads**: show which side is being fetched and
  how many files, without the per-file noise.

## Releasing (maintainers)

1. Update the version in `pyproject.toml` and move "Unreleased" in
   `CHANGELOG.md` under the new version.
2. Tag: `git tag v0.X.Y && git push origin v0.X.Y`.
3. `.github/workflows/release.yml` builds and publishes to PyPI via trusted
   publishing (one-time setup on pypi.org: add this repo, workflow
   `release.yml`, environment `pypi`).
