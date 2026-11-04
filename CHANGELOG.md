# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-03

### Added

- `--format json` and `--json OUT`: machine-readable output with a documented,
  stable layout (`"schema_version": 1`, see `docs/json-schema.md`).
- `--fail-on RULES` CI gate, e.g. `schema,fps,removed,changed>5%,idle-increase>0.05`.
  Exit codes: 0 OK, 1 a rule hit, 2 error.
- `--format markdown` for GitHub PR comments and Hugging Face discussions.
- `--verify-stats`: recompute per-episode length and min/max/mean/std from
  `data/` and flag stale metadata.
- `--match-by content`: pair episodes by a hash of their `action` +
  `observation.state` frames and report reordered episodes as *moved*.
- `--episodes 0-50,99`, `--task TEXT` and `--max-show N`.
- `--verbose` (full tracebacks) and `--version`.
- A "DATASET-LEVEL CHANGES" section for `fps`, `robot_type` and
  `codebase_version`.
- Python API: `load_dataset`, `diff_datasets`, `DiffResult` (with
  `to_dict()`), `render_text`, `render_markdown`, `render_html` and
  `DatasetError` are exported from `robot_data_diff`.
- Test suite with synthetic v2.1 and v3.0 fixtures, network tests against
  the real `lerobot/pusht@v2.1` and `@v3.0` (`pytest -m network`), GitHub
  Actions CI (ruff + pytest on Python 3.10-3.13, Linux/macOS/Windows) and a
  PyPI trusted-publishing release workflow.
- `benchmarks/bench_load.py`, CONTRIBUTING.md, issue and PR templates.

### Changed

- **Every numeric feature is compared, not only `action`.** Episodes whose
  `observation.state` (or reward, velocity, ...) stats changed were
  previously reported as unchanged; they now count as changed. Only
  features present on both sides are compared, and LeRobot's bookkeeping
  columns (`index`, `frame_index`, `timestamp`, `episode_index`,
  `task_index`) are ignored.
- The text report lists only the stats that changed for each episode
  (including `std`), instead of always printing action min/max/mean.
- The HTML report is redesigned: summary cards, a sortable changed-episodes
  table, colour-coded deltas, an episode-length histogram and dark mode.
  It now shows everything the text report shows (it was missing the
  action-mean delta and fps). Still one self-contained file.
- Errors print one line (`error: ...`) and exit with code 2 instead of a
  Python traceback. Mistyped local paths are no longer sent to the Hub.
- `DatasetView.version` holds the real `codebase_version` from
  `meta/info.json` (previously hard-coded to `v3.0`/`v2.1`).
- `no_proxy`/`NO_PROXY` are no longer removed from the environment unless
  `RDD_STRIP_NO_PROXY=1` is set.
- Frame data is read column-wise with pyarrow + numpy: loading a synthetic
  5,000-episode dataset went from 1.70 s to 0.44 s (v3.0) and 6.40 s to
  3.48 s (v2.1).
- `__version__` is read from the package metadata.

### Fixed

- Installing from source: the modules now live in `src/robot_data_diff/`,
  so `pip install .` gives a working `robot-data-diff` command.
- `--no-quality` now skips reading (and, for Hugging Face refs, downloading)
  `data/`. It used to read every data file and throw the result away.
- Redirected output on Windows (`... > report.md`) no longer crashes on
  non-ASCII characters; stdout is always UTF-8. HTML/JSON files are written
  as UTF-8 on every platform.

### Removed

- The committed wheel and sdist (release artifacts now come from the
  release workflow).

## [0.1.0] - 2026-10-12

### Added

- First release: `robot-data-diff diff A B` for LeRobot v2.1 and v3.0
  datasets (local paths or Hugging Face ids), with episode inventory,
  per-episode action stat deltas, task inventory, schema drift, idle/spike
  quality heuristics and an HTML report.

[Unreleased]: https://github.com/yafetashenafi790-lgtm/robot-data-diff/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yafetashenafi790-lgtm/robot-data-diff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yafetashenafi790-lgtm/robot-data-diff/releases/tag/v0.1.0
