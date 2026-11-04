# robot-data-diff

[![CI](https://github.com/Japheth790/robot-data-diff/actions/workflows/ci.yml/badge.svg)](https://github.com/Japheth790/robot-data-diff/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/robot-data-diff)](https://pypi.org/project/robot-data-diff/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**You retrained the policy and it got worse. `robot-data-diff` tells you what
changed in the LeRobot dataset since the last good run, episode by episode.**

CPU-only, pure Python, no GPU, no account, and it never downloads videos.

```bash
pip install robot-data-diff
robot-data-diff diff lerobot/pusht@v2.1 lerobot/pusht@v3.0
```

```text
========================================================================
robot-data-diff  —  episode-aware dataset diff
========================================================================
A: lerobot/pusht@v2.1   (v2.1, 206 episodes, 10 fps)
B: lerobot/pusht@v3.0   (v3.0, 206 episodes, 10 fps)

Episodes: 206 -> 206 (+0 added, -0 removed, ~0 changed, =206 unchanged)

No changed episodes (same index + identical stats).

TASK INVENTORY (episodes per task: A -> B):
    Push the T-shaped block onto the T-shaped target.                206 ->   206

DATASET-LEVEL CHANGES:
  ! codebase_version: v2.1 -> v3.0

Schema: identical feature sets.

QUALITY SUMMARY (heuristic, CPU-only):
  mean idle-frame ratio: 0.0309 -> 0.0309 (+0.0000)
  total action spikes:   25 -> 25 (+0)
========================================================================
```

That's the real output: converting pusht from v2.1 to v3.0 changed the
format, not the data. When episodes *do* change, you get per-episode deltas
like this (from [`demo/diff-demo-output.txt`](demo/diff-demo-output.txt)):

```text
Episodes: 206 -> 201 (+1 added, -6 removed, ~5 changed, =195 unchanged)
  added:   [206]
  removed: [200, 201, 202, 203, 204, 205]

CHANGED EPISODES (5):
  [ep 0] length 161 -> 161 (+0 frames)
           d(action_min)  [+46.5000, +35.5000]
           d(action_max)  [+187.5000, +224.5000]
           d(action_mean) [+104.5031, +144.1553]
           d(action_std)  [+44.1925, +48.3483]
           d(idle_ratio)  +0.0000
           d(spikes)      +0
```

## Install

```bash
pip install robot-data-diff
```

Python 3.10+. Dependencies: `pyarrow`, `pandas`, `numpy`, `huggingface_hub`.

## Usage

```bash
# two local dataset directories
robot-data-diff diff ./my_dataset_v1 ./my_dataset_v2

# Hugging Face ids, optionally pinned to a revision (branch, tag or commit)
robot-data-diff diff lerobot/pusht ./my_local_copy
robot-data-diff diff lerobot/pusht@main lerobot/pusht@<commit-sha>

# link the diff to a policy checkpoint and write an HTML report
robot-data-diff diff v1 v2 --checkpoint ckpt-42 --html report.html

# metadata only: skips (and doesn't download) data/; fastest
robot-data-diff diff v1 v2 --no-quality
```

Each side is a local path or a Hugging Face dataset id (`org/repo`,
`org/repo@revision`, or `hf://org/repo@revision`).

| Option | What it does |
|---|---|
| `--format text\|json\|markdown` | what to print on stdout (default `text`) |
| `--html OUT` / `--json OUT` | also write an HTML report / the JSON result to a file |
| `--checkpoint LABEL` | record which policy checkpoint this diff belongs to |
| `--no-quality` | skip the idle/spike heuristics and never read `data/` |
| `--verify-stats` | recompute per-episode length and min/max/mean/std from `data/` and flag stale metadata |
| `--match-by index\|content` | pair episodes by index (default) or by a hash of their frames, which reports reordered episodes as *moved* |
| `--episodes 0-50,99` | only diff these episode indices |
| `--task "pick cube"` | only diff episodes whose task (in A or B) contains the text; repeatable |
| `--max-show N` | truncate long episode lists with "... and K more" |
| `--fail-on RULES` | exit 1 when a rule hits (see [Use in CI](#use-in-ci)) |
| `--cache-dir DIR` | where to keep Hugging Face downloads |
| `--verbose` | show full tracebacks on errors |

## What the report shows

- **Episode inventory:** counts, plus episodes added / removed / changed (and
  moved, with `--match-by content`)
- **Per-episode deltas** for changed episodes: frame count, task strings,
  and per-dimension min/max/mean/std deltas for *every* numeric feature
  that changed (`action`, `observation.state`, rewards, ...), plus
  idle-ratio and spike-count deltas
- **Task inventory:** episodes per task, old → new
- **Dataset-level changes:** `fps`, `robot_type`, `codebase_version`
- **Schema drift:** features added / removed / changed dtype or shape
- **Stale metadata** (with `--verify-stats`): metadata stats that disagree
  with the frames in `data/`
- **Quality summary:** mean idle-frame ratio and total action spikes,
  old → new (lightweight CPU heuristics, not a full validator)

The HTML report ([example](demo/diff-demo-report.html)) is one
self-contained file: summary cards, a sortable changed-episodes table,
colour-coded deltas, an episode-length histogram for A vs B, and dark mode.

## How "changed" is detected

An episode is **changed** when the same episode exists on both sides but
its fingerprint differs. The fingerprint is:

- length (frame count) and the sorted task strings
- min / max / mean / std of every feature in the per-episode stats
  (`meta/episodes/` in v3.0, `meta/episodes_stats.jsonl` in v2.1),
  rounded to 6 decimals so float noise isn't reported
- idle ratio and spike count computed from `action` in `data/` (unless
  `--no-quality`)

Some things are deliberately left out:

- LeRobot's bookkeeping columns (`index`, `frame_index`, `timestamp`,
  `episode_index`, `task_index`). Their stats only describe where an
  episode sits in the dataset; removing episode 3 shifts the global
  `index` of every later episode without changing their data.
- Features that exist on only one side. A feature added to the whole
  dataset is schema drift, not 2,000 changed episodes.

By default "the same episode" means the same `episode_index`. With
`--match-by content`, episodes are first paired by a SHA-1 of their raw
`action` + `observation.state` frames, so data that was only reindexed is
reported as `moved: 12 -> 40`. Episodes that weren't matched by content are
then paired by index (edited in place → changed), and anything left over
is added or removed.

Per-episode stats come from the dataset's metadata. If someone edited
`data/` without regenerating the metadata, run with `--verify-stats` to
catch it.

## Use in CI

`--fail-on` turns the diff into a gate. Exit codes: **0** OK, **1** a rule
hit, **2** error (bad path, not a LeRobot dataset, bad rule, ...).

```bash
robot-data-diff diff my-org/teleop-data@main my-org/teleop-data@candidate \
    --fail-on "schema,fps,removed,changed>5%,idle-increase>0.05"
```

| Rule | Fails when |
|---|---|
| `schema` | any feature added, removed, or changed dtype/shape |
| `fps`, `robot_type`, `codebase_version` | that field of `meta/info.json` changed |
| `dataset` | any of the three above changed |
| `tasks` | the number of episodes per task changed |
| `stale` | `--verify-stats` found metadata that disagrees with `data/` |
| `added`, `removed`, `changed`, `moved` | the count is > 0; or use `>N`, `>=N`, `>N%` (percent of A's episodes) |
| `idle-increase>X` | mean idle-frame ratio rose by more than X |
| `spikes-increase>N` | total action spikes rose by more than N |

Each hit is printed to stderr as `fail-on: changed>5%: 12 changed = 6.0% of
200 episodes in A`, and recorded under `"fail_on"` in the JSON output.

A GitHub Actions job that gates a dataset revision and writes the diff to
the job summary:

```yaml
name: dataset-diff
on: workflow_dispatch

jobs:
  diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
      - run: pip install robot-data-diff
      - name: Diff dataset revisions
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}   # only needed for private datasets
        run: |
          robot-data-diff diff my-org/teleop-data@main my-org/teleop-data@candidate \
            --format markdown --max-show 50 --json diff.json \
            --fail-on "schema,fps,removed,changed>5%" >> "$GITHUB_STEP_SUMMARY"
      - uses: actions/upload-artifact@v7
        if: always()
        with:
          name: dataset-diff
          path: diff.json
```

`--format markdown` output can also be pasted straight into a PR comment or
a Hugging Face discussion.

## Use from Python

```python
import robot_data_diff as rdd

a = rdd.load_dataset("lerobot/pusht@v2.1")
b = rdd.load_dataset("lerobot/pusht@v3.0", verify_stats=True)
result = rdd.diff_datasets(a, b, checkpoint="ckpt-42")

print(rdd.render_text(result))
print(len(result.changed), "changed episodes")
for d in result.changed:
    print(d.index, d.d_length, d.feature_deltas.get("action"))

data = result.to_dict()          # JSON-ready, "schema_version": 1
html = rdd.render_html(result)   # or rdd.render_markdown(result)
```

`load_dataset(ref, cache_dir=None, with_quality=True, verify_stats=False,
content_hash=False)` returns a `DatasetView`. `diff_datasets(a, b,
checkpoint=None, match_by="index")` returns a `DiffResult`; pass
`match_by="content"` with both views loaded using `content_hash=True`.
Loader errors raise `DatasetError` (a `ValueError`). The JSON layout is
documented in [docs/json-schema.md](docs/json-schema.md).

## Supported formats

- **LeRobot v3.0**: `meta/info.json`, `meta/episodes/chunk-*/file-*.parquet`,
  `data/chunk-*/file-*.parquet`.
- **LeRobot v2.1** (and v2.0): `meta/info.json`, `meta/episodes.jsonl`,
  `meta/episodes_stats.jsonl`, `data/chunk-*/episode_*.parquet`.

Both formats are tested with synthetic fixtures in CI, and against the real
`lerobot/pusht@v2.1` and `@v3.0` releases in the network tests
(`pytest -m network`). You can diff across formats, e.g. to check that a
v2.1 → v3.0 conversion didn't change any data.

RLDS, Zarr, HDF5 and rosbag are out of scope (LeRobot only).

## FAQ

**Why aren't videos downloaded?** Every number in the report comes from
`meta/` and the parquet files in `data/`. Videos are usually most of a
dataset's size, and decoding them would need a GPU or a lot of CPU time.
Per-episode image/video stats (when LeRobot recorded them in the metadata)
are still compared.

**How does caching work?** Without `--cache-dir`, downloads go to the
standard Hugging Face cache (`~/.cache/huggingface/hub`) and are reused on
the next run. With `--cache-dir DIR`, each `repo@revision` gets its own
subfolder, so two revisions never mix files. `--no-quality` downloads only
`meta/**`; a later full run fetches `data/**` on top.

**Private or gated datasets?** Log in with `huggingface-cli login` or set
`HF_TOKEN`; `huggingface_hub` picks it up.

**How fast is it?** On a synthetic 5,000-episode × 100-frame dataset
(`benchmarks/bench_load.py`), loading one side takes about 0.4 s for v3.0
and 3.5 s for v2.1 (one file per episode), or about 0.25 s for either with
`--no-quality`.

**My proxy's `no_proxy` breaks Hugging Face downloads.** Some proxy setups
put IPv6 entries in `no_proxy` that `httpx` can't parse. Set
`RDD_STRIP_NO_PROXY=1` and the tool drops `no_proxy`/`NO_PROXY` for its own
process.

## Honest limits

- This is a **diff**, not a validator: it tells you *what changed* between
  revisions, not whether your data is good.
- Change detection uses per-episode summary stats. Two episodes with the
  same length, tasks, and min/max/mean/std for every feature are
  "unchanged" even if individual frames differ. `--match-by content` hashes
  the raw `action` and `observation.state` frames, but only uses the hash
  for pairing.
- The idle/spike heuristics are deliberately simple; treat them as a
  prompt to look, not a verdict.

## Reproduce the demo

```bash
# a local copy of pusht v3.0 (meta + data only; `hf` ships with huggingface_hub >= 0.34)
hf download lerobot/pusht --repo-type dataset \
    --include "meta/*" --include "data/*" --local-dir testdata/pusht
python demo/make_modified_fixture.py             # builds testdata/pusht_v2
robot-data-diff diff testdata/pusht testdata/pusht_v2 \
    --checkpoint ckpt-42 --html demo/diff-demo-report.html
```

## Related work

- [openbot-data](https://github.com/openbotai/openbot-data) (MIT) is a
  broader robot-data toolkit whose `openbot-data diff` compares two
  *snapshot files* (`snapshot` first, then `diff`, with `--fail-on breaking`
  for CI). robot-data-diff is narrower on purpose: it diffs two dataset
  revisions *directly* (local dirs or Hugging Face ids, no snapshot step),
  episode by episode, with per-episode stat deltas.

## Contributing

Bug reports and PRs are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and
[CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0
