"""Load LeRobot datasets (v2.1 and v3.0) from a local path or a Hugging Face repo id.

v3.0 layout (current):
    meta/info.json                  codebase_version "v3.0", fps, features, totals
    meta/episodes/chunk-*/file-*.parquet   one row per episode; includes
        episode_index, tasks, length, and per-episode stats/action/min|max|mean
    meta/tasks.parquet              task_index -> task string (in the index)
    data/chunk-*/file-*.parquet     frames: episode_index, frame_index,
        timestamp, action, observation.state, ...

v2.1 layout (legacy, still common on the Hub):
    meta/info.json                  codebase_version "v2.1", fps, features
    meta/episodes.jsonl             episode_index, tasks, length (one JSON/line)
    meta/episodes_stats.jsonl       per-episode min/max/mean/std per feature
    meta/tasks.jsonl                task_index -> task
    data/chunk-*/episode_*.parquet  one parquet per episode with the frames

Videos are never read — everything here is metadata + data parquet only.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .stats import STAT_KEYS, action_quality

# Columns LeRobot adds to every dataset. Their stats describe position in the
# dataset (e.g. the global frame `index` shifts when an earlier episode is
# removed), not the recorded data, so they are left out of change detection.
BOOKKEEPING_FEATURES = frozenset(
    {"index", "episode_index", "frame_index", "timestamp", "task_index"})


@dataclass
class EpisodeInfo:
    index: int
    length: int
    tasks: list = field(default_factory=list)
    # per-feature stats from the dataset metadata: feature -> {stat: [floats]}
    stats: dict = field(default_factory=dict)
    idle_ratio: float | None = None
    spike_count: int | None = None

    def stat(self, feature: str, name: str) -> list | None:
        return self.stats.get(feature, {}).get(name)

    @property
    def action_min(self):
        return self.stat("action", "min")

    @property
    def action_max(self):
        return self.stat("action", "max")

    @property
    def action_mean(self):
        return self.stat("action", "mean")

    @property
    def state_min(self):
        return self.stat("observation.state", "min")

    @property
    def state_max(self):
        return self.stat("observation.state", "max")


@dataclass
class DatasetView:
    name: str
    version: str
    fps: float | None
    features: dict  # feature name -> {"dtype": str, "shape": list}
    episodes: dict  # episode_index -> EpisodeInfo
    root: Path
    robot_type: str | None = None


# ---------------------------------------------------------------------------
# reference resolution: local path or Hugging Face id
# ---------------------------------------------------------------------------

def _strip_no_proxy():
    # Opt-in workaround: some proxy setups put IPv6 entries in no_proxy that
    # httpx fails to parse. Only touch the environment when asked to.
    if os.environ.get("RDD_STRIP_NO_PROXY") != "1":
        return
    for k in ("no_proxy", "NO_PROXY"):
        os.environ.pop(k, None)


def _namespaced_cache_dir(cache_dir: str | Path, repo_id: str,
                          revision: str) -> Path:
    """Give each repo@revision its own subdirectory under cache_dir.

    snapshot_download() never evicts stale files from local_dir, so pointing
    two different repos/revisions at one flat directory silently mixes files
    and produces wrong diffs. Namespacing keeps every download isolated while
    still reusing the cache when the same ref is diffed again.
    """
    def safe(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", s)

    return (Path(cache_dir) / safe(repo_id.replace("/", "__"))
            / safe(revision or "main"))


def resolve_ref(ref: str, cache_dir: str | Path | None = None) -> Path:
    """Turn a dataset reference into a local directory Path.

    Accepts a local path, or a Hugging Face dataset id like
    "lerobot/pusht" or "lerobot/pusht@<revision>".  HF downloads fetch
    meta/** and data/** only — videos are skipped.  When cache_dir is given,
    each repo@revision gets its own subdirectory inside it, so diffs never
    read each other's files.
    """
    p = Path(ref)
    if p.exists():
        return p
    _strip_no_proxy()
    from huggingface_hub import snapshot_download

    repo_id, _, revision = ref.removeprefix("hf://").partition("@")
    if "/" not in repo_id:
        raise ValueError(
            f"'{ref}' is not a local path and not a Hugging Face id "
            "(expected like 'lerobot/pusht' or 'lerobot/pusht@<rev>')."
        )
    local_dir = (_namespaced_cache_dir(cache_dir, repo_id, revision)
                 if cache_dir is not None else None)
    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision or None,
        allow_patterns=["meta/**", "data/**"],
        local_dir=local_dir,
    )
    return Path(local)


def _as_list(v):
    if v is None:
        return None
    return [float(x) for x in np.asarray(v, dtype=np.float64).ravel()]


def _put_stat(stats: dict, feature: str, stat: str, value) -> None:
    """Store one numeric stat if it is one we track; skip anything else."""
    if feature in BOOKKEEPING_FEATURES or stat not in STAT_KEYS or value is None:
        return
    try:
        stats.setdefault(feature, {})[stat] = _as_list(value)
    except (TypeError, ValueError):
        pass  # non-numeric stat; ignore


# ---------------------------------------------------------------------------
# v3 loader
# ---------------------------------------------------------------------------

def _load_v3(root: Path, name: str) -> DatasetView:
    info = json.loads((root / "meta" / "info.json").read_text())
    features = {
        k: {"dtype": str(v.get("dtype")), "shape": list(v.get("shape") or [])}
        for k, v in (info.get("features") or {}).items()
    }
    fps = info.get("fps")

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise ValueError(f"no episode metadata found under {root}/meta/episodes")
    episodes: dict[int, EpisodeInfo] = {}
    for f in ep_files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        stat_cols = [c for c in names if c.startswith("stats/") and c.count("/") >= 2]
        base = [c for c in ("episode_index", "length", "tasks") if c in names]
        cols = pf.read(columns=base + stat_cols).to_pydict()
        n = len(cols["episode_index"])
        parsed = [(c, *c[len("stats/"):].rsplit("/", 1)) for c in stat_cols]
        for i in range(n):
            idx = int(cols["episode_index"][i])
            ep = EpisodeInfo(
                index=idx,
                length=int(cols["length"][i]),
                tasks=[str(t) for t in (cols.get("tasks", [None] * n)[i] or [])],
            )
            for col, feat, stat in parsed:
                _put_stat(ep.stats, feat, stat, cols[col][i])
            episodes[idx] = ep
    ds = DatasetView(name=name, version=str(info.get("codebase_version", "")),
                     fps=fps, features=features, episodes=episodes, root=root,
                     robot_type=info.get("robot_type"))
    _add_quality(ds, _scan_frames(_data_files(ds), ["action"]))
    return ds


# ---------------------------------------------------------------------------
# v2 loader (documented v2.1 layout)
# ---------------------------------------------------------------------------

def _load_v2(root: Path, name: str) -> DatasetView:
    info = json.loads((root / "meta" / "info.json").read_text())
    features = {
        k: {"dtype": str(v.get("dtype")), "shape": list(v.get("shape") or [])}
        for k, v in (info.get("features") or {}).items()
    }
    fps = info.get("fps")

    episodes: dict[int, EpisodeInfo] = {}
    ep_path = root / "meta" / "episodes.jsonl"
    for line in ep_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        idx = int(rec["episode_index"])
        episodes[idx] = EpisodeInfo(
            index=idx,
            length=int(rec.get("length", 0)),
            tasks=[str(t) for t in (rec.get("tasks") or [])],
        )

    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        for line in stats_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = episodes.get(int(rec["episode_index"]))
            if ep is None:
                continue
            # real v2.1 nests under "stats"; accept older flat files too
            feats = rec.get("stats") or rec.get("features") or rec
            for feat, fstats in feats.items():
                if isinstance(fstats, dict):
                    for stat, value in fstats.items():
                        _put_stat(ep.stats, feat, stat, value)

    ds = DatasetView(name=name, version=str(info.get("codebase_version", "")),
                     fps=fps, features=features, episodes=episodes, root=root,
                     robot_type=info.get("robot_type"))
    _add_quality(ds, _scan_frames(_data_files(ds), ["action"]))
    return ds


# ---------------------------------------------------------------------------
# frame data (data/**/*.parquet), read column-wise with pyarrow + numpy
# ---------------------------------------------------------------------------

def _data_files(ds: DatasetView) -> list[Path]:
    pattern = "episode_*.parquet" if ds.version.startswith("v2") else "*.parquet"
    return sorted((ds.root / "data").rglob(pattern))


def _v2_file_episode(f: Path) -> int | None:
    """Episode index from a v2 file name like episode_000123.parquet."""
    if not f.stem.startswith("episode_"):
        return None
    try:
        return int(f.stem.split("_")[-1])
    except ValueError:
        return None


def _is_numeric(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)


def _vector_column(col: pa.ChunkedArray) -> np.ndarray:
    """(N, D) array (native dtype) from a list, fixed-size-list or scalar column."""
    arr = col.combine_chunks()
    n = len(arr)
    if _is_numeric(arr.type):
        return arr.to_numpy(zero_copy_only=False).reshape(n, 1)
    t = arr.type
    if (pa.types.is_list(t) or pa.types.is_large_list(t)
            or pa.types.is_fixed_size_list(t)) and arr.null_count == 0 and n:
        flat = pc.list_flatten(arr)
        if _is_numeric(flat.type) and len(flat) % n == 0:
            lens = pc.list_value_length(arr).to_numpy(zero_copy_only=False)
            if (lens == len(flat) // n).all():
                return flat.to_numpy(zero_copy_only=False).reshape(n, -1)
    # ragged, nested or null-containing rows: slow but always correct
    return np.stack([np.asarray(v, dtype=np.float64).ravel() for v in arr.to_pylist()])


def _scan_frames(files: list[Path], columns: list[str]) -> dict[int, dict[str, np.ndarray]]:
    """Read `columns` from data files, grouped as {episode: {column: (T, D) array}}.

    Only the requested columns (plus episode_index when needed) are read.
    Frames keep their on-disk order; an episode split across files is joined
    in file order.
    """
    parts: dict[int, dict[str, list]] = {}
    for f in files:
        file_ep = _v2_file_episode(f)
        # an open OSFile skips pyarrow's per-path filesystem resolution, which
        # dominates when v2.1 datasets have thousands of small files
        with pa.OSFile(str(f)) as fh:
            pf = pq.ParquetFile(fh)
            names = set(pf.schema_arrow.names)
            cols = [c for c in columns if c in names]
            if not cols or (file_ep is None and "episode_index" not in names):
                continue
            table = pf.read(columns=cols + (["episode_index"] if file_ep is None else []))
        arrays = {c: _vector_column(table.column(c)) for c in cols}
        if file_ep is not None:
            groups = [(file_ep, slice(None))]
        else:
            ep = table.column("episode_index").to_numpy()
            if len(ep) and (np.diff(ep) < 0).any():
                order = np.argsort(ep, kind="stable")
                ep = ep[order]
                arrays = {c: a[order] for c, a in arrays.items()}
            starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]]) if len(ep) else []
            ends = list(starts[1:]) + [len(ep)]
            groups = [(int(ep[s]), slice(s, e)) for s, e in zip(starts, ends, strict=True)]
        for idx, sl in groups:
            d = parts.setdefault(idx, {})
            for c, a in arrays.items():
                d.setdefault(c, []).append(a[sl])
    return {idx: {c: v[0] if len(v) == 1 else np.concatenate(v) for c, v in d.items()}
            for idx, d in parts.items()}


def _add_quality(ds: DatasetView, frames: dict) -> None:
    """Idle-ratio / spike-count per episode from its action frames."""
    for idx, cols in frames.items():
        ep = ds.episodes.get(idx)
        if ep is not None and "action" in cols:
            ep.idle_ratio, ep.spike_count = action_quality(cols["action"])


# ---------------------------------------------------------------------------

def load_dataset(ref: str, cache_dir: str | Path | None = None,
                 with_quality: bool = True) -> DatasetView:
    """Load a dataset reference into a DatasetView (v2.1 or v3.0)."""
    root = resolve_ref(ref, cache_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError(f"{root} does not look like a LeRobot dataset "
                         "(missing meta/info.json).")
    version = str(json.loads(info_path.read_text()).get("codebase_version", ""))
    name = ref if isinstance(ref, str) else str(ref)
    if version.startswith("v3"):
        ds = _load_v3(root, name)
    elif version.startswith("v2"):
        ds = _load_v2(root, name)
    else:
        raise ValueError(f"unsupported LeRobot codebase_version: {version!r}")
    if not with_quality:
        for ep in ds.episodes.values():
            ep.idle_ratio, ep.spike_count = None, None
    return ds
