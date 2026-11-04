"""Builders for tiny synthetic LeRobot datasets (v2.1 and v3.0) used by tests.

Everything is written to a caller-supplied directory (normally pytest's
tmp_path), needs no network, and is a few KB on disk. Layouts follow the
official huggingface/lerobot formats:

v2.1  meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl,
      meta/episodes_stats.jsonl  ({"episode_index": i, "stats": {feat: {...}}}),
      data/chunk-000/episode_000000.parquet  (one file per episode)
v3.0  meta/info.json, meta/tasks.parquet (task string is the index),
      meta/episodes/chunk-000/file-000.parquet  (stats/{feat}/{stat} columns),
      data/chunk-000/file-000.parquet  (all frames)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_TASK = "pick cube"
BOOKKEEPING = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def make_episodes(n: int = 4, T: int = 30, D: int = 2, seed: int = 0,
                  task: str = DEFAULT_TASK) -> list[dict]:
    """Random-walk action/state episodes: [{"index", "action", "state", "tasks"}].

    An optional "extra" dict {feature: (T, d) array} adds more numeric columns.
    """
    rng = np.random.default_rng(seed)
    return [
        {
            "index": i,
            "action": np.cumsum(rng.normal(0, 1, (T, D)), axis=0).astype(np.float32),
            "state": np.cumsum(rng.normal(0, 0.5, (T, D)), axis=0).astype(np.float32),
            "tasks": [task],
        }
        for i in range(n)
    ]


def clone(episodes: list[dict]) -> list[dict]:
    return copy.deepcopy(episodes)


def _features(D: int, extra: dict | None = None, drop: tuple = ()) -> dict:
    feats = {
        "action": {"dtype": "float32", "shape": [D]},
        "observation.state": {"dtype": "float32", "shape": [D]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    feats.update(extra or {})
    for k in drop:
        feats.pop(k, None)
    return feats


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "count": [len(x)]}


def _frames(episodes: list[dict], fps: float, task_ids: dict):
    """Per-episode frame columns plus the bookkeeping columns LeRobot adds."""
    gidx = 0
    for ep in episodes:
        T = len(ep["action"])
        cols = {
            "action": np.asarray(ep["action"], dtype=np.float32),
            "observation.state": np.asarray(ep["state"], dtype=np.float32),
            "timestamp": (np.arange(T) / fps).astype(np.float32),
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, ep["index"], dtype=np.int64),
            "index": np.arange(gidx, gidx + T, dtype=np.int64),
            "task_index": np.full(T, task_ids[ep["tasks"][0]], dtype=np.int64),
        }
        for k, v in ep.get("extra", {}).items():  # extra numeric columns
            cols[k] = np.asarray(v, dtype=np.float32)
        gidx += T
        yield ep, cols


def _vec_array(x: np.ndarray, fixed: bool) -> pa.Array:
    x = np.asarray(x, dtype=np.float32)
    flat = pa.array(x.ravel(), type=pa.float32())
    if fixed:
        return pa.FixedSizeListArray.from_arrays(flat, x.shape[1])
    offsets = pa.array(np.arange(0, x.size + 1, x.shape[1], dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, flat)


def _frame_table(cols: dict, fixed: bool) -> pa.Table:
    return pa.table({
        k: (_vec_array(v, fixed) if v.ndim == 2 else pa.array(v))
        for k, v in cols.items()
    })


def _info(version, episodes, fps, robot_type, features):
    return {
        "codebase_version": version,
        "robot_type": robot_type,
        "total_episodes": len(episodes),
        "total_frames": int(sum(len(e["action"]) for e in episodes)),
        "fps": fps,
        "features": features,
    }


def _task_ids(episodes):
    ids: dict[str, int] = {}
    for ep in episodes:
        for t in ep["tasks"]:
            ids.setdefault(t, len(ids))
    return ids


def write_v21(root: Path, episodes: list[dict], *, fps: float = 10,
              robot_type: str = "toy", stats_shape: str = "nested",
              extra_features: dict | None = None,
              drop_features: tuple = ()) -> Path:
    """Write a v2.1 dataset. stats_shape: "nested" (real) or "toplevel" (legacy)."""
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    D = np.asarray(episodes[0]["action"]).shape[1]
    feats = _features(D, extra_features, drop_features)
    (root / "meta" / "info.json").write_text(json.dumps(
        _info("v2.1", episodes, fps, robot_type, feats), indent=2))
    tids = _task_ids(episodes)
    (root / "meta" / "tasks.jsonl").write_text("".join(
        json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tids.items()))

    ep_lines, stat_lines = [], []
    for ep, cols in _frames(episodes, fps, tids):
        pq.write_table(_frame_table(cols, fixed=False),
                       root / "data" / "chunk-000" / f"episode_{ep['index']:06d}.parquet")
        ep_lines.append({"episode_index": ep["index"], "tasks": ep["tasks"],
                         "length": len(ep["action"])})
        st = {k: _stats(v) for k, v in cols.items()}
        if stats_shape == "nested":
            stat_lines.append({"episode_index": ep["index"], "stats": st})
        else:
            # flat layout can't hold the "episode_index" feature's stats (the
            # key would collide), which is why LeRobot nests under "stats"
            flat = {k: v for k, v in st.items() if k not in BOOKKEEPING}
            stat_lines.append({"episode_index": ep["index"], **flat})
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ep_lines))
    (root / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in stat_lines))
    return root


def write_v30(root: Path, episodes: list[dict], *, fps: float = 10,
              robot_type: str = "toy", extra_features: dict | None = None,
              drop_features: tuple = (), stats_override: dict | None = None) -> Path:
    """Write a v3.0 dataset (single episodes file + single data file).

    stats_override: {episode_index: {"stats/action/min": [...], ...}} to make
    the metadata disagree with the data (simulates stale meta/episodes).
    """
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    D = np.asarray(episodes[0]["action"]).shape[1]
    feats = _features(D, extra_features, drop_features)
    (root / "meta" / "info.json").write_text(json.dumps(
        _info("v3.0", episodes, fps, robot_type, feats), indent=2))
    tids = _task_ids(episodes)
    pd.DataFrame({"task_index": list(tids.values())},
                 index=pd.Index(list(tids.keys()), name="task")
                 ).to_parquet(root / "meta" / "tasks.parquet")

    frame_tables, rows = [], []
    for ep, cols in _frames(episodes, fps, tids):
        frame_tables.append(_frame_table(cols, fixed=True))
        row = {"episode_index": ep["index"], "tasks": ep["tasks"],
               "length": len(ep["action"]),
               "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": int(cols["index"][0]),
               "dataset_to_index": int(cols["index"][-1]) + 1}
        for feat, v in cols.items():
            for stat, val in _stats(v).items():
                row[f"stats/{feat}/{stat}"] = val
        row.update((stats_override or {}).get(ep["index"], {}))
        rows.append(row)
    pq.write_table(pa.concat_tables(frame_tables),
                   root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root
