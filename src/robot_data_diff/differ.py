"""Episode-aware diff between two DatasetViews."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .loader import DatasetView
from .stats import STAT_KEYS, fingerprint, rounded

# Bump when a key in DiffResult.to_dict() is renamed, removed or changes
# meaning. Adding keys does not bump it.
SCHEMA_VERSION = 1


@dataclass
class EpisodeDelta:
    index: int                    # episode index in B
    old_length: int
    new_length: int
    d_length: int
    tasks_changed: bool
    old_tasks: list = field(default_factory=list)
    new_tasks: list = field(default_factory=list)
    # feature -> {stat: per-dim new - old}; only stats that changed are kept.
    # A value of None means the stat can't be subtracted (shape changed, or
    # missing on one side).
    feature_deltas: dict = field(default_factory=dict)
    d_action_min: list | None = None   # per-dim new - old
    d_action_max: list | None = None
    d_action_mean: list | None = None
    d_idle_ratio: float | None = None
    d_spike_count: int | None = None
    old_index: int | None = None  # episode index in A (differs only if moved)


@dataclass
class DiffResult:
    a_name: str
    b_name: str
    a_version: str
    b_version: str
    a_fps: float | None
    b_fps: float | None
    a_episodes: int
    b_episodes: int
    added: list[int] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    changed: list[EpisodeDelta] = field(default_factory=list)
    unchanged: int = 0
    # (index in A, index in B) for identical content at a new index;
    # only filled with match_by="content"
    moved: list = field(default_factory=list)
    match_by: str = "index"
    task_counts: dict = field(default_factory=dict)  # task -> (count_a, count_b)
    # (field, value in A, value in B) for fps / robot_type / codebase_version
    dataset_changes: list = field(default_factory=list)
    a_robot_type: str | None = None
    b_robot_type: str | None = None
    features_added: list = field(default_factory=list)
    features_removed: list = field(default_factory=list)
    features_changed: list = field(default_factory=list)  # "f: {..} -> {..}"
    # (feature, spec in A, spec in B) for each entry of features_changed
    features_changed_specs: list = field(default_factory=list)
    a_stats_verified: bool = False
    b_stats_verified: bool = False
    a_stats_mismatches: list = field(default_factory=list)
    b_stats_mismatches: list = field(default_factory=list)
    mean_idle_a: float | None = None
    mean_idle_b: float | None = None
    spikes_a: int | None = None
    spikes_b: int | None = None
    checkpoint: str | None = None
    a_lengths: list = field(default_factory=list)  # every episode's length
    b_lengths: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-ready dict with a stable layout (see docs/json-schema.md)."""
        from . import __version__

        def side(x):
            lengths = getattr(self, f"{x}_lengths")
            return {
                "name": getattr(self, f"{x}_name"),
                "codebase_version": getattr(self, f"{x}_version"),
                "fps": getattr(self, f"{x}_fps"),
                "robot_type": getattr(self, f"{x}_robot_type"),
                "episodes": getattr(self, f"{x}_episodes"),
                "frames": sum(lengths),
                "mean_idle_ratio": getattr(self, f"mean_idle_{x}"),
                "total_spikes": getattr(self, f"spikes_{x}"),
                "stats_verified": getattr(self, f"{x}_stats_verified"),
                "stats_mismatches": getattr(self, f"{x}_stats_mismatches"),
            }

        out = {
            "schema_version": SCHEMA_VERSION,
            "tool": {"name": "robot-data-diff", "version": __version__},
            "checkpoint": self.checkpoint,
            "match_by": self.match_by,
            "a": side("a"),
            "b": side("b"),
            "summary": {"added": len(self.added), "removed": len(self.removed),
                        "changed": len(self.changed), "moved": len(self.moved),
                        "unchanged": self.unchanged},
            "episodes": {
                "added": self.added,
                "removed": self.removed,
                "moved": [{"from": ia, "to": ib} for ia, ib in self.moved],
                "changed": [{
                    "index": d.index,
                    "old_index": d.old_index if d.old_index is not None else d.index,
                    "old_length": d.old_length, "new_length": d.new_length,
                    "d_length": d.d_length,
                    "tasks_changed": d.tasks_changed,
                    "old_tasks": d.old_tasks, "new_tasks": d.new_tasks,
                    "feature_deltas": d.feature_deltas,
                    "d_idle_ratio": d.d_idle_ratio,
                    "d_spike_count": d.d_spike_count,
                } for d in self.changed],
            },
            "tasks": [{"task": t, "a": ca, "b": cb}
                      for t, (ca, cb) in self.task_counts.items()],
            "dataset_changes": [{"field": n, "a": va, "b": vb}
                                for n, va, vb in self.dataset_changes],
            "schema": {
                "added": self.features_added,
                "removed": self.features_removed,
                "changed": [{"feature": f, "a": sa, "b": sb}
                            for f, sa, sb in self.features_changed_specs],
            },
        }
        return _json_safe(out)


def _json_safe(x):
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _sub(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    return [round(float(x) - float(y), 6) for x, y in zip(b, a, strict=True)]


def _stats_features(ds: DatasetView) -> set:
    return {f for ep in ds.episodes.values() for f in ep.stats}


def _feature_deltas(ea, eb, features) -> dict:
    out = {}
    for f in features:
        fd = {}
        for k in STAT_KEYS:
            va, vb = ea.stat(f, k), eb.stat(f, k)
            if rounded(va) != rounded(vb):
                fd[k] = _sub(va, vb)
        if fd:
            out[f] = fd
    return out


def _pair_by_content(a: DatasetView, b: DatasetView) -> list[tuple[int, int]]:
    """Pair A and B episodes: same index + same content first, then same
    content at a different index (moved), then leftovers sharing an index
    (edited in place). Unpaired episodes are added / removed."""
    a_left, b_left = set(a.episodes), set(b.episodes)
    pairs = []
    for idx in sorted(a_left & b_left):
        h = a.episodes[idx].content_hash
        if h is not None and h == b.episodes[idx].content_hash:
            pairs.append((idx, idx))
    a_left -= {p[0] for p in pairs}
    b_left -= {p[1] for p in pairs}

    by_hash: dict[str, list[int]] = {}
    for ia in sorted(a_left):
        h = a.episodes[ia].content_hash
        if h is not None:
            by_hash.setdefault(h, []).append(ia)
    for ib in sorted(b_left):
        candidates = by_hash.get(b.episodes[ib].content_hash)
        if candidates:
            ia = candidates.pop(0)
            pairs.append((ia, ib))
            a_left.discard(ia)
            b_left.discard(ib)

    pairs.extend((idx, idx) for idx in sorted(a_left & b_left))
    return sorted(pairs, key=lambda p: p[1])


def diff_datasets(a: DatasetView, b: DatasetView,
                  checkpoint: str | None = None,
                  match_by: str = "index") -> DiffResult:
    """Diff two dataset views.

    match_by="index" (default) pairs episodes by episode_index.
    match_by="content" pairs them by a hash of their action/state frames, so
    reordered episodes are reported as moved instead of added + removed;
    both views must be loaded with load_dataset(..., content_hash=True).
    """
    if match_by not in ("index", "content"):
        raise ValueError(f"match_by must be 'index' or 'content', not {match_by!r}")
    if match_by == "content":
        for ds in (a, b):
            if any(e.content_hash is None for e in ds.episodes.values()):
                raise ValueError(f"match_by='content' needs {ds.name} loaded with "
                                 "content_hash=True (and its data/ files present)")
    res = DiffResult(
        a_name=a.name, b_name=b.name,
        a_version=a.version, b_version=b.version,
        a_fps=a.fps, b_fps=b.fps,
        a_episodes=len(a.episodes), b_episodes=len(b.episodes),
        a_robot_type=a.robot_type, b_robot_type=b.robot_type,
        a_stats_verified=a.stats_verified, b_stats_verified=b.stats_verified,
        a_stats_mismatches=list(a.stats_mismatches),
        b_stats_mismatches=list(b.stats_mismatches),
        checkpoint=checkpoint, match_by=match_by,
    )
    for name, va, vb in (("fps", a.fps, b.fps),
                         ("robot_type", a.robot_type, b.robot_type),
                         ("codebase_version", a.version, b.version)):
        if va != vb:
            res.dataset_changes.append((name, va, vb))
    if match_by == "content":
        pairs = _pair_by_content(a, b)
    else:
        pairs = [(i, i) for i in sorted(set(a.episodes) & set(b.episodes))]
    res.added = sorted(set(b.episodes) - {p[1] for p in pairs})
    res.removed = sorted(set(a.episodes) - {p[0] for p in pairs})
    res.moved = [(ia, ib) for ia, ib in pairs if ia != ib]

    # only compare per-feature stats that both revisions have
    common = sorted(_stats_features(a) & _stats_features(b))

    for ia, ib in pairs:
        ea, eb = a.episodes[ia], b.episodes[ib]
        if fingerprint(ea, common) == fingerprint(eb, common):
            if ia == ib:
                res.unchanged += 1
            continue
        res.changed.append(EpisodeDelta(
            index=ib, old_index=ia,
            old_length=ea.length, new_length=eb.length,
            d_length=eb.length - ea.length,
            tasks_changed=sorted(ea.tasks) != sorted(eb.tasks),
            old_tasks=ea.tasks, new_tasks=eb.tasks,
            feature_deltas=_feature_deltas(ea, eb, common),
            d_action_min=_sub(ea.action_min, eb.action_min),
            d_action_max=_sub(ea.action_max, eb.action_max),
            d_action_mean=_sub(ea.action_mean, eb.action_mean),
            d_idle_ratio=(round(eb.idle_ratio - ea.idle_ratio, 4)
                          if ea.idle_ratio is not None and eb.idle_ratio is not None
                          else None),
            d_spike_count=(eb.spike_count - ea.spike_count
                           if ea.spike_count is not None and eb.spike_count is not None
                           else None),
        ))

    # task inventory: task string -> (count in A, count in B)
    def task_hist(ds: DatasetView) -> dict:
        h: dict[str, int] = {}
        for ep in ds.episodes.values():
            for t in ep.tasks or ["<no task>"]:
                h[t] = h.get(t, 0) + 1
        return h

    ha, hb = task_hist(a), task_hist(b)
    for t in sorted(set(ha) | set(hb)):
        res.task_counts[t] = (ha.get(t, 0), hb.get(t, 0))

    # schema drift
    fa, fb = set(a.features), set(b.features)
    res.features_added = sorted(fb - fa)
    res.features_removed = sorted(fa - fb)
    for f in sorted(fa & fb):
        if a.features[f] != b.features[f]:
            res.features_changed.append(
                f"{f}: {a.features[f]} -> {b.features[f]}")
            res.features_changed_specs.append((f, a.features[f], b.features[f]))

    # quality summary
    def qmean(ds: DatasetView):
        vals = [e.idle_ratio for e in ds.episodes.values()
                if e.idle_ratio is not None]
        sp = [e.spike_count for e in ds.episodes.values()
              if e.spike_count is not None]
        return (round(sum(vals) / len(vals), 4) if vals else None,
                sum(sp) if sp else None)

    res.mean_idle_a, res.spikes_a = qmean(a)
    res.mean_idle_b, res.spikes_b = qmean(b)
    res.a_lengths = [e.length for e in a.episodes.values()]
    res.b_lengths = [e.length for e in b.episodes.values()]
    return res
