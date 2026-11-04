"""Per-episode statistics and lightweight quality heuristics (CPU-only, numpy)."""
from __future__ import annotations

import numpy as np

# per-feature stats compared between revisions
STAT_KEYS = ("min", "max", "mean", "std")


def action_quality(action: np.ndarray) -> tuple[float, int]:
    """Idle-frame ratio and action-spike count for one episode's actions.

    action: (T, D) float array of the episode's action vectors.
    Returns (idle_ratio, spike_count).
    - idle frame: max |delta| across dims below eps (eps scales with the
      episode's own action range, so it works for any robot/action units).
    - spike: a frame-to-frame jump far above the episode's typical motion.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        action = action[:, None]
    if len(action) < 2:
        return 0.0, 0
    deltas = np.abs(np.diff(action, axis=0))
    maxd = deltas.max(axis=1)
    overall_range = float(np.ptp(action))
    eps = max(1e-3 * overall_range, 1e-9)
    idle_ratio = float((maxd < eps).mean())
    med = float(np.median(maxd))
    spike_thresh = max(10.0 * med, 5.0 * eps)
    spike_count = int((maxd > spike_thresh).sum())
    return idle_ratio, spike_count


def _r6(x) -> float:
    return round(float(x), 6)


def rounded(vals) -> tuple | None:
    """Stats vector rounded to 6 decimals, so float noise is not a change."""
    return tuple(_r6(v) for v in vals) if vals is not None else None


def fingerprint(ep, features=None) -> tuple:
    """Hashable summary of an episode used to detect 'changed' episodes.

    Two episodes with the same index but different fingerprints changed
    between dataset revisions. `features` restricts the per-feature stats
    that are compared (the differ passes the features both sides have, so a
    feature added to the whole dataset is schema drift, not 2000 changes).
    """
    feats = sorted(ep.stats if features is None else features)
    return (
        ep.length,
        tuple(sorted(ep.tasks)),
        tuple((f, k, rounded(ep.stat(f, k))) for f in feats for k in STAT_KEYS),
        round(ep.idle_ratio, 4) if ep.idle_ratio is not None else None,
        ep.spike_count,
    )
