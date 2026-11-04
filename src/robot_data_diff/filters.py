"""--episodes / --task: restrict a diff to a subset of episodes."""
from __future__ import annotations

from dataclasses import replace

from .loader import DatasetView


def parse_episodes(spec: str) -> set[int]:
    """'0-50,99' -> {0, ..., 50, 99}; ranges are inclusive."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        try:
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if lo > hi:
                    raise ValueError
                out.update(range(lo, hi + 1))
            else:
                out.add(int(part))
        except ValueError:
            raise ValueError(f"bad --episodes item {part!r} "
                             "(expected e.g. '0-50,99')") from None
    return out


def select_episodes(a: DatasetView, b: DatasetView, episodes: set[int] | None = None,
                    tasks: list[str] | None = None) -> tuple[DatasetView, DatasetView]:
    """Keep only matching episode indices, on both sides.

    tasks: case-insensitive substrings; an index is kept when its episode in
    A *or* B has a matching task, so an episode whose task was renamed away
    from the filter still shows up as changed rather than removed.
    """
    keep = set(a.episodes) | set(b.episodes)
    if episodes is not None:
        keep &= episodes
    if tasks:
        needles = [t.lower() for t in tasks]

        def matches(ep) -> bool:
            return any(n in t.lower() for n in needles for t in ep.tasks)

        keep &= {i for ds in (a, b) for i, ep in ds.episodes.items() if matches(ep)}

    def restrict(ds: DatasetView) -> DatasetView:
        return replace(
            ds,
            episodes={i: ep for i, ep in ds.episodes.items() if i in keep},
            stats_mismatches=[m for m in ds.stats_mismatches if m["episode"] in keep],
        )

    return restrict(a), restrict(b)
