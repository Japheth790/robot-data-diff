"""Time load_dataset() on a synthetic 5,000-episode dataset.

    python benchmarks/bench_load.py [--episodes 5000] [--frames 100] [--format v3.0]

The dataset is generated once into a temp dir (outside the repo) and
reused by later runs with the same parameters.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from robot_data_diff.loader import load_dataset  # noqa: E402

from lerobot_fixtures import make_episodes, write_v21, write_v30  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--dims", type=int, default=6)
    p.add_argument("--format", choices=["v2.1", "v3.0"], default="v3.0")
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()

    root = (Path(tempfile.gettempdir()) / "rdd-bench"
            / f"{a.format}-{a.episodes}x{a.frames}x{a.dims}")
    if not (root / "meta" / "info.json").exists():
        t0 = time.perf_counter()
        eps = make_episodes(n=a.episodes, T=a.frames, D=a.dims)
        (write_v21 if a.format == "v2.1" else write_v30)(root, eps)
        print(f"built {root} in {time.perf_counter() - t0:.1f}s")

    for with_quality in (True, False):
        best = float("inf")
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            ds = load_dataset(str(root), with_quality=with_quality)
            best = min(best, time.perf_counter() - t0)
        print(f"{a.format} {len(ds.episodes)} episodes x {a.frames} frames, "
              f"with_quality={with_quality}: best of {a.repeat} = {best:.3f}s")


if __name__ == "__main__":
    main()
