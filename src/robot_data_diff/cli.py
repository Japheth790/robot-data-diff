"""CLI entry point: robot-data-diff diff <A> <B> [--html OUT] [--checkpoint LBL]."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from . import __version__
from .differ import diff_datasets
from .loader import load_dataset
from .report import render_html, render_text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robot-data-diff",
        description="Episode-aware diff for LeRobot robot-training datasets. "
                    "Answers: what changed in the data since the last good run?",
    )
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff", help="diff two dataset revisions")
    d.add_argument("dataset_a", help="local path or HF id (org/repo[@rev])")
    d.add_argument("dataset_b", help="local path or HF id (org/repo[@rev])")
    d.add_argument("--html", metavar="OUT",
                   help="also write an HTML report to OUT")
    d.add_argument("--checkpoint", metavar="LABEL",
                   help="policy checkpoint name to link this diff to")
    d.add_argument("--no-quality", action="store_true",
                   help="skip per-episode quality stats (metadata-only, faster)")
    d.add_argument("--verify-stats", action="store_true",
                   help="recompute per-episode min/max/mean from data/ and warn "
                        "where the metadata stats disagree (stale metadata)")
    d.add_argument("--match-by", choices=["index", "content"], default="index",
                   help="pair episodes by episode_index (default) or by a hash "
                        "of their action/state frames, which reports reordered "
                        "episodes as moved instead of added + removed")
    d.add_argument("--verbose", action="store_true",
                   help="show full tracebacks on errors")
    d.add_argument("--cache-dir", metavar="DIR", default=None,
                   help="where to cache Hugging Face downloads")
    return p


def cmd_diff(args) -> int:
    print("loading A ...", file=sys.stderr)
    a = load_dataset(args.dataset_a, cache_dir=args.cache_dir,
                     with_quality=not args.no_quality,
                     verify_stats=args.verify_stats,
                     content_hash=args.match_by == "content")
    print(f"  {a.version}, {len(a.episodes)} episodes", file=sys.stderr)
    print("loading B ...", file=sys.stderr)
    b = load_dataset(args.dataset_b, cache_dir=args.cache_dir,
                     with_quality=not args.no_quality,
                     verify_stats=args.verify_stats,
                     content_hash=args.match_by == "content")
    print(f"  {b.version}, {len(b.episodes)} episodes", file=sys.stderr)

    for label, ds in (("A", a), ("B", b)):
        if ds.stats_mismatches:
            n = len({m["episode"] for m in ds.stats_mismatches})
            print(f"warning: {label}: metadata stats disagree with data/ for {n} "
                  "episode(s); metadata may be stale", file=sys.stderr)

    result = diff_datasets(a, b, checkpoint=args.checkpoint, match_by=args.match_by)
    print(render_text(result))
    if args.html:
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(result))
        print(f"\nHTML report written to {args.html}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    """Exit codes: 0 OK, 2 error (bad input, unreadable dataset, ...)."""
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "diff":
            return cmd_diff(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        if getattr(args, "verbose", False):
            traceback.print_exc()
        else:
            msg = (str(e).strip().splitlines() or [type(e).__name__])[0]
            print(f"error: {msg}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
