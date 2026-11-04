"""CLI entry point: robot-data-diff diff <A> <B> [--format F] [--html OUT] [--json OUT]."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import __version__
from .differ import diff_datasets
from .filters import parse_episodes, select_episodes
from .gate import evaluate, parse_fail_on
from .loader import load_dataset
from .report import render_html, render_markdown, render_text


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
    d.add_argument("--format", choices=["text", "json", "markdown"], default="text",
                   help="what to print on stdout (default: text)")
    d.add_argument("--html", metavar="OUT",
                   help="also write an HTML report to OUT")
    d.add_argument("--json", metavar="OUT", dest="json_out",
                   help="also write the machine-readable JSON result to OUT")
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
    d.add_argument("--episodes", metavar="SPEC",
                   help="only diff these episode indices, e.g. '0-50,99'")
    d.add_argument("--task", metavar="TEXT", action="append",
                   help="only diff episodes whose task (in A or B) contains TEXT "
                        "(case-insensitive; repeatable)")
    d.add_argument("--max-show", metavar="N", type=int, default=None,
                   help="show at most N changed episodes (and N added/removed "
                        "indices) in the text/HTML reports")
    d.add_argument("--fail-on", metavar="RULES",
                   help="exit 1 if any rule hits, e.g. "
                        "'schema,fps,removed,changed>5%%,idle-increase>0.05' "
                        "(see README: Use in CI)")
    d.add_argument("--verbose", action="store_true",
                   help="show full tracebacks on errors")
    d.add_argument("--cache-dir", metavar="DIR", default=None,
                   help="where to cache Hugging Face downloads")
    return p


def cmd_diff(args) -> int:
    rules = parse_fail_on(args.fail_on) if args.fail_on else []
    episodes = parse_episodes(args.episodes) if args.episodes else None
    if args.max_show is not None and args.max_show < 0:
        raise ValueError("--max-show must be >= 0")
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

    if episodes is not None or args.task:
        a, b = select_episodes(a, b, episodes=episodes, tasks=args.task)
        print(f"filtered to {len(a.episodes)} (A) / {len(b.episodes)} (B) episodes",
              file=sys.stderr)

    result = diff_datasets(a, b, checkpoint=args.checkpoint, match_by=args.match_by)
    hits = evaluate(rules, result) if rules else []
    data = result.to_dict()
    if rules:
        data["fail_on"] = {"rules": [r.text for r in rules], "failed": hits}
    as_json = json.dumps(data, indent=2, allow_nan=False)
    if args.format == "json":
        print(as_json)
    elif args.format == "markdown":
        print(render_markdown(result, args.max_show))
    else:
        print(render_text(result, args.max_show))
    for path, render, what in ((args.html, lambda r: render_html(r, args.max_show),
                                "HTML report"),
                               (args.json_out, lambda _: as_json, "JSON result")):
        if path:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render(result) + "\n", encoding="utf-8")
            print(f"{what} written to {path}", file=sys.stderr)
    for hit in hits:
        print(f"fail-on: {hit}", file=sys.stderr)
    return 1 if hits else 0


def main(argv=None) -> int:
    """Exit codes: 0 OK, 1 a --fail-on rule hit, 2 error (bad input,
    unreadable dataset, ...)."""
    args = build_parser().parse_args(argv)
    # Reports contain non-ASCII (arrows, task names); a redirected stdout on
    # Windows would otherwise use the ANSI code page and crash.
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
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
