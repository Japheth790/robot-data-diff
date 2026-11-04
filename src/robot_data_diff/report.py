"""Human-readable text, Markdown and HTML reports for a DiffResult."""
from __future__ import annotations

import html

from .differ import DiffResult, EpisodeDelta

MAX_DIMS = 12  # longer vectors are truncated in the reports


def _fmt_vec(v, nd=4):
    if v is None:
        return "n/a"
    head = ", ".join(f"{x:+.{nd}f}" for x in v[:MAX_DIMS])
    more = f", ... +{len(v) - MAX_DIMS} dims" if len(v) > MAX_DIMS else ""
    return f"[{head}{more}]"


def _delta_lines(d: EpisodeDelta) -> list[tuple[str, str]]:
    """(label, formatted delta) for every per-feature stat that changed."""
    out = []
    for feat, fd in d.feature_deltas.items():
        for stat, v in fd.items():
            out.append((f"d({feat}_{stat})",
                        _fmt_vec(v) if v is not None else "n/a (shape changed or missing)"))
    return out


def _head(items: list, max_show: int | None) -> tuple[list, int]:
    """First max_show items and how many were left out (None = show all)."""
    if max_show is None or len(items) <= max_show:
        return items, 0
    return items[:max_show], len(items) - max_show


def _index_list(items: list, max_show: int | None) -> str:
    shown, more = _head(items, max_show)
    return str(shown) + (f" ... and {more} more" if more else "")


def mismatch_lines(r: DiffResult, limit: int = 20) -> list[str]:
    """One line per stale-metadata finding, e.g. 'B: ep 1 action max: ...'."""
    out = []
    for side, ms in (("A", r.a_stats_mismatches), ("B", r.b_stats_mismatches)):
        for m in ms[:limit]:
            what = m["stat"] if m["feature"] is None else f"{m['feature']} {m['stat']}"
            meta, data = m["metadata"], m["data"]
            if isinstance(meta, list):
                meta, data = _fmt_vec(meta), _fmt_vec(data)
            out.append(f"{side}: ep {m['episode']} {what}: metadata {meta} vs data {data}")
        if len(ms) > limit:
            out.append(f"{side}: ... and {len(ms) - limit} more")
    return out


def render_text(r: DiffResult, max_show: int | None = None) -> str:
    """Plain-text report. max_show truncates long episode lists."""
    L: list[str] = []
    L.append("=" * 72)
    L.append("robot-data-diff  —  episode-aware dataset diff")
    L.append("=" * 72)
    L.append(f"A: {r.a_name}   ({r.a_version}, {r.a_episodes} episodes"
             + (f", {r.a_fps} fps" if r.a_fps else "") + ")")
    L.append(f"B: {r.b_name}   ({r.b_version}, {r.b_episodes} episodes"
             + (f", {r.b_fps} fps" if r.b_fps else "") + ")")
    if r.checkpoint:
        L.append(f"policy checkpoint: {r.checkpoint}")
    L.append("")

    moved = f", >{len(r.moved)} moved" if r.match_by == "content" else ""
    L.append(f"Episodes: {r.a_episodes} -> {r.b_episodes} "
             f"(+{len(r.added)} added, -{len(r.removed)} removed, "
             f"~{len(r.changed)} changed{moved}, ={r.unchanged} unchanged)")
    if r.added:
        L.append(f"  added:   {_index_list(r.added, max_show)}")
    if r.removed:
        L.append(f"  removed: {_index_list(r.removed, max_show)}")
    if r.moved:
        shown, more = _head(r.moved, max_show)
        L.append("  moved:   " + ", ".join(f"{ia} -> {ib}" for ia, ib in shown)
                 + (f" ... and {more} more" if more else ""))
    L.append("")

    if r.changed:
        L.append(f"CHANGED EPISODES ({len(r.changed)}):")
        shown, more = _head(r.changed, max_show)
        for d in shown:
            was = (f" (was {d.old_index})"
                   if d.old_index is not None and d.old_index != d.index else "")
            L.append(f"  [ep {d.index}{was}] length {d.old_length} -> {d.new_length} "
                     f"({d.d_length:+d} frames)")
            if d.tasks_changed:
                L.append(f"           tasks: {d.old_tasks} -> {d.new_tasks}")
            for label, val in _delta_lines(d):
                L.append(f"           {label:<14} {val}")
            if d.d_idle_ratio is not None:
                L.append(f"           d(idle_ratio)  {d.d_idle_ratio:+.4f}")
            if d.d_spike_count is not None:
                L.append(f"           d(spikes)      {d.d_spike_count:+d}")
        if more:
            L.append(f"  ... and {more} more changed episodes (raise --max-show to see them)")
        L.append("")
    else:
        L.append("No changed episodes (same index + identical stats).")
        L.append("")

    if r.task_counts:
        L.append("TASK INVENTORY (episodes per task: A -> B):")
        for t, (ca, cb) in r.task_counts.items():
            mark = "  *" if ca != cb else "   "
            L.append(f"{mark} {t[:60]:62s} {ca:5d} -> {cb:5d}")
        L.append("")

    if r.dataset_changes:
        L.append("DATASET-LEVEL CHANGES:")
        for name, va, vb in r.dataset_changes:
            L.append(f"  ! {name}: {va} -> {vb}")
        L.append("")
    else:
        L.append("Dataset-level: same fps, robot_type and codebase_version.")
        L.append("")

    if r.features_added or r.features_removed or r.features_changed:
        L.append("SCHEMA DRIFT:")
        for f in r.features_added:
            L.append(f"  + {f}")
        for f in r.features_removed:
            L.append(f"  - {f}")
        for f in r.features_changed:
            L.append(f"  ~ {f}")
        L.append("")
    else:
        L.append("Schema: identical feature sets.")
        L.append("")

    if r.a_stats_mismatches or r.b_stats_mismatches:
        L.append("STALE METADATA (per-episode stats disagree with data/):")
        L.extend(f"  ! {line}" for line in mismatch_lines(r))
        L.append("")
    elif r.a_stats_verified or r.b_stats_verified:
        sides = ", ".join(s for s, v in (("A", r.a_stats_verified),
                                         ("B", r.b_stats_verified)) if v)
        L.append(f"Stats verified: metadata matches data/ ({sides}).")
        L.append("")

    if r.mean_idle_a is not None:
        L.append("QUALITY SUMMARY (heuristic, CPU-only):")
        L.append(f"  mean idle-frame ratio: {r.mean_idle_a:.4f} -> {r.mean_idle_b:.4f} "
                 f"({r.mean_idle_b - r.mean_idle_a:+.4f})")
        L.append(f"  total action spikes:   {r.spikes_a} -> {r.spikes_b} "
                 f"({r.spikes_b - r.spikes_a:+d})")
        L.append("")

    L.append("=" * 72)
    return "\n".join(L)


def _md(s) -> str:
    """Safe inside a GitHub-flavoured Markdown table cell."""
    return html.escape(" ".join(str(s).split())).replace("|", "&#124;")


def render_markdown(r: DiffResult, max_show: int | None = None) -> str:
    """GitHub-flavoured Markdown, for PR comments and Hub discussions."""
    L: list[str] = []
    L.append(f"### robot-data-diff: `{_md(r.a_name)}` → `{_md(r.b_name)}`")
    L.append("")
    if r.checkpoint:
        L.append(f"Policy checkpoint: `{_md(r.checkpoint)}`")
        L.append("")

    changed_fields = {n for n, _, _ in r.dataset_changes}
    L.append("| | A | B |")
    L.append("|---|---|---|")
    for label, field, va, vb in (
            ("codebase_version", "codebase_version", r.a_version, r.b_version),
            ("fps", "fps", r.a_fps, r.b_fps),
            ("robot_type", "robot_type", r.a_robot_type, r.b_robot_type),
            ("episodes", None, r.a_episodes, r.b_episodes),
            ("frames", None, sum(r.a_lengths), sum(r.b_lengths))):
        flag = " ⚠️" if field in changed_fields else ""
        L.append(f"| {label} | {_md(va)} | {_md(vb)}{flag} |")
    L.append("")

    moved = f" · >{len(r.moved)} moved" if r.match_by == "content" else ""
    L.append(f"**Episodes:** +{len(r.added)} added · −{len(r.removed)} removed · "
             f"~{len(r.changed)} changed{moved} · ={r.unchanged} unchanged")
    L.append("")
    if r.added:
        L.append(f"- Added: `{_md(_index_list(r.added, max_show))}`")
    if r.removed:
        L.append(f"- Removed: `{_md(_index_list(r.removed, max_show))}`")
    if r.moved:
        shown, more = _head(r.moved, max_show)
        L.append("- Moved: " + ", ".join(f"{a} → {b}" for a, b in shown)
                 + (f" … and {more} more" if more else ""))
    if r.added or r.removed or r.moved:
        L.append("")

    if r.dataset_changes or r.features_added or r.features_removed or r.features_changed:
        L.append("**Dataset-level changes / schema drift**")
        L.append("")
        L.extend(f"- ⚠️ {_md(n)}: `{_md(va)}` → `{_md(vb)}`" for n, va, vb in r.dataset_changes)
        L.extend(f"- ➕ `{_md(f)}`" for f in r.features_added)
        L.extend(f"- ➖ `{_md(f)}`" for f in r.features_removed)
        L.extend(f"- ✏️ `{_md(f)}`: `{_md(sa)}` → `{_md(sb)}`"
                 for f, sa, sb in r.features_changed_specs)
        L.append("")

    if r.changed:
        shown, more = _head(r.changed, max_show)
        L.append(f"<details><summary>Changed episodes ({len(r.changed)})</summary>")
        L.append("")
        L.append("| ep | length | changes (B − A) | Δidle | Δspikes |")
        L.append("|---|---|---|---|---|")
        for d in shown:
            was = f" (was {d.old_index})" if d.old_index not in (None, d.index) else ""
            parts = ([f"tasks: {_md(d.old_tasks)} → {_md(d.new_tasks)}"]
                     if d.tasks_changed else [])
            parts += [f"`{_md(label[2:-1])}` {_md(val)}" for label, val in _delta_lines(d)]
            idle = "n/a" if d.d_idle_ratio is None else f"{d.d_idle_ratio:+.4f}"
            spikes = "n/a" if d.d_spike_count is None else f"{d.d_spike_count:+d}"
            L.append(f"| {d.index}{was} | {d.old_length} → {d.new_length} "
                     f"({d.d_length:+d}) | {'<br>'.join(parts) or '—'} | {idle} | {spikes} |")
        if more:
            L.append(f"| … | | {more} more changed episodes | | |")
        L.append("")
        L.append("</details>")
        L.append("")

    diff_tasks = [(t, ca, cb) for t, (ca, cb) in r.task_counts.items() if ca != cb]
    if diff_tasks:
        L.append("**Task inventory changes** (episodes per task)")
        L.append("")
        L.append("| task | A | B |")
        L.append("|---|---|---|")
        L.extend(f"| {_md(t)} | {ca} | {cb} |" for t, ca, cb in diff_tasks)
        L.append("")

    if r.a_stats_mismatches or r.b_stats_mismatches:
        L.append("**Stale metadata** (per-episode stats disagree with `data/`)")
        L.append("")
        L.extend(f"- {_md(line)}" for line in mismatch_lines(r))
        L.append("")

    if r.mean_idle_a is not None:
        L.append(f"**Quality (heuristic):** mean idle ratio {r.mean_idle_a:.4f} → "
                 f"{r.mean_idle_b:.4f} ({r.mean_idle_b - r.mean_idle_a:+.4f}), "
                 f"action spikes {r.spikes_a} → {r.spikes_b} "
                 f"({r.spikes_b - r.spikes_a:+d})")
        L.append("")
    return "\n".join(L)


def render_html(r: DiffResult, max_show: int | None = None) -> str:
    """Self-contained HTML report (inline CSS/JS/SVG, dark mode, sortable
    changed-episodes table). max_show truncates long episode lists."""
    from .html_report import render

    return render(r, max_show)
