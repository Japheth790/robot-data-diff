"""Human-readable text and HTML reports for a DiffResult."""
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


def render_html(r: DiffResult, max_show: int | None = None) -> str:
    """Self-contained HTML report. max_show truncates long episode lists."""
    e = html.escape
    shown_changed, more_changed = _head(r.changed, max_show)

    def num(v, fmt):
        return "n/a" if v is None else format(v, fmt)

    def changed_rows():
        rows = []
        for d in shown_changed:
            was = (f" (was {d.old_index})"
                   if d.old_index is not None and d.old_index != d.index else "")
            changes = [f"tasks: {e(str(d.old_tasks))} &rarr; {e(str(d.new_tasks))}"
                       for _ in [0] if d.tasks_changed]
            changes += [f"<code>&Delta;{e(label[2:-1])}</code> {e(val)}"
                        for label, val in _delta_lines(d)]
            rows.append(
                f"<tr><td>{d.index}{was}</td><td>{d.old_length} &rarr; {d.new_length} "
                f"({d.d_length:+d})</td>"
                f"<td>{'<br>'.join(changes) or '&mdash;'}</td>"
                f"<td>{num(d.d_idle_ratio, '+.4f')}</td>"
                f"<td>{num(d.d_spike_count, '+d')}</td></tr>"
            )
        if more_changed:
            rows.append(f'<tr><td colspan="5">&hellip; and {more_changed} more '
                        'changed episodes</td></tr>')
        return "\n".join(rows) or '<tr><td colspan="5">none</td></tr>'

    def task_rows():
        rows = []
        for t, (ca, cb) in r.task_counts.items():
            cls = ' class="hl"' if ca != cb else ""
            rows.append(f'<tr{cls}><td>{e(t[:80])}</td><td>{ca}</td><td>{cb}</td></tr>')
        return "\n".join(rows)

    def side(name, version, n, fps, robot):
        extra = "".join(f", {x}" for x in (f"{fps} fps" if fps else None,
                                           f"robot {e(str(robot))}" if robot else None) if x)
        return f"{e(name)} <code>{e(version)}</code> ({n} episodes{extra})"

    ckpt = f"<p><b>Policy checkpoint:</b> {e(r.checkpoint)}</p>" if r.checkpoint else ""
    moved = f", &gt;{len(r.moved)} moved" if r.match_by == "content" else ""
    moved_list = ("<br>Moved: <code>" + e(", ".join(f"{a} -> {b}" for a, b in r.moved))
                  + "</code>") if r.moved else ""
    dataset = ("<ul>" + "".join(f"<li><b>{e(n)}</b>: {e(str(va))} &rarr; {e(str(vb))}</li>"
                                for n, va, vb in r.dataset_changes) + "</ul>"
               if r.dataset_changes else "<p>Same fps, robot_type and codebase_version.</p>")
    stale = ""
    if r.a_stats_mismatches or r.b_stats_mismatches:
        stale = ("<h2>Stale metadata</h2><p>Per-episode stats disagree with data/:</p><ul>"
                 + "".join(f"<li>{e(line)}</li>" for line in mismatch_lines(r)) + "</ul>")
    elif r.a_stats_verified or r.b_stats_verified:
        stale = "<h2>Stale metadata</h2><p>Stats verified: metadata matches data/.</p>"
    qual = "<p>Not computed (--no-quality).</p>"
    if r.mean_idle_a is not None:
        qual = (f"<p>Mean idle-frame ratio: {r.mean_idle_a:.4f} &rarr; "
                f"{r.mean_idle_b:.4f} ({r.mean_idle_b - r.mean_idle_a:+.4f})<br>"
                f"Total action spikes: {r.spikes_a} &rarr; {r.spikes_b} "
                f"({r.spikes_b - r.spikes_a:+d})</p>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>robot-data-diff report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 14px; }}
th {{ background: #f4f4f4; }} .hl {{ background: #fff8e1; }}
code {{ background: #f4f4f4; padding: 1px 5px; border-radius: 3px; }}
h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 1.6em; }}
</style></head><body>
<h1>robot-data-diff report</h1>
<p><b>A:</b> {side(r.a_name, r.a_version, r.a_episodes, r.a_fps, r.a_robot_type)}<br>
<b>B:</b> {side(r.b_name, r.b_version, r.b_episodes, r.b_fps, r.b_robot_type)}</p>
{ckpt}
<h2>Episode inventory</h2>
<p>{r.a_episodes} &rarr; {r.b_episodes} episodes &mdash;
+{len(r.added)} added, &minus;{len(r.removed)} removed,
~{len(r.changed)} changed{moved}, ={r.unchanged} unchanged</p>
<p>Added: <code>{e(_index_list(r.added, max_show))}</code><br>
Removed: <code>{e(_index_list(r.removed, max_show))}</code>{moved_list}</p>
<h2>Changed episodes</h2>
<table><tr><th>ep</th><th>length</th><th>changes (B &minus; A)</th>
<th>&Delta;idle</th><th>&Delta;spikes</th></tr>
{changed_rows()}</table>
<h2>Task inventory</h2>
<table><tr><th>task</th><th>A</th><th>B</th></tr>{task_rows()}</table>
<h2>Dataset-level changes</h2>
{dataset}
<h2>Schema drift</h2>
<p>Added: <code>{e(str(r.features_added))}</code><br>
Removed: <code>{e(str(r.features_removed))}</code><br>
Changed: <code>{e(str(r.features_changed))}</code></p>
{stale}
<h2>Quality summary</h2>
{qual}
</body></html>
"""
