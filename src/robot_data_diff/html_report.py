"""Self-contained HTML report: one file, inline CSS/JS/SVG, no network."""
from __future__ import annotations

import html

import numpy as np

from .differ import DiffResult, EpisodeDelta
from .report import MAX_DIMS, _head, _index_list, mismatch_lines

e = html.escape

CSS = """
:root {
  --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --border: #d0d7de;
  --card: #f6f8fa; --hl: #fff8c5; --warn-bg: #fff1e5; --warn: #bc4c00;
  --pos: #0969da; --neg: #8250df; --bad: #cf222e; --good: #1a7f37;
  --a: #8c959f; --b: #0969da; --code: #eff1f3;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8d96a0; --border: #30363d;
    --card: #161b22; --hl: #3b2e00; --warn-bg: #3d1d00; --warn: #ffa657;
    --pos: #58a6ff; --neg: #bc8cff; --bad: #ff7b72; --good: #3fb950;
    --a: #6e7681; --b: #58a6ff; --code: #1f242c;
  }
}
* { box-sizing: border-box; }
body { font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
       background: var(--bg); color: var(--fg); max-width: 1180px;
       margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 8px; }
.meta { color: var(--muted); margin: 0; }
.meta b { color: var(--fg); }
code { background: var(--code); padding: 1px 5px; border-radius: 4px;
       font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.cards { display: grid; gap: 10px; margin: 20px 0;
         grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 8px; padding: 10px 12px; }
.card .n { font-size: 22px; font-weight: 600; }
.card .l { color: var(--muted); font-size: 12px; }
.card.warn { border-color: var(--warn); background: var(--warn-bg); }
.alert { background: var(--warn-bg); border: 1px solid var(--warn);
         border-radius: 8px; padding: 8px 14px; margin: 12px 0; }
.alert ul { margin: 4px 0; padding-left: 20px; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid var(--border); padding: 6px 8px;
         text-align: left; vertical-align: top; }
th { position: sticky; top: 0; background: var(--card); font-weight: 600;
     white-space: nowrap; }
th.sort { cursor: pointer; user-select: none; }
th.sort::after { content: " \\2195"; color: var(--muted); }
th.asc::after { content: " \\2191"; } th.desc::after { content: " \\2193"; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.hl td { background: var(--hl); }
.scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
.chg { font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.pos { color: var(--pos); } .neg { color: var(--neg); }
.bad { color: var(--bad); font-weight: 600; } .good { color: var(--good); }
.muted { color: var(--muted); }
svg text { fill: var(--muted); font-size: 11px; }
svg .bar-a { fill: var(--a); } svg .bar-b { fill: var(--b); }
svg .axis { stroke: var(--border); }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
"""

# Click a header to sort; numeric cells carry data-v.
JS = """
document.querySelectorAll("table.sortable").forEach(function (table) {
  table.querySelectorAll("th.sort").forEach(function (th, col) {
    th.addEventListener("click", function () {
      var body = table.tBodies[0], asc = !th.classList.contains("asc");
      table.querySelectorAll("th").forEach(function (h) {
        h.classList.remove("asc", "desc"); });
      th.classList.add(asc ? "asc" : "desc");
      var rows = Array.prototype.slice.call(body.querySelectorAll("tr[data-ep]"));
      rows.sort(function (r1, r2) {
        var a = r1.children[col], b = r2.children[col];
        var x = a.dataset.v !== undefined ? parseFloat(a.dataset.v) : a.textContent;
        var y = b.dataset.v !== undefined ? parseFloat(b.dataset.v) : b.textContent;
        if (x < y) return asc ? -1 : 1;
        if (x > y) return asc ? 1 : -1;
        return 0;
      });
      rows.forEach(function (r) { body.insertBefore(r, body.querySelector("tr.more")); });
    });
  });
});
"""


def _sign_class(v, higher_is_bad=False) -> str:
    if v is None or v == 0:
        return "muted"
    if higher_is_bad:
        return "bad" if v > 0 else "good"
    return "pos" if v > 0 else "neg"


def _num_cell(v, fmt, higher_is_bad=False) -> str:
    if v is None:
        return '<td class="num muted" data-v="0">n/a</td>'
    return (f'<td class="num {_sign_class(v, higher_is_bad)}" data-v="{v}">'
            f"{format(v, fmt)}</td>")


def _vec_html(v) -> str:
    if v is None:
        return '<span class="muted">n/a (shape changed or missing)</span>'
    parts = [f'<span class="{_sign_class(x)}">{x:+.4f}</span>' for x in v[:MAX_DIMS]]
    more = f", &hellip; +{len(v) - MAX_DIMS} dims" if len(v) > MAX_DIMS else ""
    return "[" + ", ".join(parts) + more + "]"


def _changes_cell(d: EpisodeDelta) -> str:
    lines = []
    if d.tasks_changed:
        lines.append(f"tasks: {e(str(d.old_tasks))} &rarr; {e(str(d.new_tasks))}")
    for feat, fd in d.feature_deltas.items():
        for stat, v in fd.items():
            lines.append(f"<b>&Delta;{e(feat)}_{e(stat)}</b> {_vec_html(v)}")
    return "<br>".join(lines) or '<span class="muted">&mdash;</span>'


def _histogram_svg(a: list[int], b: list[int]) -> str:
    """Side-by-side bar histogram of episode lengths, A vs B."""
    if not a and not b:
        return '<p class="muted">No episodes.</p>'
    both = np.asarray(a + b, dtype=float)
    lo, hi = float(both.min()), float(both.max())
    n_bins = int(min(24, max(1, len(np.unique(both)))))
    edges = np.linspace(lo, hi if hi > lo else lo + 1, n_bins + 1)
    ca, _ = np.histogram(a, bins=edges)
    cb, _ = np.histogram(b, bins=edges)
    top = max(int(ca.max(initial=0)), int(cb.max(initial=0)), 1)

    W, H, L, R, T, B = 720, 200, 44, 12, 24, 30
    pw, ph = W - L - R, H - T - B
    slot = pw / n_bins
    bw = max(slot * 0.4, 1.0)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
           'aria-label="Histogram of episode lengths in A and B">']
    out.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    out.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}"/>')
    out.append(f'<text x="{L - 6}" y="{T + 4}" text-anchor="end">{top}</text>')
    out.append(f'<text x="{L - 6}" y="{T + ph}" text-anchor="end">0</text>')
    for i in range(n_bins):
        x0 = L + i * slot + (slot - 2 * bw) / 2
        rng = f"{edges[i]:.0f}&ndash;{edges[i + 1]:.0f} frames"
        for j, (cls, counts, label) in enumerate((("bar-a", ca, "A"), ("bar-b", cb, "B"))):
            h = ph * counts[i] / top
            out.append(f'<rect class="{cls}" x="{x0 + j * bw:.1f}" y="{T + ph - h:.1f}" '
                       f'width="{bw:.1f}" height="{h:.1f}"><title>{label}: {counts[i]} '
                       f"episodes, {rng}</title></rect>")
    for frac in (0.0, 0.5, 1.0):
        x = L + frac * pw
        val = edges[0] + frac * (edges[-1] - edges[0])
        anchor = "start" if frac == 0 else "end" if frac == 1 else "middle"
        out.append(f'<text x="{x:.1f}" y="{H - 10}" text-anchor="{anchor}">{val:.0f}</text>')
    out.append(f'<text x="{L + pw / 2:.1f}" y="{H}" text-anchor="middle">'
               "episode length (frames)</text>")
    lx = L + pw - 120
    out.append(f'<rect class="bar-a" x="{lx}" y="6" width="10" height="10"/>'
               f'<text x="{lx + 14}" y="15">A ({len(a)})</text>'
               f'<rect class="bar-b" x="{lx + 62}" y="6" width="10" height="10"/>'
               f'<text x="{lx + 76}" y="15">B ({len(b)})</text>')
    out.append("</svg>")
    return "".join(out)


def _card(n, label, warn=False) -> str:
    cls = "card warn" if warn else "card"
    return f'<div class="{cls}"><div class="n">{n}</div><div class="l">{label}</div></div>'


def render(r: DiffResult, max_show: int | None = None) -> str:
    from . import __version__

    def side(name, version, n, fps, robot):
        bits = [f"<code>{e(str(version))}</code>", f"{n} episodes"]
        if fps:
            bits.append(f"{e(str(fps))} fps")
        if robot:
            bits.append(f"robot {e(str(robot))}")
        return f"{e(name)} &middot; " + " &middot; ".join(bits)

    n_schema = len(r.features_added) + len(r.features_removed) + len(r.features_changed)
    n_stale = len(r.a_stats_mismatches) + len(r.b_stats_mismatches)
    cards = [
        _card(f"{r.a_episodes} &rarr; {r.b_episodes}", "episodes"),
        _card(f"+{len(r.added)}", "added"),
        _card(f"&minus;{len(r.removed)}", "removed"),
        _card(f"~{len(r.changed)}", "changed", warn=bool(r.changed)),
    ]
    if r.match_by == "content":
        cards.append(_card(f"&gt;{len(r.moved)}", "moved"))
    cards += [
        _card(f"={r.unchanged}", "unchanged"),
        _card(len(r.dataset_changes), "dataset-level changes", warn=bool(r.dataset_changes)),
        _card(n_schema, "schema changes", warn=bool(n_schema)),
    ]
    if r.mean_idle_a is not None:
        d_idle = r.mean_idle_b - r.mean_idle_a
        d_sp = r.spikes_b - r.spikes_a
        cards.append(_card(f'<span class="{_sign_class(d_idle, True)}">{d_idle:+.4f}</span>',
                           f"mean idle ratio ({r.mean_idle_a:.4f} &rarr; {r.mean_idle_b:.4f})"))
        cards.append(_card(f'<span class="{_sign_class(d_sp, True)}">{d_sp:+d}</span>',
                           f"action spikes ({r.spikes_a} &rarr; {r.spikes_b})"))
    if r.a_stats_verified or r.b_stats_verified:
        cards.append(_card(n_stale, "stale metadata stats", warn=bool(n_stale)))

    alerts = []
    if r.dataset_changes:
        alerts.append("<b>Dataset-level changes</b><ul>" + "".join(
            f"<li><b>{e(n)}</b>: {e(str(va))} &rarr; {e(str(vb))}</li>"
            for n, va, vb in r.dataset_changes) + "</ul>")
    if n_schema:
        items = ([f"<li>added <code>{e(f)}</code></li>" for f in r.features_added]
                 + [f"<li>removed <code>{e(f)}</code></li>" for f in r.features_removed]
                 + [f"<li>changed <code>{e(f)}</code>: {e(str(sa))} &rarr; {e(str(sb))}</li>"
                    for f, sa, sb in r.features_changed_specs])
        alerts.append("<b>Schema drift</b><ul>" + "".join(items) + "</ul>")
    if n_stale:
        alerts.append("<b>Stale metadata</b> &mdash; per-episode stats disagree with "
                      "<code>data/</code><ul>"
                      + "".join(f"<li>{e(line)}</li>" for line in mismatch_lines(r))
                      + "</ul>")
    alerts_html = "".join(f'<div class="alert">{a}</div>' for a in alerts)

    lists = [f"Added: <code>{e(_index_list(r.added, max_show))}</code>" if r.added else "",
             f"Removed: <code>{e(_index_list(r.removed, max_show))}</code>"
             if r.removed else ""]
    if r.moved:
        shown, more = _head(r.moved, max_show)
        lists.append("Moved: <code>" + e(", ".join(f"{a} -> {b}" for a, b in shown))
                     + (f" &hellip; and {more} more" if more else "") + "</code>")
    lists_html = "<br>".join(x for x in lists if x)

    shown, more = _head(r.changed, max_show)
    rows = []
    for d in shown:
        old = d.old_index if d.old_index is not None else d.index
        rows.append(
            f'<tr data-ep="{d.index}"><td class="num" data-v="{d.index}">{d.index}</td>'
            f'<td class="num" data-v="{old}">{old}</td>'
            f'<td class="num" data-v="{d.old_length}">{d.old_length}</td>'
            f'<td class="num" data-v="{d.new_length}">{d.new_length}</td>'
            + _num_cell(d.d_length, "+d")
            + f'<td class="chg">{_changes_cell(d)}</td>'
            + _num_cell(d.d_idle_ratio, "+.4f", higher_is_bad=True)
            + _num_cell(d.d_spike_count, "+d", higher_is_bad=True) + "</tr>")
    if more:
        rows.append(f'<tr class="more"><td colspan="8" class="muted">&hellip; and {more} '
                    "more changed episodes (raise --max-show to see them)</td></tr>")
    if r.changed:
        changed_html = (
            '<div class="scroll"><table class="sortable"><thead><tr>'
            '<th class="sort num">ep</th><th class="sort num">ep in A</th>'
            '<th class="sort num">len A</th><th class="sort num">len B</th>'
            '<th class="sort num">&Delta;len</th><th>changes (B &minus; A)</th>'
            '<th class="sort num">&Delta;idle</th><th class="sort num">&Delta;spikes</th>'
            "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div>")
    else:
        changed_html = '<p class="muted">No changed episodes.</p>'

    task_rows = []
    for t, (ca, cb) in r.task_counts.items():
        cls = ' class="hl"' if ca != cb else ""
        task_rows.append(f"<tr{cls}><td>{e(t)}</td><td class=\"num\">{ca}</td>"
                         f'<td class="num">{cb}</td>' + _num_cell(cb - ca, "+d") + "</tr>")
    tasks_html = ('<div class="scroll"><table><thead><tr><th>task</th><th class="num">A</th>'
                  '<th class="num">B</th><th class="num">&Delta;</th></tr></thead><tbody>'
                  + "".join(task_rows) + "</tbody></table></div>")

    if r.mean_idle_a is None:
        quality = '<p class="muted">Not computed (--no-quality).</p>'
    else:
        quality = (f"<p>Mean idle-frame ratio {r.mean_idle_a:.4f} &rarr; {r.mean_idle_b:.4f}; "
                   f"total action spikes {r.spikes_a} &rarr; {r.spikes_b}. "
                   '<span class="muted">Lightweight CPU heuristics, not a validator.</span></p>')
    verified = ""
    if (r.a_stats_verified or r.b_stats_verified) and not n_stale:
        verified = '<p class="good">Stats verified: metadata matches data/.</p>'
    ckpt = (f'<p class="meta"><b>Policy checkpoint:</b> {e(r.checkpoint)}</p>'
            if r.checkpoint else "")
    # pre-3.12 f-strings can't nest quotes, so assemble pieces first
    cards_html = "".join(cards)
    lists_block = f"<p>{lists_html}</p>" if lists_html else ""
    side_a = side(r.a_name, r.a_version, r.a_episodes, r.a_fps, r.a_robot_type)
    side_b = side(r.b_name, r.b_version, r.b_episodes, r.b_fps, r.b_robot_type)
    histogram = _histogram_svg(r.a_lengths, r.b_lengths)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>robot-data-diff report</title>
<style>{CSS}</style></head><body>
<h1>robot-data-diff report</h1>
<p class="meta"><b>A:</b> {side_a}</p>
<p class="meta"><b>B:</b> {side_b}</p>
{ckpt}
<div class="cards">{cards_html}</div>
{alerts_html}
{lists_block}
<h2>Changed episodes</h2>
{changed_html}
<h2>Episode lengths</h2>
{histogram}
<h2>Task inventory</h2>
{tasks_html}
<h2>Quality summary</h2>
{quality}
{verified}
<footer>Generated by robot-data-diff {e(__version__)} &middot; match by {e(r.match_by)}</footer>
<script>{JS}</script>
</body></html>
"""
