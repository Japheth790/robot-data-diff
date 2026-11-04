from robot_data_diff.differ import diff_datasets
from robot_data_diff.loader import load_dataset
from robot_data_diff.report import render_html, render_text

from lerobot_fixtures import clone, write_v30


def _result(tmp_path, episodes, **kw):
    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 1.5
    b[1]["tasks"] = ["<script>alert(1)</script>"]
    return diff_datasets(load_dataset(str(write_v30(tmp_path / "a", episodes))),
                         load_dataset(str(write_v30(tmp_path / "b", b))), **kw)


def test_render_text(tmp_path, episodes):
    out = render_text(_result(tmp_path, episodes, checkpoint="ckpt-42"))
    assert "~2 changed" in out
    assert "[ep 0]" in out and "d(action_max)" in out
    assert "policy checkpoint: ckpt-42" in out


def test_render_html_escapes_task_names(tmp_path, episodes):
    out = render_html(_result(tmp_path, episodes, checkpoint="<b>x</b>"))
    assert out.startswith("<!DOCTYPE html>")
    assert "<script>alert" not in out and out.count("<script>") == 1  # ours
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "&lt;b&gt;x&lt;/b&gt;" in out


def _rich_result(tmp_path):
    """A diff exercising every report section."""
    import shutil

    from lerobot_fixtures import make_episodes

    a = make_episodes(n=5)
    b = clone(a)[1:]                       # drop episode 0, reindex -> moved
    for i, e in enumerate(b):
        e["index"] = i
    b[0]["state"] = b[0]["state"] * 2      # state change (A's ep 1)
    b[1]["action"] = b[1]["action"] * 1.5  # action change
    b[2]["tasks"] = ["place cube"]
    pa_, pb = write_v30(tmp_path / "a", a), write_v30(tmp_path / "b", b, fps=30)
    stale = clone(b)
    stale[3]["action"] = stale[3]["action"] + 1
    shutil.rmtree(pb / "data")
    shutil.copytree(write_v30(tmp_path / "s", stale) / "data", pb / "data")
    kw = dict(content_hash=True, verify_stats=True)
    return diff_datasets(load_dataset(str(pa_), **kw), load_dataset(str(pb), **kw),
                         match_by="content")


def test_html_shows_what_text_shows(tmp_path):
    r = _rich_result(tmp_path)
    text, page = render_text(r), render_html(r)
    assert "STALE METADATA" in text and "DATASET-LEVEL CHANGES" in text
    for needle in ("30 fps", "<b>fps</b>: 10 &rarr; 30", "moved", "action_mean",
                   "observation.state_max", "Stale metadata", "place cube",
                   "Dataset-level changes", "idle"):
        assert needle in page, needle


def test_render_markdown(tmp_path):
    from robot_data_diff.report import render_markdown

    r = _rich_result(tmp_path)
    md = render_markdown(r)
    assert md.startswith("### robot-data-diff:")
    assert "| fps | 10 | 30 ⚠️ |" in md
    assert "<details><summary>Changed episodes" in md
    assert "`observation.state_max`" in md
    assert "**Stale metadata**" in md
    assert "| place cube | 0 | 1 |" in md
    # every table row has the same number of cells as its header
    for block in md.split("\n\n"):
        rows = [ln for ln in block.splitlines() if ln.startswith("|")]
        if rows:
            assert len({ln.count("|") for ln in rows}) == 1, block


def test_markdown_escapes_pipes_and_html(tmp_path, episodes):
    from robot_data_diff.report import render_markdown

    md = render_markdown(_result(tmp_path, episodes))
    assert "<script>" not in md and "&lt;script&gt;" in md


def test_cli_markdown(tmp_path, episodes, capsys):
    from robot_data_diff.cli import main

    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 2
    pa, pb = str(write_v30(tmp_path / "a", episodes)), str(write_v30(tmp_path / "b", b))
    assert main(["diff", pa, pb, "--format", "markdown"]) == 0
    assert "~1 changed" in capsys.readouterr().out


def test_html_is_self_contained_with_histogram_and_sorting(tmp_path):
    page = render_html(_rich_result(tmp_path))
    assert "<svg" in page and 'class="bar-a"' in page and 'class="bar-b"' in page
    assert 'class="sortable"' in page and "<script>" in page
    assert "prefers-color-scheme: dark" in page
    # no external resources
    for needle in ("http://", "https://", "<link", "src="):
        assert needle not in page, needle


def test_html_colors_increases_and_decreases(tmp_path, episodes):
    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 1.5
    b[1]["action"], b[1]["state"] = b[1]["action"][:-5], b[1]["state"][:-5]
    r = diff_datasets(load_dataset(str(write_v30(tmp_path / "a", episodes))),
                      load_dataset(str(write_v30(tmp_path / "b", b))))
    page = render_html(r)
    assert 'class="num neg" data-v="-5"' in page        # shorter episode
    assert 'class="pos"' in page and 'class="neg"' in page
