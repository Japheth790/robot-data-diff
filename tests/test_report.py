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
    assert "<script>" not in out
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
