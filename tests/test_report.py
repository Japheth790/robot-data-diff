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
