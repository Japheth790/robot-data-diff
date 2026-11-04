"""--episodes, --task and --max-show."""
import pytest

from robot_data_diff.cli import main
from robot_data_diff.differ import diff_datasets
from robot_data_diff.filters import parse_episodes, select_episodes
from robot_data_diff.loader import load_dataset
from robot_data_diff.report import render_html, render_text

from lerobot_fixtures import clone, make_episodes, write_v30


def test_parse_episodes():
    assert parse_episodes("0-3,7, 9") == {0, 1, 2, 3, 7, 9}
    for bad in ("3-1", "a", "1-", ""):
        with pytest.raises(ValueError):
            parse_episodes(bad)


@pytest.fixture
def views(tmp_path):
    a = make_episodes(n=10)
    for e in a[5:]:
        e["tasks"] = ["Place the Cube"]
    b = clone(a)
    for e in b:
        e["action"] = e["action"] * 2          # every episode changed
    b[2]["tasks"] = ["place the cube"]         # renamed into the filter
    return (load_dataset(str(write_v30(tmp_path / "a", a))),
            load_dataset(str(write_v30(tmp_path / "b", b))))


def test_episode_filter(views):
    a, b = select_episodes(*views, episodes=parse_episodes("0-2,8,99"))
    r = diff_datasets(a, b)
    assert [d.index for d in r.changed] == [0, 1, 2, 8]
    assert r.a_episodes == r.b_episodes == 4


def test_task_filter_matches_either_side(views):
    a, b = select_episodes(*views, tasks=["place"])
    r = diff_datasets(a, b)
    assert [d.index for d in r.changed] == [2, 5, 6, 7, 8, 9]
    assert r.removed == [] and r.added == []


def test_filters_combine(views):
    a, b = select_episodes(*views, episodes={0, 1, 2, 3, 9}, tasks=["PLACE"])
    assert [d.index for d in diff_datasets(a, b).changed] == [2, 9]


def test_max_show_truncates(views):
    r = diff_datasets(*views)
    text = render_text(r, max_show=3)
    assert text.count("[ep ") == 3
    assert "... and 7 more changed episodes" in text
    assert render_text(r).count("[ep ") == 10
    page = render_html(r, max_show=3)
    assert "and 7 more" in page


def test_cli_filters(tmp_path, capsys):
    a = make_episodes(n=6)
    b = clone(a)
    for e in b:
        e["action"] = e["action"] * 2
    pa, pb = str(write_v30(tmp_path / "a", a)), str(write_v30(tmp_path / "b", b))
    assert main(["diff", pa, pb, "--episodes", "1-4", "--max-show", "2"]) == 0
    out = capsys.readouterr().out
    assert "~4 changed" in out and "... and 2 more changed episodes" in out
    assert main(["diff", pa, pb, "--episodes", "4-1"]) == 2
