"""fps / robot_type / codebase_version changes from meta/info.json."""
from robot_data_diff.differ import diff_datasets
from robot_data_diff.loader import load_dataset
from robot_data_diff.report import render_text

from lerobot_fixtures import clone, write_v21, write_v30


def _diff(path_a, path_b, **kw):
    return diff_datasets(load_dataset(str(path_a)), load_dataset(str(path_b)), **kw)


def test_fps_change_is_reported_at_dataset_level(write, episodes):
    r = _diff(write("a", episodes), write("b", clone(episodes), fps=30))
    assert ("fps", 10, 30) in r.dataset_changes
    out = render_text(r)
    assert "DATASET-LEVEL CHANGES" in out
    assert "fps: 10 -> 30" in out


def test_robot_type_and_version_changes(tmp_path, episodes):
    a = write_v21(tmp_path / "a", episodes, robot_type="so100")
    b = write_v30(tmp_path / "b", clone(episodes), robot_type="so101")
    r = _diff(a, b)
    assert ("robot_type", "so100", "so101") in r.dataset_changes
    assert ("codebase_version", "v2.1", "v3.0") in r.dataset_changes
    # converting formats alone must not mark episodes changed
    assert r.changed == []


def test_no_dataset_level_changes(write, episodes):
    r = _diff(write("a", episodes), write("b", clone(episodes)))
    assert r.dataset_changes == []
