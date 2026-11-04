import numpy as np
import pytest

from robot_data_diff.differ import diff_datasets
from robot_data_diff.loader import load_dataset

from lerobot_fixtures import clone, make_episodes, write_v21


def _diff(path_a, path_b, **kw):
    return diff_datasets(load_dataset(str(path_a)), load_dataset(str(path_b)), **kw)


def test_identical_datasets_have_no_changes(write, episodes):
    r = _diff(write("a", episodes), write("b", clone(episodes)))
    assert (r.added, r.removed, r.changed, r.unchanged) == ([], [], [], 4)
    assert r.a_version == r.b_version == write.version


def test_added_removed_changed_unchanged_counts(write):
    a = make_episodes(n=6)
    b = clone(a)
    b = [ep for ep in b if ep["index"] not in (4, 5)]          # removed 4, 5
    b[0]["action"] = b[0]["action"] * 1.5                       # changed 0
    extra = clone([a[1]])[0]
    extra["index"] = 9                                          # added 9
    b.append(extra)
    r = _diff(write("a", a), write("b", b))
    assert r.added == [9]
    assert r.removed == [4, 5]
    assert [d.index for d in r.changed] == [0]
    assert r.unchanged == 3
    assert (r.a_episodes, r.b_episodes) == (6, 5)


def test_action_scale_shows_in_action_deltas(write, episodes):
    b = clone(episodes)
    b[2]["action"] = b[2]["action"] * 1.5
    r = _diff(write("a", episodes), write("b", b))
    (d,) = r.changed
    act = episodes[2]["action"].astype(np.float64)
    np.testing.assert_allclose(d.d_action_min, act.min(0) * 0.5, atol=1e-4)
    np.testing.assert_allclose(d.d_action_max, act.max(0) * 0.5, atol=1e-4)
    np.testing.assert_allclose(d.d_action_mean, act.mean(0) * 0.5, atol=1e-4)
    assert d.d_length == 0 and not d.tasks_changed


def test_task_change_is_flagged(write, episodes):
    b = clone(episodes)
    b[1]["tasks"] = ["place cube"]
    r = _diff(write("a", episodes), write("b", b))
    (d,) = r.changed
    assert d.index == 1 and d.tasks_changed
    assert (d.old_tasks, d.new_tasks) == (["pick cube"], ["place cube"])
    assert r.task_counts == {"pick cube": (4, 3), "place cube": (0, 1)}


def test_length_change_is_flagged(write, episodes):
    b = clone(episodes)
    b[3]["action"], b[3]["state"] = b[3]["action"][:-5], b[3]["state"][:-5]
    r = _diff(write("a", episodes), write("b", b))
    (d,) = r.changed
    assert (d.index, d.old_length, d.new_length, d.d_length) == (3, 30, 25, -5)


def test_schema_drift(write, episodes):
    a = write("a", episodes)
    b = write("b", clone(episodes),
              extra_features={"observation.images.top": {"dtype": "video",
                                                         "shape": [96, 96, 3]},
                              "observation.state": {"dtype": "float32",
                                                    "shape": [7]}},
              drop_features=("task_index",))
    r = _diff(a, b)
    assert r.features_added == ["observation.images.top"]
    assert r.features_removed == ["task_index"]
    assert len(r.features_changed) == 1
    assert r.features_changed[0].startswith("observation.state:")


@pytest.mark.parametrize("shape", ["nested", "toplevel"])
def test_v21_episode_stats_shapes(tmp_path, episodes, shape):
    """Real v2.1 nests stats under "stats"; older files put them top-level."""
    a = write_v21(tmp_path / "a", episodes, stats_shape=shape)
    ds = load_dataset(str(a))
    ep = ds.episodes[0]
    act = episodes[0]["action"].astype(np.float64)
    np.testing.assert_allclose(ep.action_min, act.min(0), atol=1e-5)
    np.testing.assert_allclose(ep.action_mean, act.mean(0), atol=1e-5)

    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 1.5
    r = _diff(a, write_v21(tmp_path / "b", b, stats_shape=shape))
    assert [d.index for d in r.changed] == [0]
    assert r.changed[0].d_action_max is not None


def test_quality_summary_present(write, episodes):
    r = _diff(write("a", episodes), write("b", clone(episodes)))
    assert r.mean_idle_a is not None and r.spikes_a is not None
    assert r.mean_idle_a == r.mean_idle_b


def test_no_quality_drops_quality(write, episodes):
    a = load_dataset(str(write("a", episodes)), with_quality=False)
    assert all(e.idle_ratio is None for e in a.episodes.values())
