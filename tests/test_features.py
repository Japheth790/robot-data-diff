"""Per-feature stats and bookkeeping columns."""
import numpy as np

from robot_data_diff.differ import diff_datasets
from robot_data_diff.loader import load_dataset
from robot_data_diff.report import render_text

from lerobot_fixtures import clone, make_episodes


def _diff(path_a, path_b, **kw):
    return diff_datasets(load_dataset(str(path_a)), load_dataset(str(path_b)), **kw)


def test_state_only_change_is_detected(write, episodes):
    b = clone(episodes)
    b[1]["state"] = b[1]["state"] + 0.25
    r = _diff(write("a", episodes), write("b", b))
    (d,) = r.changed
    assert d.index == 1
    assert set(d.feature_deltas) == {"observation.state"}
    np.testing.assert_allclose(d.feature_deltas["observation.state"]["mean"],
                               [0.25, 0.25], atol=1e-5)
    # std and spread are unaffected by a constant offset
    assert "std" not in d.feature_deltas["observation.state"]


def test_other_numeric_features_are_tracked(write):
    a = make_episodes()
    for ep in a:
        ep["extra"] = {"observation.velocity": np.ones((30, 3))}
    b = clone(a)
    b[2]["extra"]["observation.velocity"][:, 0] = 2.0
    r = _diff(write("a", a), write("b", b))
    assert [d.index for d in r.changed] == [2]
    assert set(r.changed[0].feature_deltas) == {"observation.velocity"}


def test_feature_only_in_one_side_does_not_mark_every_episode(write, episodes):
    b = clone(episodes)
    for ep in b:
        ep["extra"] = {"observation.velocity": np.ones((30, 3))}
    r = _diff(write("a", episodes), write("b", b))
    assert r.changed == [] and r.unchanged == 4


def test_bookkeeping_columns_are_ignored(write, episodes):
    """Removing episode 0 shifts the global frame `index` of every later
    episode; that alone must not mark them changed."""
    r = _diff(write("a", episodes), write("b", clone(episodes)[1:]))
    assert r.removed == [0] and r.changed == [] and r.unchanged == 3


def test_text_shows_only_changed_features(write, episodes):
    b = clone(episodes)
    b[0]["tasks"] = ["place cube"]
    b[1]["state"] = b[1]["state"] * 2
    out = render_text(_diff(write("a", episodes), write("b", b)))
    ep0 = out.split("[ep 0]")[1].split("[ep 1]")[0]
    assert "tasks:" in ep0 and "d(action" not in ep0
    ep1 = out.split("[ep 1]")[1].split("TASK INVENTORY")[0]
    assert "d(observation.state_max)" in ep1 and "d(action" not in ep1
