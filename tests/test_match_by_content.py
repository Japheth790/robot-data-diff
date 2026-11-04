"""--match-by content: pair episodes across revisions by a hash of their frames."""
import pytest

from robot_data_diff.differ import diff_datasets
from robot_data_diff.loader import load_dataset
from robot_data_diff.report import render_text

from lerobot_fixtures import clone, make_episodes


def _load(path, **kw):
    return load_dataset(str(path), content_hash=True, **kw)


def _drop_and_reindex(eps, drop):
    out = [e for e in clone(eps) if e["index"] != drop]
    for i, e in enumerate(out):
        e["index"] = i
    return out


def test_reindexed_episodes_show_as_moved(write):
    a = make_episodes(n=6)
    b = _drop_and_reindex(a, drop=2)          # B[i] = A[i + 1] for i >= 2
    pa, pb = write("a", a), write("b", b)

    by_index = diff_datasets(_load(pa), _load(pb))
    assert by_index.removed == [5] and len(by_index.changed) == 3

    r = diff_datasets(_load(pa), _load(pb), match_by="content")
    assert r.removed == [2] and r.added == []
    assert r.moved == [(3, 2), (4, 3), (5, 4)]
    assert r.changed == [] and r.unchanged == 2
    out = render_text(r)
    assert ">3 moved" in out and "3 -> 2" in out


def test_edited_in_place_is_still_changed(write, episodes):
    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 1.5
    r = diff_datasets(_load(write("a", episodes)), _load(write("b", b)),
                      match_by="content")
    assert [d.index for d in r.changed] == [0]
    assert r.moved == [] and r.unchanged == 3


def test_moved_episode_with_new_task_is_changed(write):
    a = make_episodes(n=3)
    b = _drop_and_reindex(a, drop=0)
    b[0]["tasks"] = ["place cube"]             # was A's episode 1
    r = diff_datasets(_load(write("a", a)), _load(write("b", b)), match_by="content")
    (d,) = r.changed
    assert (d.old_index, d.index, d.tasks_changed) == (1, 0, True)
    assert r.moved == [(1, 0), (2, 1)]


def test_content_mode_requires_hashes(write, episodes):
    a = load_dataset(str(write("a", episodes)))
    with pytest.raises(ValueError, match="content_hash"):
        diff_datasets(a, a, match_by="content")


def test_hash_ignores_metadata_only_changes(write, episodes):
    b = clone(episodes)
    b[0]["tasks"] = ["place cube"]
    ha = _load(write("a", episodes)).episodes[0].content_hash
    hb = _load(write("b", b)).episodes[0].content_hash
    assert ha is not None and ha == hb
