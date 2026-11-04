"""--verify-stats: per-episode metadata stats vs a recomputation from data/."""
import shutil

from robot_data_diff.differ import diff_datasets
from robot_data_diff.loader import load_dataset
from robot_data_diff.report import render_text

from lerobot_fixtures import clone


def _stale(write, episodes):
    """Metadata from `episodes`, but data/ where episode 1's action was doubled."""
    edited = clone(episodes)
    edited[1]["action"] = edited[1]["action"] * 2
    stale, fresh = write("stale", episodes), write("fresh", edited)
    shutil.rmtree(stale / "data")
    shutil.copytree(fresh / "data", stale / "data")
    return stale


def test_stale_metadata_is_flagged(write, episodes):
    ds = load_dataset(str(_stale(write, episodes)), verify_stats=True)
    assert ds.stats_verified
    bad = {(m["episode"], m["feature"]) for m in ds.stats_mismatches}
    assert bad == {(1, "action")}
    stats = {m["stat"] for m in ds.stats_mismatches}
    assert stats <= {"min", "max", "mean", "std"} and "max" in stats


def test_consistent_metadata_passes(write, episodes):
    ds = load_dataset(str(write("ok", episodes)), verify_stats=True)
    assert ds.stats_verified and ds.stats_mismatches == []


def test_verify_is_off_by_default(write, episodes):
    ds = load_dataset(str(_stale(write, episodes)))
    assert not ds.stats_verified and ds.stats_mismatches == []


def test_verify_works_without_quality(write, episodes):
    ds = load_dataset(str(_stale(write, episodes)), with_quality=False,
                      verify_stats=True)
    assert ds.stats_mismatches and ds.episodes[0].idle_ratio is None


def test_report_shows_stale_metadata(write, episodes):
    a = load_dataset(str(write("a", episodes)), verify_stats=True)
    b = load_dataset(str(_stale(write, episodes)), verify_stats=True)
    out = render_text(diff_datasets(a, b))
    assert "STALE METADATA" in out
    assert "B: ep 1 action" in out


def test_stale_std_alone_is_flagged(tmp_path, episodes):
    """Scaling data and regenerating only min/max/mean leaves std stale."""
    import numpy as np

    from lerobot_fixtures import write_v30

    edited = clone(episodes)
    edited[0]["action"] = edited[0]["action"] * 1.5
    act = edited[0]["action"].astype(np.float64)
    stale_std = {0: {"stats/action/std": episodes[0]["action"].astype(np.float64).std(0)}}
    root = write_v30(tmp_path / "ds", edited, stats_override=stale_std)
    ds = load_dataset(str(root), verify_stats=True)
    assert [(m["episode"], m["feature"], m["stat"]) for m in ds.stats_mismatches] == [
        (0, "action", "std")]
    np.testing.assert_allclose(ds.stats_mismatches[0]["data"], act.std(0), rtol=1e-5)
