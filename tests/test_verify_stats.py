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
    assert stats <= {"min", "max", "mean"} and "max" in stats


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
