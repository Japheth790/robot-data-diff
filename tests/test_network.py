"""Tests that hit the Hugging Face Hub. Skipped by default; run with
`pytest -m network`."""
import pytest

from robot_data_diff.differ import diff_datasets
from robot_data_diff.loader import load_dataset


@pytest.mark.network
def test_pusht_against_itself(tmp_path):
    a = load_dataset("lerobot/pusht", cache_dir=tmp_path, verify_stats=True)
    assert a.version.startswith("v3")
    assert a.stats_verified and a.stats_mismatches == [], a.stats_mismatches[:3]
    assert "action" in a.episodes[0].stats
    assert len(a.episodes) > 100
    r = diff_datasets(a, a)
    assert r.changed == [] and r.unchanged == len(a.episodes)


@pytest.mark.network
def test_pusht_v21_to_v30_conversion_changes_no_episodes(tmp_path):
    """The real v2.1 and v3.0 releases of pusht hold the same data."""
    kw = dict(cache_dir=tmp_path, verify_stats=True, content_hash=True)
    a = load_dataset("lerobot/pusht@v2.1", **kw)
    b = load_dataset("lerobot/pusht@v3.0", **kw)
    assert a.version == "v2.1" and b.version == "v3.0"
    assert a.stats_mismatches == [] and b.stats_mismatches == []
    r = diff_datasets(a, b, match_by="content")
    assert r.dataset_changes == [("codebase_version", "v2.1", "v3.0")]
    assert (r.added, r.removed, r.moved, r.changed) == ([], [], [], [])
    assert r.unchanged == 206
