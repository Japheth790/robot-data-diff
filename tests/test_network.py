"""Tests that hit the Hugging Face Hub. Skipped by default; run with
`pytest -m network`."""
import pytest

from robot_data_diff.differ import diff_datasets
from robot_data_diff.loader import load_dataset


@pytest.mark.network
def test_pusht_against_itself(tmp_path):
    a = load_dataset("lerobot/pusht", cache_dir=tmp_path)
    assert a.version.startswith("v3")
    assert len(a.episodes) > 100
    r = diff_datasets(a, a)
    assert r.changed == [] and r.unchanged == len(a.episodes)
