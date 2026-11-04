"""Column-wise frame reading (data/**/*.parquet)."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robot_data_diff.loader import _scan_frames, _vector_column, load_dataset
from robot_data_diff.stats import action_quality

from lerobot_fixtures import write_v30


def test_vector_column_list_and_fixed_size_list():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    flat = pa.array(x.ravel())
    fixed = pa.chunked_array([pa.FixedSizeListArray.from_arrays(flat, 3)])
    var = pa.chunked_array([pa.array(x.tolist(), type=pa.list_(pa.float32()))])
    np.testing.assert_array_equal(_vector_column(fixed), x)
    np.testing.assert_array_equal(_vector_column(var), x)
    # sliced arrays (non-zero offset) must still line up
    np.testing.assert_array_equal(_vector_column(fixed.slice(1, 2)), x[1:3])


def test_vector_column_scalar():
    np.testing.assert_array_equal(
        _vector_column(pa.chunked_array([pa.array([1.0, 2.0])])), [[1.0], [2.0]])


def test_shuffled_rows_and_episode_split_across_files(tmp_path, episodes):
    root = write_v30(tmp_path / "ds", episodes)
    expected = {i: action_quality(e["action"]) for i, e in enumerate(episodes)}

    data = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data)
    # interleave episodes 2 and 3, and split episode 1 across two files
    ep = table.column("episode_index").to_numpy()
    idx = np.arange(len(ep))
    first = np.r_[idx[ep == 0], idx[ep == 1][:10]]
    rest = idx[~np.isin(idx, first)]
    e2, e3 = rest[ep[rest] == 2], rest[ep[rest] == 3]
    inter = np.empty(len(e2) + len(e3), dtype=int)
    inter[0::2], inter[1::2] = e2, e3
    rest = np.r_[rest[ep[rest] == 1], inter]
    pq.write_table(table.take(first), data)
    pq.write_table(table.take(rest), root / "data" / "chunk-000" / "file-001.parquet")

    frames = _scan_frames(sorted((root / "data").rglob("*.parquet")), ["action"])
    for i, e in enumerate(episodes):
        np.testing.assert_allclose(frames[i]["action"], e["action"])
    ds = load_dataset(str(root))
    for i, (idle, spikes) in expected.items():
        assert ds.episodes[i].idle_ratio == idle
        assert ds.episodes[i].spike_count == spikes


def test_no_quality_reads_no_data_files(write, episodes, monkeypatch):
    import robot_data_diff.loader as loader

    def boom(*a, **k):
        raise AssertionError("data files were read")

    root = write("a", episodes)
    monkeypatch.setattr(loader, "_scan_frames", boom)
    ds = load_dataset(str(root), with_quality=False)
    assert len(ds.episodes) == 4
    assert all(e.idle_ratio is None for e in ds.episodes.values())


def test_no_quality_skips_hf_data_download(tmp_path, episodes, monkeypatch):
    import huggingface_hub

    root = write_v30(tmp_path / "ds", episodes)
    calls = []

    def fake_snapshot_download(repo_id, **kw):
        calls.append(kw["allow_patterns"])
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    load_dataset("org/repo", with_quality=False)
    load_dataset("org/repo@v1")
    assert calls == [["meta/**"], ["meta/**", "data/**"]]
