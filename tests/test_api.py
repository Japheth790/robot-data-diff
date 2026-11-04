"""Public Python API and DiffResult.to_dict()."""
import json

import robot_data_diff as rdd

from lerobot_fixtures import clone


def test_public_api(write, episodes):
    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 1.5
    r = rdd.diff_datasets(rdd.load_dataset(str(write("a", episodes))),
                          rdd.load_dataset(str(write("b", b))))
    assert isinstance(r, rdd.DiffResult)
    assert "~1 changed" in rdd.render_text(r)
    assert rdd.render_html(r).startswith("<!DOCTYPE html>")


def test_to_dict_is_json_with_stable_keys(write, episodes):
    b = clone(episodes)[:3]
    b[0]["action"] = b[0]["action"] * 1.5
    r = rdd.diff_datasets(rdd.load_dataset(str(write("a", episodes))),
                          rdd.load_dataset(str(write("b", b))), checkpoint="c1")
    d = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    assert d["schema_version"] == rdd.SCHEMA_VERSION == 1
    assert set(d) == {"schema_version", "tool", "checkpoint", "match_by", "a", "b",
                      "summary", "episodes", "tasks", "dataset_changes", "schema"}
    assert d["summary"] == {"added": 0, "removed": 1, "changed": 1, "moved": 0,
                            "unchanged": 2}
    assert d["episodes"]["removed"] == [3]
    (c,) = d["episodes"]["changed"]
    assert c["index"] == c["old_index"] == 0
    assert set(c["feature_deltas"]["action"]) >= {"min", "max", "mean"}
    assert d["a"]["episodes"] == 4 and d["a"]["frames"] == 120
    assert d["tasks"] == [{"task": "pick cube", "a": 4, "b": 3}]
    assert d["checkpoint"] == "c1" and d["tool"]["name"] == "robot-data-diff"


def test_to_dict_schema_changes_are_structured(write, episodes):
    r = rdd.diff_datasets(
        rdd.load_dataset(str(write("a", episodes))),
        rdd.load_dataset(str(write("b", clone(episodes), fps=30, extra_features={
            "observation.state": {"dtype": "float32", "shape": [7]}}))))
    d = r.to_dict()
    assert d["dataset_changes"] == [{"field": "fps", "a": 10, "b": 30}]
    assert d["schema"]["changed"] == [{
        "feature": "observation.state",
        "a": {"dtype": "float32", "shape": [2]},
        "b": {"dtype": "float32", "shape": [7]}}]
