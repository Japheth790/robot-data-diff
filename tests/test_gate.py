"""--fail-on rules (CI gate)."""
import json

import pytest

from robot_data_diff.cli import main
from robot_data_diff.differ import diff_datasets
from robot_data_diff.gate import evaluate, parse_fail_on
from robot_data_diff.loader import load_dataset

from lerobot_fixtures import clone, make_episodes, write_v30


@pytest.fixture
def result(tmp_path):
    """20 episodes in A; B removes 1, changes 2 (10%), adds 1, and runs at 30 fps."""
    a = make_episodes(n=20)
    b = [e for e in clone(a) if e["index"] != 19]
    b[0]["action"] = b[0]["action"] * 1.5
    b[1]["state"] = b[1]["state"] + 1
    extra = clone([a[5]])[0]
    extra["index"] = 30
    b.append(extra)
    pa, pb = write_v30(tmp_path / "a", a), write_v30(tmp_path / "b", b, fps=30)
    return diff_datasets(load_dataset(str(pa)), load_dataset(str(pb)))


@pytest.mark.parametrize("spec,hit", [
    ("schema", False),
    ("fps", True),
    ("dataset", True),
    ("robot_type", False),
    ("removed", True),
    ("removed>1", False),
    ("added>=1", True),
    ("changed>2", False),
    ("changed>1", True),
    ("changed>5%", True),
    ("changed>10%", False),
    ("changed>=10%", True),
    ("moved", False),
    ("tasks", False),
    ("idle-increase>0.5", False),
    ("spikes-increase>100", False),
])
def test_single_rules(result, spec, hit):
    assert bool(evaluate(parse_fail_on(spec), result)) is hit


def test_comma_list_reports_every_hit(result):
    hits = evaluate(parse_fail_on("schema, fps,changed>5%,removed>3"), result)
    assert len(hits) == 2
    assert hits[0].startswith("fps") and "10 -> 30" in hits[0]
    assert hits[1].startswith("changed>5%") and "10.0%" in hits[1]


@pytest.mark.parametrize("bad", ["", "bogus", "changed>", "changed<5", "fps>2",
                                 "changed>x%", "idle-increase>5%"])
def test_bad_specs_raise(bad):
    with pytest.raises(ValueError):
        parse_fail_on(bad)


def test_quality_rules_need_quality(tmp_path, episodes):
    p = str(write_v30(tmp_path / "a", episodes))
    r = diff_datasets(load_dataset(p, with_quality=False),
                      load_dataset(p, with_quality=False))
    with pytest.raises(ValueError, match="no-quality"):
        evaluate(parse_fail_on("idle-increase>0.05"), r)


def test_stale_rule_needs_verify(tmp_path, episodes):
    p = str(write_v30(tmp_path / "a", episodes))
    r = diff_datasets(load_dataset(p), load_dataset(p))
    with pytest.raises(ValueError, match="verify-stats"):
        evaluate(parse_fail_on("stale"), r)


def test_cli_exit_codes(tmp_path, episodes, capsys):
    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 2
    pa = str(write_v30(tmp_path / "a", episodes))
    pb = str(write_v30(tmp_path / "b", b))

    assert main(["diff", pa, pb, "--fail-on", "schema,removed"]) == 0
    assert main(["diff", pa, pb, "--fail-on", "schema,changed>10%"]) == 1
    err = capsys.readouterr().err
    assert "fail-on: changed>10%" in err

    out = tmp_path / "r.json"
    assert main(["diff", pa, pb, "--fail-on", "changed", "--json", str(out)]) == 1
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["fail_on"]["rules"] == ["changed"] and len(d["fail_on"]["failed"]) == 1

    # a bad rule is an error (2), reported before any dataset is loaded
    capsys.readouterr()
    assert main(["diff", pa, pb, "--fail-on", "chnaged"]) == 2
    assert "loading A" not in capsys.readouterr().err
