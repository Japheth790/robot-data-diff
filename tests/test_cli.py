"""CLI behaviour: output, exit codes and error messages."""
import pytest

from robot_data_diff.cli import main

from lerobot_fixtures import clone, write_v30


@pytest.fixture
def pair(tmp_path, episodes):
    b = clone(episodes)
    b[0]["action"] = b[0]["action"] * 1.5
    return str(write_v30(tmp_path / "a", episodes)), str(write_v30(tmp_path / "b", b))


def test_diff_ok(pair, capsys):
    assert main(["diff", *pair]) == 0
    out = capsys.readouterr().out
    assert "~1 changed" in out


def test_html_output(pair, tmp_path, capsys):
    out = tmp_path / "sub" / "r.html"
    assert main(["diff", *pair, "--html", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_missing_local_path_is_one_line_error(pair, tmp_path, capsys):
    missing = str(tmp_path / "nope")
    assert main(["diff", pair[0], missing]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    last = err.strip().splitlines()[-1]
    assert last.startswith("error:") and "nope" in last


def test_relative_missing_path_is_not_sent_to_the_hub(capsys, monkeypatch):
    import huggingface_hub

    def no_network(*a, **k):
        raise AssertionError("tried to download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    assert main(["diff", "./does-not-exist", "./also/missing"]) == 2
    assert "does-not-exist" in capsys.readouterr().err


def test_dir_without_info_json(tmp_path, pair, capsys):
    (tmp_path / "empty").mkdir()
    assert main(["diff", pair[0], str(tmp_path / "empty")]) == 2
    err = capsys.readouterr().err
    assert "meta/info.json" in err and "Traceback" not in err


def test_verbose_shows_traceback(tmp_path, pair, capsys):
    (tmp_path / "empty").mkdir()
    assert main(["diff", pair[0], str(tmp_path / "empty"), "--verbose"]) == 2
    assert "Traceback" in capsys.readouterr().err


def test_hub_failure_is_one_line_error(pair, capsys, monkeypatch):
    import huggingface_hub

    def fail(*a, **k):
        raise OSError("404 Client Error.\n\nRepository Not Found for url: ...")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    assert main(["diff", pair[0], "someorg/nonexistent"]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert err[-1].startswith("error:") and "someorg/nonexistent" in err[-1]
    assert "Traceback" not in "\n".join(err)


def test_format_json_on_stdout(pair, capsys):
    import json

    assert main(["diff", *pair, "--format", "json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["schema_version"] == 1 and d["summary"]["changed"] == 1


def test_json_file_next_to_text(pair, tmp_path, capsys):
    import json

    out = tmp_path / "r.json"
    assert main(["diff", *pair, "--json", str(out)]) == 0
    assert "~1 changed" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["changed"] == 1
