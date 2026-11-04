import pytest

from lerobot_fixtures import make_episodes, write_v21, write_v30


@pytest.fixture
def episodes():
    return make_episodes()


@pytest.fixture(params=["v2.1", "v3.0"])
def write(request, tmp_path):
    """write(name, episodes, **kw) -> Path, in whichever format is parametrized."""
    writer = write_v21 if request.param == "v2.1" else write_v30

    def _write(name, eps, **kw):
        return writer(tmp_path / name, eps, **kw)

    _write.version = request.param
    return _write
