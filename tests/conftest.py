import pytest

from passos_wearable.config import load_config
from passos_wearable.data.synthetic import make_synthetic_recording


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def rec(cfg):
    """SYNTHETIC recording - unit tests only."""
    return make_synthetic_recording(300, seed=1, config=cfg)
