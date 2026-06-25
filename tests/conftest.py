import pytest

from himadri.config import Config
from himadri.pipeline import build_features, preprocess
from himadri.synth.generate import generate_scene


@pytest.fixture(scope="session")
def cfg():
    c = Config()
    c.grid.height = 192
    c.grid.width = 192
    c.volume.n_mc = 400  # keep tests fast
    return c


@pytest.fixture(scope="session")
def scene(cfg):
    return preprocess(generate_scene(cfg), cfg)


@pytest.fixture(scope="session")
def feats(scene, cfg):
    return build_features(scene, cfg)
