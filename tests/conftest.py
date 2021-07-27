import pathlib

import pytest

from tests.tf.conftest import *  # noqa
from tests.torch.conftest import *  # noqa

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


@pytest.fixture
def assets():
    return ASSETS_DIR


@pytest.fixture
def schema_file():
    return ASSETS_DIR / "schema.pbtxt"
