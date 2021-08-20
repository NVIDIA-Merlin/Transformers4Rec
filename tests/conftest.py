import pathlib

import pytest

from transformers4rec.utils.schema import DatasetSchema

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


@pytest.fixture
def assets():
    return ASSETS_DIR


@pytest.fixture
def schema_file():
    return ASSETS_DIR / "schema.pbtxt"


YOOCHOOSE_SCHEMA = ASSETS_DIR / "yoochoose" / "schema.pbtxt"


@pytest.fixture
def yoochoose_schema_file():
    return YOOCHOOSE_SCHEMA


@pytest.fixture
def yoochoose_data_file():
    return ASSETS_DIR / "yoochoose" / "data.parquet"


@pytest.fixture
def yoochoose_schema():
    return DatasetSchema.from_schema(str(YOOCHOOSE_SCHEMA))


from tests.tf.conftest import *  # noqa
from tests.torch.conftest import *  # noqa
