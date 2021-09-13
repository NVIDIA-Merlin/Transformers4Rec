#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pathlib

import pytest

from merlin_standard_lib import Schema

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


@pytest.fixture
def assets():
    return ASSETS_DIR


@pytest.fixture
def schema_file():
    return ASSETS_DIR / "schema.pbtxt"


YOOCHOOSE_SCHEMA = ASSETS_DIR / "yoochoose" / "schema.pbtxt"
YOOCHOOSE_PATH = ASSETS_DIR / "yoochoose" / "data.parquet"


@pytest.fixture
def yoochoose_path_file():
    return YOOCHOOSE_PATH


@pytest.fixture
def yoochoose_schema_file():
    return YOOCHOOSE_SCHEMA


@pytest.fixture
def yoochoose_data_file():
    return ASSETS_DIR / "yoochoose" / "data.parquet"


@pytest.fixture
def yoochoose_schema():
    schema = Schema().from_proto_text(str(YOOCHOOSE_SCHEMA))

    return schema.remove_by_name(["session_id", "session_start", "day_idx"])


from tests.tf.conftest import *  # noqa
from tests.torch.conftest import *  # noqa
