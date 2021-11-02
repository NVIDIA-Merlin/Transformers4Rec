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

import importlib

import pytest

from merlin_standard_lib import Schema
from transformers4rec.data import tabular_sequence_testing_data, tabular_testing_data


@pytest.fixture
def yoochoose_path_file() -> str:
    return tabular_sequence_testing_data.path


@pytest.fixture
def yoochoose_schema_file() -> str:
    return tabular_sequence_testing_data.schema_path


@pytest.fixture
def yoochoose_schema() -> Schema:
    return tabular_sequence_testing_data.schema


@pytest.fixture
def tabular_data_file() -> str:
    return tabular_testing_data.path


@pytest.fixture
def tabular_schema_file() -> str:
    return tabular_testing_data.schema_path


@pytest.fixture
def tabular_schema() -> Schema:
    return tabular_testing_data.schema.remove_by_name(["session_id", "session_start", "day_idx"])


tf = importlib.util.find_spec("tensorflow")
if tf is not None:
    from tests.tf.conftest import *  # noqa

torch = importlib.util.find_spec("torch")
if torch is not None:
    from tests.torch.conftest import *  # noqa
