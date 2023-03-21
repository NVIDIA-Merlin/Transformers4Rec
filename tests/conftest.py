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

from __future__ import absolute_import

from pathlib import Path

import numpy as np
import pytest
from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata

from merlin_standard_lib import Schema
from transformers4rec.data import tabular_sequence_testing_data, tabular_testing_data

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def ecommerce_data() -> Dataset:
    np.random.seed(0)
    return generate_data("e-commerce", num_rows=100)


@pytest.fixture
def testing_data() -> Dataset:
    data = generate_data("testing", num_rows=100)
    data.schema = data.schema.without(["session_id", "session_start", "day_idx"])

    return data


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


@pytest.fixture
def tabular_core_schema(tabular_schema):
    return TensorflowMetadata.from_json(tabular_schema.to_json()).to_merlin_schema()


def parametrize_schemas(name):
    if name == "tabular":
        schema = tabular_testing_data.schema.remove_by_name(
            ["session_id", "session_start", "day_idx"]
        )
    elif name == "yoochoose":
        schema = tabular_sequence_testing_data.schema

    return pytest.mark.parametrize(
        "schema",
        [
            pytest.param(schema, id="merlin-standard-lib"),
            pytest.param(
                TensorflowMetadata.from_json(schema.to_json()).to_merlin_schema(),
                id="merlin-core",
            ),
        ],
    )


try:
    import torchmetrics  # noqa

    from tests.unit.torch._conftest import *  # noqa
except ModuleNotFoundError:
    pass
