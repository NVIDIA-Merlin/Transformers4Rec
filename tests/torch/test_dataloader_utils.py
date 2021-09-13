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

import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers4rec.torch")


def test_pyarrow_load(yoochoose_schema, yoochoose_path_file):
    pytest.importorskip("pyarrow")
    max_sequence_length = 20
    batch_size = 16
    loader = tr.utils.data_utils.PyarrowDataLoader.from_schema(
        yoochoose_schema,
        yoochoose_path_file,
        batch_size,
        max_sequence_length,
        drop_last=True,
        shuffle=False,
        shuffle_buffer_size=0.1,
    )
    batch = next(iter(loader))
    assert all(feat.ndim == 2 for feat in batch.values())
    assert all(feat.size()[0] == batch_size for feat in batch.values())
    assert all(feat.size()[-1] == max_sequence_length for feat in batch.values())
    assert all(feat.device == torch.device("cpu") for feat in batch.values())


def test_features_from_schema(yoochoose_schema, yoochoose_path_file):
    pytest.importorskip("pyarrow")
    max_sequence_length = 20
    batch_size = 16
    loader = tr.utils.data_utils.PyarrowDataLoader.from_schema(
        yoochoose_schema,
        yoochoose_path_file,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        drop_last=True,
        shuffle=False,
        shuffle_buffer_size=0.1,
    )
    batch = next(iter(loader))
    features = yoochoose_schema.column_names

    assert set(batch.keys()).issubset(set(features))


def test_loader_from_registry(yoochoose_schema, yoochoose_path_file):
    pytest.importorskip("pyarrow")
    max_sequence_length = 70
    batch_size = 8
    loader = tr.utils.data_utils.T4RecDataLoader.parse("pyarrow").from_schema(
        yoochoose_schema,
        str(yoochoose_path_file),
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        drop_last=True,
        shuffle=False,
        shuffle_buffer_size=0,
    )
    batch = next(iter(loader))
    features = yoochoose_schema.column_names
    assert set(batch.keys()).issubset(set(features))
