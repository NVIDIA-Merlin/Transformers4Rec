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

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_sequential_block_yoochoose(yoochoose_schema, tf_yoochoose_like):
    inputs = tf4rec.TabularFeatures.from_schema(yoochoose_schema, aggregation="concat")

    body = tf4rec.SequentialBlock([inputs, tf4rec.MLPBlock([64])])

    outputs = body(tf_yoochoose_like)

    assert list(outputs.shape) == [100, 64]


def test_sequential_block_yoochoose_without_aggregation(yoochoose_schema, tf_yoochoose_like):
    inputs = tf4rec.TabularFeatures.from_schema(yoochoose_schema)

    with pytest.raises(TypeError) as excinfo:
        body = tf4rec.SequentialBlock([inputs, tf4rec.MLPBlock([64])])

        body(tf_yoochoose_like)

        assert "did you forget to add aggregation to TabularFeatures" in str(excinfo.value)
