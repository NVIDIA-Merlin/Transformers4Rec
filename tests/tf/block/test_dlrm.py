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

tr = pytest.importorskip("transformers4rec.tf")


def test_dlrm_block_yoochoose(tabular_schema, tf_tabular_data):
    all_features_schema = tabular_schema

    dlrm = tr.DLRMBlock.from_schema(all_features_schema, bottom_mlp=tr.MLPBlock([64]))

    body = tr.SequentialBlock([dlrm, tr.MLPBlock([64])])

    outputs = body(tf_tabular_data)

    assert list(outputs.shape) == [100, 64]
