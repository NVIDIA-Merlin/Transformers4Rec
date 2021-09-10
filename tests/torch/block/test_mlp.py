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

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_mlp_block(yoochoose_schema, torch_yoochoose_like):
    tab_module = torch4rec.TabularFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, aggregation="concat"
    )

    block = tab_module >> torch4rec.MLPBlock([64, 32])

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 2
    assert outputs.shape[-1] == 32
