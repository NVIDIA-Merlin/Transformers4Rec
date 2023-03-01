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

import transformers4rec.torch as tr
from transformers4rec.config import transformer as tconf
from transformers4rec.torch.experimental import PostContextFusion


@pytest.mark.parametrize("fusion_aggregation", ["concat", "elementwise-mul", "elementwise-sum"])
def test_post_fusion_context_block(yoochoose_schema, torch_yoochoose_like, fusion_aggregation):
    tab_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        aggregation="concat",
        d_output=64,
        masking="causal",
    )

    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )
    sequential_block = tr.SequentialBlock(
        tab_module,
        tr.TransformerBlock(transformer_config, masking=tab_module.masking),
    )

    post_context = tr.SequenceEmbeddingFeatures.from_schema(
        yoochoose_schema.select_by_name("category/list"), aggregation="concat"
    )

    post_fusion_block = PostContextFusion(
        sequential_block, post_context, fusion_aggregation=fusion_aggregation
    )

    outputs = post_fusion_block(torch_yoochoose_like)

    assert outputs.ndim == 3
    if fusion_aggregation == "concat":
        assert outputs.shape[-1] == 128
    else:
        assert outputs.shape[-1] == 64
