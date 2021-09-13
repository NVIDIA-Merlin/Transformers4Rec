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
import torch

tr = pytest.importorskip("transformers4rec.torch")

if torch.cuda.is_available():
    devices = ["cpu", "cuda"]
else:
    devices = ["cpu"]


def test_filter_features(torch_con_features):
    features = ["con_a", "con_b"]
    con = tr.FilterFeatures(features)(torch_con_features)

    assert list(con.keys()) == features


def test_as_tabular(torch_con_features):
    name = "tabular"
    con = tr.AsTabular(name)(torch_con_features)

    assert list(con.keys()) == [name]


def test_tabular_module(torch_con_features):
    class _DummyTabular(tr.TabularModule):
        def forward(self, inputs):
            return inputs

    tabular = _DummyTabular()

    assert tabular(torch_con_features) == torch_con_features
    assert tabular(torch_con_features, aggregation="concat").size()[1] == 6
    assert tabular(torch_con_features, aggregation=tr.ConcatFeatures()).size()[1] == 6

    tabular_concat = _DummyTabular(aggregation="concat", pre="ssn")
    assert tabular_concat(torch_con_features).size()[1] == 6

    tab_a = ["con_a"] >> _DummyTabular()
    tab_b = tr.SequentialBlock(["con_b"], _DummyTabular())

    assert tab_a(torch_con_features, merge_with=tab_b, aggregation="stack").size()[1] == 1
    assert (tab_a + tab_b)(torch_con_features, aggregation="concat").size()[1] == 2


@pytest.mark.parametrize("device", devices)
def test_tabular_module_to_device(yoochoose_schema, device):
    schema = yoochoose_schema
    tab_module = tr.TabularSequenceFeatures.from_schema(
        schema, max_sequence_length=20, aggregation="concat"
    )
    tab_module.to(device)

    # Flatten nested torch modules
    def flatten(el):
        flattened = [flatten(children) for children in el.children()]
        res = [el]
        for c in flattened:
            res += c
        return res

    flatten_layers = flatten(tab_module)

    # Check params of pytorch modules are moved to appropriate device
    assert all(
        [
            list(el.parameters())[-1].device.type == device
            for el in flatten_layers
            if list(el.parameters())
        ]
    )
