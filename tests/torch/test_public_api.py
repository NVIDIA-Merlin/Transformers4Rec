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

tr = pytest.importorskip("transformers4rec.torch")

transformer_config_names = [
    "AlbertConfig",
    "ElectraConfig",
    "GPT2Config",
    "LongformerConfig",
    "ReformerConfig",
    "XLNetConfig",
]


@pytest.mark.parametrize("config", transformer_config_names)
def test_transformer_config_imports(config):
    config_cls = getattr(tr, config)

    assert issubclass(config_cls, tr.T4RecConfig)


def test_torch_import():
    pytest.importorskip("torch")

    assert tr is not None
    assert tr.Head is not None
    assert tr.Model is not None
