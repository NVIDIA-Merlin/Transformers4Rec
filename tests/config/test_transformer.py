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
from transformers import PreTrainedModel, TFPreTrainedModel

from transformers4rec.config import transformer as tconf

config_classes = [
    tconf.ReformerConfig,
    tconf.XLNetConfig,
    tconf.ElectraConfig,
    tconf.LongformerConfig,
    tconf.GPT2Config,
]


@pytest.mark.parametrize("config_cls", config_classes)
def test_to_hugginface_torch_model(config_cls):
    config = config_cls.build(100, 4, 2, 20)

    model = config.to_huggingface_torch_model()

    assert isinstance(model, PreTrainedModel)


@pytest.mark.parametrize("config_cls", list(set(config_classes) - {tconf.ReformerConfig}))
def test_to_hugginface_tf_model(config_cls):
    config = config_cls.build(100, 4, 2, 20)

    model = config.to_huggingface_tf_model()

    assert isinstance(model, TFPreTrainedModel)
