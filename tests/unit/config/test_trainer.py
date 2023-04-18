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

from transformers import TFTrainingArguments, TrainingArguments

from transformers4rec.config import trainer


def test_torch_trainer_config():
    config = trainer.T4RecTrainingArguments(output_dir=".", predict_top_k=5, learning_rate=0.008)

    assert isinstance(config, TrainingArguments)
    assert config.learning_rate == 0.008
    assert config.predict_top_k == 5


def test_tf_trainer_config():
    config = trainer.T4RecTrainingArgumentsTF(output_dir=".", predict_top_k=5, learning_rate=0.008)

    assert isinstance(config, TFTrainingArguments)
    assert config.learning_rate == 0.008
    assert config.predict_top_k == 5
