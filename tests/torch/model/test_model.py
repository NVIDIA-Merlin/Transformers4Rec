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

from transformers4rec.config import transformer as tconf

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")

if pytorch.cuda.is_available():
    devices = ["cpu", "cuda"]
else:
    devices = ["cpu"]


def test_simple_model(torch_yoochoose_tabular_features, torch_yoochoose_like):
    targets = {"target": pytorch.randint(2, (100,)).float()}

    inputs = torch_yoochoose_tabular_features
    body = torch4rec.SequentialBlock(inputs, torch4rec.MLPBlock([64]))
    model = torch4rec.BinaryClassificationTask("target").to_model(body, inputs)

    dataset = [(torch_yoochoose_like, targets)]
    losses = model.fit(dataset, num_epochs=5)
    metrics = model.evaluate(dataset, mode="eval")

    assert list(metrics.keys()) == ["precision", "recall", "accuracy"]
    assert len(losses) == 5
    assert all(loss.min() >= 0 and loss.max() <= 1 for loss in losses)


@pytest.mark.parametrize("task", [torch4rec.BinaryClassificationTask, torch4rec.RegressionTask])
def test_sequential_prediction_model(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, task
):
    inputs = torch_yoochoose_tabular_transformer_features

    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )
    body = torch4rec.SequentialBlock(
        inputs, torch4rec.MLPBlock([64]), torch4rec.TransformerBlock(transformer_config)
    )

    head_1 = torch4rec.Head(
        body,
        torch4rec.NextItemPredictionTask(weight_tying=True, hf_format=True),
        inputs=inputs,
    )
    head_2 = task("target", summary_type="mean").to_head(body, inputs)

    model = torch4rec.Model(head_1, head_2)
    output = model(torch_yoochoose_like)

    assert isinstance(output, dict)
    assert len(list(output.keys())) == 2


def test_model_with_multiple_heads_and_tasks(
    torch_yoochoose_tabular_features,
    torch_yoochoose_tabular_transformer_features,
    torch_yoochoose_like,
):
    # Tabular classification and regression tasks
    targets = {
        "classification": pytorch.randint(2, (100,)).float(),
        "regression": pytorch.randint(2, (100,)).float(),
    }
    body = torch4rec.SequentialBlock(torch_yoochoose_tabular_features, torch4rec.MLPBlock([64]))
    tasks = [
        torch4rec.BinaryClassificationTask("classification"),
        torch4rec.RegressionTask("regression"),
    ]
    head_1 = torch4rec.Head(body, tasks)

    # Session-based classification and regression tasks
    targets_2 = {
        "classification_session": pytorch.randint(2, (100,)).float(),
        "regression_session": pytorch.randint(2, (100,)).float(),
    }
    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )
    body_2 = torch4rec.SequentialBlock(
        torch_yoochoose_tabular_transformer_features,
        torch4rec.MLPBlock([64]),
        torch4rec.TransformerBlock(transformer_config),
    )
    tasks_2 = [
        torch4rec.BinaryClassificationTask("classification_session", summary_type="last"),
        torch4rec.RegressionTask("regression_session", summary_type="mean"),
    ]
    head_2 = torch4rec.Head(body_2, tasks_2)

    # Final model with two heads
    model = torch4rec.Model(head_1, head_2)

    # launch training
    targets.update(targets_2)
    dataset = [(torch_yoochoose_like, targets)]
    losses = model.fit(dataset, num_epochs=5)
    metrics = model.evaluate(dataset)

    assert list(metrics.keys()) == [
        "eval_classification",
        "eval_regression",
        "eval_classification_session",
        "eval_regression_session",
    ]
    assert len(losses) == 5
    assert all(loss is not None for loss in losses)


def test_multi_head_model_wrong_weights(torch_yoochoose_tabular_features, torch_yoochoose_like):
    with pytest.raises(ValueError) as excinfo:
        inputs = torch_yoochoose_tabular_features
        body = torch4rec.SequentialBlock(inputs, torch4rec.MLPBlock([64]))

        head_1 = torch4rec.BinaryClassificationTask("classification").to_head(body, inputs)
        head_2 = torch4rec.RegressionTask("regression", summary_type="mean").to_head(body, inputs)

        torch4rec.Model(head_1, head_2, head_weights=[0.4])

    assert "`head_weights` needs to have the same length " "as the number of heads" in str(
        excinfo.value
    )


config_classes = [
    tconf.XLNetConfig,
    # TODO: Add Electra when HF fix is released
    # tconf.ElectraConfig,
    tconf.LongformerConfig,
    tconf.GPT2Config,
]


@pytest.mark.parametrize("config_cls", config_classes)
def test_transformer_torch_model_from_config(yoochoose_schema, torch_yoochoose_like, config_cls):
    transformer_config = config_cls.build(128, 4, 2, 20)

    input_module = torch4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=128,
        masking="causal",
    )
    task = torch4rec.BinaryClassificationTask("classification")
    model = transformer_config.to_torch_model(input_module, task)

    out = model(torch_yoochoose_like)

    assert out.size()[0] == 100
    assert len(out.size()) == 1


@pytest.mark.parametrize("config_cls", config_classes)
def test_item_prediction_transformer_torch_model_from_config(
    yoochoose_schema, torch_yoochoose_like, config_cls
):
    transformer_config = config_cls.build(128, 4, 2, 20)

    input_module = torch4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=128,
        masking="causal",
    )

    task = torch4rec.NextItemPredictionTask()
    model = transformer_config.to_torch_model(input_module, task)

    out = model(torch_yoochoose_like)

    assert out.size()[1] == task.target_dim
    assert len(out.size()) == 2


@pytest.mark.parametrize("device", devices)
def test_set_model_to_device(
    torch_yoochoose_like, torch_yoochoose_next_item_prediction_model, device
):
    model = torch_yoochoose_next_item_prediction_model
    model.to(device)

    assert model.heads[0].body.inputs.masking.masked_item_embedding.device.type == device
    assert next(model.parameters()).device.type == device

    inputs = {k: v.to(device) for k, v in torch_yoochoose_like.items()}
    assert model(inputs)


@pytest.mark.parametrize("masking", ["causal", "mlm", "plm", "rtd"])
def test_eval_metrics_with_masking(torch_yoochoose_like, yoochoose_schema, masking):
    transformer_config = tconf.XLNetConfig.build(64, 4, 2, 20)
    input_module = torch4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=64,
        masking=masking,
    )
    task = torch4rec.NextItemPredictionTask(hf_format=True)
    model = transformer_config.to_torch_model(input_module, task)
    out = model(torch_yoochoose_like)
    result = model.calculate_metrics(
        inputs=out["predictions"], targets=out["labels"], call_body=False, forward=False
    )
    assert result is not None


@pytest.mark.parametrize("d_model", [32, 64, 128])
def test_with_d_model_different_from_item_dim(torch_yoochoose_like, yoochoose_schema, d_model):
    transformer_config = tconf.XLNetConfig.build(d_model, 4, 2, 20)
    input_module = torch4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=d_model,
        masking="mlm",
    )
    task = torch4rec.NextItemPredictionTask(hf_format=True, weight_tying=True)
    model = transformer_config.to_torch_model(input_module, task)
    assert model(torch_yoochoose_like)
