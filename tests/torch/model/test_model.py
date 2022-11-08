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

import os
import tempfile

import pytest
import torch
from merlin.schema import Schema as Core_Schema

from transformers4rec.config import transformer as tconf

pytorch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers4rec.torch")

if pytorch.cuda.is_available():
    devices = ["cpu", "cuda"]
else:
    devices = ["cpu"]


def test_simple_model(torch_tabular_features, torch_tabular_data):
    targets = {"target": pytorch.randint(2, (100,)).float()}

    inputs = torch_tabular_features
    body = tr.SequentialBlock(inputs, tr.MLPBlock([64]))
    model = tr.BinaryClassificationTask("target").to_model(body, inputs)

    dataset = [(torch_tabular_data, targets)]
    losses = model.fit(dataset, num_epochs=5)
    metrics = model.evaluate(dataset, mode="eval")

    in_schema = model.input_schema
    assert isinstance(in_schema, Core_Schema)
    assert set(in_schema.column_names) == set(inputs.schema.column_names)
    out_schema = model.output_schema
    assert isinstance(out_schema, Core_Schema)
    assert len(out_schema) == 1
    assert out_schema.column_names[0] == "target/binary_classification_task"

    # assert list(metrics.keys()) == ["precision", "recall", "accuracy"]
    assert len(metrics) == 3
    assert len(losses) == 5
    assert all(loss.min() >= 0 and loss.max() <= 1 for loss in losses)


@pytest.mark.parametrize("task", [tr.BinaryClassificationTask, tr.RegressionTask])
def test_sequential_prediction_model(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, task
):
    inputs = torch_yoochoose_tabular_transformer_features

    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )
    body = tr.SequentialBlock(inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config))

    head_1 = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True, hf_format=True),
        inputs=inputs,
    )
    head_2 = task("target", summary_type="mean").to_head(body, inputs)

    model = tr.Model(head_1, head_2)
    output = model(torch_yoochoose_like)

    assert isinstance(output, dict)
    assert len(list(output.keys())) == 2


def test_model_with_multiple_heads_and_tasks(
    yoochoose_schema,
    torch_yoochoose_tabular_transformer_features,
    torch_yoochoose_like,
):
    # Tabular classification and regression tasks
    targets = {
        "classification": pytorch.randint(2, (100,)).float(),
        "regression": pytorch.randint(2, (100,)).float(),
    }

    non_sequential_features_schema = yoochoose_schema.select_by_name(["user_age", "user_country"])

    tabular_features = tr.TabularFeatures.from_schema(
        non_sequential_features_schema,
        max_sequence_length=20,
        continuous_projection=64,
        aggregation="concat",
    )

    body = tr.SequentialBlock(tabular_features, tr.MLPBlock([64]))
    tasks = [
        tr.BinaryClassificationTask("classification"),
        tr.RegressionTask("regression"),
    ]
    head_1 = tr.Head(body, tasks)

    # Session-based classification and regression tasks
    targets_2 = {
        "classification_session": pytorch.randint(2, (100,)).float(),
        "regression_session": pytorch.randint(2, (100,)).float(),
    }
    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )
    body_2 = tr.SequentialBlock(
        torch_yoochoose_tabular_transformer_features,
        tr.MLPBlock([64]),
        tr.TransformerBlock(transformer_config),
    )
    tasks_2 = [
        tr.BinaryClassificationTask("classification_session", summary_type="last"),
        tr.RegressionTask("regression_session", summary_type="mean"),
    ]
    head_2 = tr.Head(body_2, tasks_2)

    # Final model with two heads
    model = tr.Model(head_1, head_2)

    # get output of the model
    output = model(torch_yoochoose_like)

    # launch training
    targets.update(targets_2)
    dataset = [(torch_yoochoose_like, targets)]
    losses = model.fit(dataset, num_epochs=5)
    metrics = model.evaluate(dataset)

    assert list(metrics.keys()) == [
        "eval_classification/binary_classification_task",
        "eval_regression/regression_task",
        "eval_classification_session/binary_classification_task",
        "eval_regression_session/regression_task",
    ]
    assert len(losses) == 5
    assert all(loss is not None for loss in losses)

    in_schema = model.input_schema
    assert isinstance(in_schema, Core_Schema)
    assert set(in_schema.column_names) == set(
        non_sequential_features_schema.column_names
        + torch_yoochoose_tabular_transformer_features.schema.column_names
    )
    # Get output schema
    out_schema = model.output_schema
    assert isinstance(out_schema, Core_Schema)
    assert set(out_schema.column_names) == set(output.keys())


def test_multi_head_model_wrong_weights(torch_tabular_features, torch_yoochoose_like):
    with pytest.raises(ValueError) as excinfo:
        inputs = torch_tabular_features
        body = tr.SequentialBlock(inputs, tr.MLPBlock([64]))

        head_1 = tr.BinaryClassificationTask("classification").to_head(body, inputs)
        head_2 = tr.RegressionTask("regression", summary_type="mean").to_head(body, inputs)

        tr.Model(head_1, head_2, head_weights=[0.4])

    assert "`head_weights` needs to have the same length " "as the number of heads" in str(
        excinfo.value
    )


config_classes = [
    tconf.XLNetConfig,
    # TODO: Add support of Electra
    tconf.AlbertConfig,
    tconf.LongformerConfig,
    tconf.GPT2Config,
]


@pytest.mark.parametrize("config_cls", config_classes)
def test_transformer_torch_model_from_config(yoochoose_schema, torch_yoochoose_like, config_cls):
    transformer_config = config_cls.build(128, 4, 2, 20)

    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=128,
        masking="causal",
    )
    task = tr.BinaryClassificationTask("classification")
    model = transformer_config.to_torch_model(input_module, task)

    out = model(torch_yoochoose_like)

    assert out.size()[0] == 100
    assert len(out.size()) == 1


@pytest.mark.parametrize("config_cls", config_classes)
def test_item_prediction_transformer_torch_model_from_config(
    yoochoose_schema, torch_yoochoose_like, config_cls
):
    transformer_config = config_cls.build(128, 4, 2, 20)

    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=128,
        masking="causal",
    )

    task = tr.NextItemPredictionTask()
    model = transformer_config.to_torch_model(input_module, task)

    out = model(torch_yoochoose_like)

    assert out.size()[1] == task.target_dim
    assert len(out.size()) == 2

    in_schema = model.input_schema
    assert isinstance(in_schema, Core_Schema)
    assert set(in_schema.column_names).issubset(yoochoose_schema.column_names)
    # Get output schema
    out_schema = model.output_schema
    assert isinstance(out_schema, Core_Schema)
    assert len(out_schema) == 1
    assert out_schema.column_names[0] == "next-item"


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
    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=64,
        masking=masking,
    )
    task = tr.NextItemPredictionTask(hf_format=True)
    model = transformer_config.to_torch_model(input_module, task)
    out = model(torch_yoochoose_like)
    result = model.calculate_metrics(
        inputs=out["predictions"], targets=out["labels"], call_body=False, forward=False
    )
    assert result is not None


@pytest.mark.parametrize("d_model", [32, 64, 128])
def test_with_d_model_different_from_item_dim(torch_yoochoose_like, yoochoose_schema, d_model):
    transformer_config = tconf.XLNetConfig.build(d_model, 4, 2, 20)
    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=d_model,
        masking="mlm",
    )
    task = tr.NextItemPredictionTask(hf_format=True, weight_tying=True)
    model = transformer_config.to_torch_model(input_module, task)
    assert model(torch_yoochoose_like)


@pytest.mark.parametrize("masking", ["causal", "mlm", "plm", "rtd"])
def test_output_shape_mode_eval(torch_yoochoose_like, yoochoose_schema, masking):
    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        d_output=64,
        masking=masking,
    )
    prediction_task = tr.NextItemPredictionTask(hf_format=True, weight_tying=True)
    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=8, n_layer=2, total_seq_length=20
    )
    model = transformer_config.to_torch_model(input_module, prediction_task)

    out = model(torch_yoochoose_like, training=False)
    assert out["predictions"].shape[0] == torch_yoochoose_like["item_id/list"].size(0)


def test_save_next_item_prediction_model(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    inputs = torch_yoochoose_tabular_transformer_features
    transformer_config = tconf.XLNetConfig.build(100, 4, 2, 20)
    task = tr.NextItemPredictionTask(hf_format=True, weight_tying=True)
    model = transformer_config.to_torch_model(inputs, task)
    output = model(torch_yoochoose_like, training=False)
    assert isinstance(output, dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        assert "t4rec_model_class.pkl" in os.listdir(tmpdir)
        loaded_model = model.load(tmpdir)
        # deactivate the hf_format model to get the tensor of predictions as output
        # instead of a dictionary of three tensors `loss, labels, and predictions`
        loaded_model.hf_format = False

    output = loaded_model(torch_yoochoose_like, training=False)
    assert isinstance(output, torch.Tensor)

    in_schema = loaded_model.input_schema

    assert isinstance(in_schema, Core_Schema)
    assert set(in_schema.column_names) == set(inputs.schema.column_names)

    out_schema = loaded_model.output_schema
    assert isinstance(out_schema, Core_Schema)
    assert len(out_schema) == 1
    assert out_schema.column_names[0] == "next-item"
