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

import transformers4rec.torch as tr

# fixed parameters for tests
METRICS = [
    tr.ranking_metric.NDCGAt(top_ks=[2, 5, 10], labels_onehot=True),
    tr.ranking_metric.AvgPrecisionAt(top_ks=[2, 5, 10], labels_onehot=True),
]


@pytest.mark.parametrize("task", [tr.BinaryClassificationTask, tr.RegressionTask])
def test_simple_heads(torch_tabular_features, torch_tabular_data, task):
    targets = {"target": torch.randint(2, (100,)).float()}

    body = tr.SequentialBlock(torch_tabular_features, tr.MLPBlock([64]))
    head = task("target").to_head(body, torch_tabular_features)

    body_out = body(torch_tabular_data)
    loss = head(body_out, targets=targets, training=True)["loss"]

    assert loss.min() >= 0 and loss.max() <= 1


@pytest.mark.parametrize("task", [tr.BinaryClassificationTask, tr.RegressionTask])
@pytest.mark.parametrize("task_block", [None, tr.MLPBlock([32]), tr.MLPBlock([32]).build([-1, 64])])
@pytest.mark.parametrize("summary", [None, "last", "first", "mean", "cls_index"])
def test_simple_heads_on_sequence(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, task, task_block, summary
):
    inputs = torch_yoochoose_tabular_transformer_features
    if summary:
        targets = {"target": torch.randint(2, (100,)).float()}
    else:
        targets = {"target": torch.randint(2, (100, 20)).float()}

    body = tr.SequentialBlock(inputs, tr.MLPBlock([64]))
    head = task("target", task_block=task_block, summary_type=summary).to_head(body, inputs)

    body_out = body(torch_yoochoose_like)
    loss = head(body_out, targets=targets, training=True)["loss"]

    assert loss.min() >= 0 and loss.max() <= 1


@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        tr.MLPBlock([32]),
        tr.MLPBlock([32]).build([-1, 64]),
        dict(classification=tr.MLPBlock([16]), regression=tr.MLPBlock([20])),
    ],
)
def test_head_with_multiple_tasks(torch_tabular_features, torch_tabular_data, task_blocks):
    targets = {
        "classification": torch.randint(2, (100,)).float(),
        "regression": torch.randint(2, (100,)).float(),
    }

    body = tr.SequentialBlock(torch_tabular_features, tr.MLPBlock([64]))
    tasks = [
        tr.BinaryClassificationTask("classification", task_name="classification"),
        tr.RegressionTask("regression", task_name="regression"),
    ]
    # TODO: how to get targets with no dataloader?
    head = tr.Head(body, tasks, task_blocks=task_blocks)
    optimizer = torch.optim.Adam(head.parameters())

    with torch.set_grad_enabled(mode=True):
        body_out = body(torch_tabular_data)
        output = head(body_out, targets=targets, training=True)
        loss = output["loss"]
        metrics = head.calculate_metrics(output["predictions"], targets=output["labels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.min() >= 0 and loss.max() <= 1
    assert len(metrics.keys()) == 4
    if task_blocks:
        assert head.task_blocks["classification"][0] != head.task_blocks["regression"][0]

        assert not torch.equal(
            head.task_blocks["classification"][0][0].weight,
            head.task_blocks["regression"][0][0].weight,
        )


def test_item_prediction_head(torch_yoochoose_tabular_transformer_features, torch_yoochoose_like):
    input_module = torch_yoochoose_tabular_transformer_features
    body = tr.SequentialBlock(input_module, tr.MLPBlock([64]))
    head = tr.Head(body, tr.NextItemPredictionTask(), inputs=input_module)

    outputs = head(body(torch_yoochoose_like))

    assert (
        outputs["next-item"].size()[-1]
        == input_module.categorical_module.item_embedding_table.num_embeddings
    )


def test_item_prediction_head_weight_tying(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    input_module = torch_yoochoose_tabular_transformer_features
    body = tr.SequentialBlock(input_module, tr.MLPBlock([64]))
    head = tr.Head(body, tr.NextItemPredictionTask(weight_tying=True), inputs=input_module)

    outputs = head(body(torch_yoochoose_like))

    assert (
        list(outputs.values())[0].size()[-1]
        == input_module.categorical_module.item_embedding_table.num_embeddings
    )


# Test loss and metrics outputs
@pytest.mark.parametrize("weight_tying", [True, False])
def test_item_prediction_loss_and_metrics(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, weight_tying
):
    input_module = torch_yoochoose_tabular_transformer_features
    body = tr.SequentialBlock(input_module, tr.MLPBlock([64]))
    head = tr.Head(body, tr.NextItemPredictionTask(weight_tying=weight_tying), inputs=input_module)

    body_outputs = body(torch_yoochoose_like, testing=True)

    trg_flat = input_module.masking.masked_targets.flatten()
    non_pad_mask = trg_flat != input_module.masking.padding_idx
    labels_all = torch.masked_select(trg_flat, non_pad_mask)

    output = head.prediction_task_dict["next-item"](
        inputs=body_outputs,
        targets=labels_all,
        testing=True,
    )
    loss = output["loss"]

    metrics = head.prediction_task_dict["next-item"].calculate_metrics(
        predictions=output["predictions"], targets=output["labels"]
    )
    assert all(len(m) == 2 for m in metrics.values())
    assert loss != 0


# Test output formats
def test_item_prediction_HF_output(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    input_module = torch_yoochoose_tabular_transformer_features

    body = tr.SequentialBlock(input_module, tr.MLPBlock([64]))
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True),
        inputs=input_module,
    )

    outputs = head(body(torch_yoochoose_like, training=True), training=True)

    assert isinstance(outputs, dict)
    assert [
        param in outputs
        for param in ["loss", "labels", "predictions", "pred_metadata", "model_outputs"]
    ]


def test_head_not_inferring_output_size_body(torch_tabular_features):
    with pytest.raises(ValueError) as excinfo:
        body = tr.SequentialBlock(torch_tabular_features, torch.nn.Dropout(0.5))
        tr.Head(
            body,
            tr.BinaryClassificationTask(),
        )

        assert "Can't infer output-size of the body" in str(excinfo.value)


def test_item_prediction_head_with_wrong_body(torch_tabular_features):
    with pytest.raises(ValueError) as excinfo:
        body = tr.SequentialBlock(torch_tabular_features, torch.nn.Dropout(0.5))
        tr.Head(
            body,
            tr.NextItemPredictionTask(),
        )

        assert "NextItemPredictionTask needs a 3-dim vector as input, found:" in str(excinfo.value)


def test_item_prediction_head_with_input_size(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    input_module = torch_yoochoose_tabular_transformer_features

    body = tr.SequentialBlock(
        input_module,
        tr.MLPBlock([64]),
        torch.nn.GRU(input_size=64, hidden_size=64, num_layers=2),
        output_size=[None, 20, 64],
    )
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True),
        inputs=input_module,
    )

    outputs = head(body(torch_yoochoose_like, training=True), training=True)

    assert outputs


# Test output formats
def test_item_prediction_with_rnn(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    input_module = torch_yoochoose_tabular_transformer_features

    body = tr.SequentialBlock(
        input_module,
        tr.MLPBlock([64]),
        tr.Block(torch.nn.GRU(input_size=64, hidden_size=64, num_layers=2), [None, 20, 64]),
    )
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True),
        inputs=input_module,
    )

    outputs = head(body(torch_yoochoose_like, training=True), training=True)

    assert isinstance(outputs, dict)
    assert list(outputs.keys()) == [
        "loss",
        "labels",
        "predictions",
    ]
