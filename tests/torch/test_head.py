import pytest

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")
torch_metric = pytest.importorskip("transformers4rec.torch.ranking_metric")

# fixed parameters for tests
METRICS = [
    torch_metric.NDCGAt(top_ks=[2, 5, 10], labels_onehot=True),
    torch_metric.AvgPrecisionAt(top_ks=[2, 5, 10], labels_onehot=True),
]


@pytest.mark.parametrize("task", [torch4rec.BinaryClassificationTask, torch4rec.RegressionTask])
def test_simple_heads(torch_yoochoose_tabular_features, torch_yoochoose_like, task):
    targets = {"target": pytorch.randint(2, (100,)).float()}

    body = torch4rec.SequentialBlock(torch_yoochoose_tabular_features, torch4rec.MLPBlock([64]))
    head = task("target").to_head(body, torch_yoochoose_tabular_features)

    body_out = body(torch_yoochoose_like)
    loss = head.compute_loss(body_out, targets)

    assert loss.min() >= 0 and loss.max() <= 1


@pytest.mark.parametrize("task", [torch4rec.BinaryClassificationTask, torch4rec.RegressionTask])
@pytest.mark.parametrize(
    "task_block", [None, torch4rec.MLPBlock([32]), torch4rec.MLPBlock([32]).build([-1, 64])]
)
@pytest.mark.parametrize("summary", ["last", "first", "mean", "cls_index"])
def test_simple_heads_on_sequence(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, task, task_block, summary
):
    inputs = torch_yoochoose_tabular_transformer_features
    targets = {"target": pytorch.randint(2, (100,)).float()}

    body = torch4rec.SequentialBlock(inputs, torch4rec.MLPBlock([64]))
    head = task("target", task_block=task_block, summary_type=summary).to_head(body, inputs)

    body_out = body(torch_yoochoose_like)
    loss = head.compute_loss(body_out, targets)

    assert loss.min() >= 0 and loss.max() <= 1


@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        torch4rec.MLPBlock([32]),
        torch4rec.MLPBlock([32]).build([-1, 64]),
        dict(classification=torch4rec.MLPBlock([16]), regression=torch4rec.MLPBlock([20])),
    ],
)
def test_head_with_multiple_tasks(
    torch_yoochoose_tabular_features, torch_yoochoose_like, task_blocks
):
    targets = {
        "classification": pytorch.randint(2, (100,)).float(),
        "regression": pytorch.randint(2, (100,)).float(),
    }

    body = torch4rec.SequentialBlock(torch_yoochoose_tabular_features, torch4rec.MLPBlock([64]))
    tasks = [
        torch4rec.BinaryClassificationTask("classification"),
        torch4rec.RegressionTask("regression"),
    ]
    head = torch4rec.Head(body, tasks, task_blocks=task_blocks)
    optimizer = pytorch.optim.Adam(head.parameters())

    with pytorch.set_grad_enabled(mode=True):
        body_out = body(torch_yoochoose_like)
        loss = head.compute_loss(body_out, targets)
        metrics = head.calculate_metrics(body_out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.min() >= 0 and loss.max() <= 1
    assert list(metrics.keys()) == ["classification", "regression"]
    assert list(metrics["classification"].keys()) == ["val_precision", "val_recall", "val_accuracy"]
    assert list(metrics["regression"].keys()) == ["val_meansquarederror"]
    if task_blocks:
        assert head.task_blocks["classification"][0] != head.task_blocks["regression"][0]

        assert not pytorch.equal(
            head.task_blocks["classification"][0][0].weight,
            head.task_blocks["regression"][0][0].weight,
        )


def test_item_prediction_head(torch_yoochoose_tabular_transformer_features, torch_yoochoose_like):
    input_module = torch_yoochoose_tabular_transformer_features
    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = torch4rec.Head(body, torch4rec.NextItemPredictionTask(), inputs=input_module)

    outputs = head(body(torch_yoochoose_like))

    assert outputs.size()[-1] == input_module.categorical_module.item_embedding_table.num_embeddings


def test_item_prediction_head_weight_tying(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    input_module = torch_yoochoose_tabular_transformer_features
    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = torch4rec.Head(
        body, torch4rec.NextItemPredictionTask(weight_tying=True), inputs=input_module
    )

    outputs = head(body(torch_yoochoose_like))

    assert outputs.size()[-1] == input_module.categorical_module.item_embedding_table.num_embeddings


# Test loss and metrics outputs
@pytest.mark.parametrize("weight_tying", [True, False])
def test_item_prediction_loss_and_metrics(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, weight_tying
):
    input_module = torch_yoochoose_tabular_transformer_features
    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = torch4rec.Head(
        body, torch4rec.NextItemPredictionTask(weight_tying=weight_tying), inputs=input_module
    )

    body_outputs = body(torch_yoochoose_like)

    trg_flat = input_module.masking.masked_targets.flatten()
    non_pad_mask = trg_flat != input_module.masking.pad_token
    labels_all = pytorch.masked_select(trg_flat, non_pad_mask)

    loss = head.prediction_tasks["0"].compute_loss(
        inputs=body_outputs,
        targets=labels_all,
    )

    metrics = head.prediction_tasks["0"].calculate_metrics(
        predictions=body_outputs, targets=labels_all
    )
    assert all(len(m) == 3 for m in metrics.values())
    assert loss != 0


# Test output formats
def test_item_prediction_HF_output(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    input_module = torch_yoochoose_tabular_transformer_features

    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = torch4rec.Head(
        body,
        torch4rec.NextItemPredictionTask(weight_tying=True, hf_format=True),
        inputs=input_module,
    )

    outputs = head(body(torch_yoochoose_like))

    assert isinstance(outputs, dict)
    assert [
        param in outputs
        for param in ["loss", "labels", "predictions", "pred_metadata", "model_outputs"]
    ]


def test_head_not_inferring_output_size_body(torch_yoochoose_tabular_features):
    with pytest.raises(ValueError) as excinfo:
        body = torch4rec.SequentialBlock(torch_yoochoose_tabular_features, pytorch.nn.Dropout(0.5))
        torch4rec.Head(
            body,
            torch4rec.BinaryClassificationTask(),
        )

        assert "Can't infer output-size of the body" in str(excinfo.value)


def test_item_prediction_head_with_wrong_body(torch_yoochoose_tabular_features):
    with pytest.raises(ValueError) as excinfo:
        body = torch4rec.SequentialBlock(torch_yoochoose_tabular_features, pytorch.nn.Dropout(0.5))
        torch4rec.Head(
            body,
            torch4rec.NextItemPredictionTask(),
        )

        assert "NextItemPredictionTask needs a 3-dim vector as input, found:" in str(excinfo.value)


def test_item_prediction_head_with_input_size(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    input_module = torch_yoochoose_tabular_transformer_features

    body = torch4rec.SequentialBlock(
        input_module,
        torch4rec.MLPBlock([64]),
        pytorch.nn.GRU(input_size=64, hidden_size=64, num_layers=2),
    )
    head = torch4rec.Head(
        body,
        torch4rec.NextItemPredictionTask(weight_tying=True, hf_format=True),
        inputs=input_module,
        body_output_size=[None, 20, 64],
    )

    outputs = head(body(torch_yoochoose_like))

    assert outputs


# Test output formats
def test_item_prediction_with_rnn(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like
):
    input_module = torch_yoochoose_tabular_transformer_features

    body = torch4rec.SequentialBlock(
        input_module,
        torch4rec.MLPBlock([64]),
        torch4rec.Block(
            pytorch.nn.GRU(input_size=64, hidden_size=64, num_layers=2), [None, 20, 64]
        ),
    )
    head = torch4rec.Head(
        body,
        torch4rec.NextItemPredictionTask(weight_tying=True, hf_format=True),
        inputs=input_module,
    )

    outputs = head(body(torch_yoochoose_like))

    assert isinstance(outputs, dict)
    assert list(outputs.keys()) == [
        "loss",
        "labels",
        "predictions",
        "pred_metadata",
        "model_outputs",
    ]
