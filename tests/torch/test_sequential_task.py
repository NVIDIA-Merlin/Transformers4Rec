import pytest

torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")
torch_metric = pytest.importorskip("transformers4rec.torch.ranking_metric")
torch_head = pytest.importorskip("transformers4rec.torch.head")

# fixed parameters for tests
METRICS = [
    torch_metric.NDCGAt(top_ks=[2, 5, 10], labels_onehot=True),
    torch_metric.AvgPrecisionAt(top_ks=[2, 5, 10], labels_onehot=True),
]


# Test of output of sequential_task when mf_constrained_embeddings is disabled
def test_sequential_task_output(torch_seq_prediction_head_inputs):
    task = torch_head.SequentialPredictionTask(
        loss=torch.nn.NLLLoss(ignore_index=0),
        metrics=METRICS,
        mf_constrained_embeddings=False,
        input_size=torch_seq_prediction_head_inputs["item_dim"],
        vocab_size=torch_seq_prediction_head_inputs["vocab_size"],
    )
    loss = task.compute_loss(
        inputs=torch_seq_prediction_head_inputs["seq_model_output"],
        targets=torch_seq_prediction_head_inputs["labels_all"],
    )
    metrics = task.calculate_metrics(
        predictions=torch_seq_prediction_head_inputs["seq_model_output"],
        labels=torch_seq_prediction_head_inputs["labels_all"],
    )
    assert all(len(m) == 3 for m in metrics.values())
    assert loss != 0


# Test of output of sequential_task when mf_constrained_embeddings is enabled
def test_sequential_task_output_constrained(torch_seq_prediction_head_inputs):
    task = torch_head.SequentialPredictionTask(
        loss=torch.nn.NLLLoss(ignore_index=0),
        metrics=METRICS,
        mf_constrained_embeddings=True,
        item_embedding_table=torch_seq_prediction_head_inputs["item_embedding_table"],
        input_size=torch_seq_prediction_head_inputs["item_dim"],
        vocab_size=torch_seq_prediction_head_inputs["vocab_size"],
    )
    loss = task.compute_loss(
        inputs=torch_seq_prediction_head_inputs["seq_model_output"],
        targets=torch_seq_prediction_head_inputs["labels_all"],
    )
    metrics = task.calculate_metrics(
        predictions=torch_seq_prediction_head_inputs["seq_model_output"],
        labels=torch_seq_prediction_head_inputs["labels_all"],
    )
    assert all(len(m) == 3 for m in metrics.values())
    assert loss != 0


# Test of output of sequential_task when mf_constrained_embeddings is enabled
def test_build_sequential_task_from_block(torch_seq_prediction_head_link_to_block):
    inputs = torch4rec.features.tabular.TabularFeatures.from_config(
        torch_seq_prediction_head_link_to_block["config"]
    )
    block = torch4rec.block.base.SequentialBlock(inputs, torch.nn.Linear(64, 64))

    task = torch_head.SequentialPredictionTask(
        loss=torch.nn.NLLLoss(ignore_index=0),
        item_id_name="item",
        metrics=METRICS,
        mf_constrained_embeddings=True,
    )
    task.build(block)

    loss = task.compute_loss(
        inputs=torch_seq_prediction_head_link_to_block["seq_model_output"],
        targets=torch_seq_prediction_head_link_to_block["labels_all"],
    )
    metrics = task.calculate_metrics(
        predictions=torch_seq_prediction_head_link_to_block["seq_model_output"],
        labels=torch_seq_prediction_head_link_to_block["labels_all"],
    )
    assert all(len(m) == 3 for m in metrics.values())
    assert loss != 0
