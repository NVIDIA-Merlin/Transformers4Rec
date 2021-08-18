import pytest

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")
torch_metric = pytest.importorskip("transformers4rec.torch.ranking_metric")
torch_head = pytest.importorskip("transformers4rec.torch.head")

# fixed parameters for tests
METRICS = [
    torch_metric.NDCGAt(top_ks=[2, 5, 10], labels_onehot=True),
    torch_metric.AvgPrecisionAt(top_ks=[2, 5, 10], labels_onehot=True),
]


@pytest.mark.parametrize("task", [torch4rec.BinaryClassificationTask, torch4rec.RegressionTask])
def test_simple_heads(yoochoose_schema, torch_yoochoose_like, task):
    input_module = torch4rec.TabularFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, continuous_projection=64, aggregation="concat"
    )

    targets = {"target": pytorch.randint(2, (100,)).float()}

    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = task("target").to_head(body, input_module)

    body_out = body(torch_yoochoose_like)
    outputs, loss = head(body_out), head.compute_loss(body_out, targets)

    assert outputs.min() >= 0 and outputs.max() <= 1
    assert loss.min() >= 0 and loss.max() <= 1


def test_item_prediction_head(yoochoose_schema, torch_yoochoose_like):
    input_module = torch4rec.SequentialTabularFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="causal",
    )

    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = torch4rec.Head(body, torch4rec.ItemPredictionTask(), inputs=input_module)

    outputs = head(body(torch_yoochoose_like))

    assert outputs.size()[-1] == input_module.categorical_module.item_embedding_table.num_embeddings


# # Test of output of sequential_task when mf_constrained_embeddings is disabled
# def test_sequential_task_output(torch_seq_prediction_head_inputs):
#     task = torch_head.SequentialPredictionTask(
#         loss=pytorch.nn.NLLLoss(ignore_index=0),
#         metrics=METRICS,
#         mf_constrained_embeddings=False,
#         input_size=torch_seq_prediction_head_inputs["item_dim"],
#         vocab_size=torch_seq_prediction_head_inputs["vocab_size"],
#     )
#     loss = task.compute_loss(
#         inputs=torch_seq_prediction_head_inputs["seq_model_output"],
#         targets=torch_seq_prediction_head_inputs["labels_all"],
#     )
#     metrics = task.calculate_metrics(
#         predictions=torch_seq_prediction_head_inputs["seq_model_output"],
#         labels=torch_seq_prediction_head_inputs["labels_all"],
#     )
#     assert all(len(m) == 3 for m in metrics.values())
#     assert loss != 0
#
#
# # Test of output of sequential_task when mf_constrained_embeddings is enabled
# def test_sequential_task_output_constrained(torch_seq_prediction_head_inputs):
#     task = torch_head.SequentialPredictionTask(
#         loss=pytorch.nn.NLLLoss(ignore_index=0),
#         metrics=METRICS,
#         mf_constrained_embeddings=True,
#         item_embedding_table=torch_seq_prediction_head_inputs["item_embedding_table"],
#         input_size=torch_seq_prediction_head_inputs["item_dim"],
#         vocab_size=torch_seq_prediction_head_inputs["vocab_size"],
#     )
#     loss = task.compute_loss(
#         inputs=torch_seq_prediction_head_inputs["seq_model_output"],
#         targets=torch_seq_prediction_head_inputs["labels_all"],
#     )
#     metrics = task.calculate_metrics(
#         predictions=torch_seq_prediction_head_inputs["seq_model_output"],
#         labels=torch_seq_prediction_head_inputs["labels_all"],
#     )
#     assert all(len(m) == 3 for m in metrics.values())
#     assert loss != 0


# TODO: We need the sequential aggregator to fix this test
# Test of output of sequential_task when mf_constrained_embeddings is enabled
# def test_build_sequential_task_from_block(yoochoose_schema):
#     inputs = torch4rec.features.tabular.TabularFeatures.from_schema(
#         yoochoose_schema
#     )
#
#     block = torch4rec.block.base.SequentialBlock(inputs, pytorch.nn.Linear(64, 64))

#
# task = torch_head.SequentialPredictionTask(
#     loss=pytorch.nn.NLLLoss(ignore_index=0),
#     item_id_name="item",
#     metrics=METRICS,
#     mf_constrained_embeddings=True,
# )
# task.build(block)
#
# loss = task.compute_loss(
#     inputs=torch_seq_prediction_head_link_to_block["seq_model_output"],
#     targets=torch_seq_prediction_head_link_to_block["labels_all"],
# )
# metrics = task.calculate_metrics(
#     predictions=torch_seq_prediction_head_link_to_block["seq_model_output"],
#     labels=torch_seq_prediction_head_link_to_block["labels_all"],
# )
# assert all(len(m) == 3 for m in metrics.values())
# assert loss != 0
