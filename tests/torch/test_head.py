import pytest
import torch

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")
torch_metric = pytest.importorskip("transformers4rec.torch.ranking_metric")
torch_head = pytest.importorskip("transformers4rec.torch.head")
from transformers4rec.config import transformer as tconf
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
    loss = head.compute_loss(body_out, targets)

    assert loss.min() >= 0 and loss.max() <= 1


def test_head_with_multiple_tasks(yoochoose_schema, torch_yoochoose_like):
    input_module = torch4rec.TabularFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, continuous_projection=64, aggregation="concat"
    )

    targets = {
        "classification": pytorch.randint(2, (100,)).float(),
        "regression": pytorch.randint(2, (100,)).float(),
    }

    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    tasks = [
        torch4rec.BinaryClassificationTask("classification"),
        torch4rec.RegressionTask("regression"),
    ]
    head = torch4rec.Head(body, tasks)

    body_out = body(torch_yoochoose_like)
    loss = head.compute_loss(body_out, targets)

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
    head = torch4rec.Head(body, torch4rec.NextItemPredictionTask(), inputs=input_module)

    outputs = head(body(torch_yoochoose_like))

    assert outputs.size()[-1] == input_module.categorical_module.item_embedding_table.num_embeddings


def test_item_prediction_head_weight_tying(yoochoose_schema, torch_yoochoose_like):
    input_module = torch4rec.SequentialTabularFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=64,
        masking="causal",
        )
    
    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = torch4rec.Head(body, torch4rec.NextItemPredictionTask(weight_tying=True), inputs=input_module)
    
    outputs = head(body(torch_yoochoose_like))
    
    assert outputs.size()[-1] == input_module.categorical_module.item_embedding_table.num_embeddings

# Test loss and metrics outputs
@pytest.mark.parametrize("weight_tying", [True, False])
def test_item_prediction_loss_and_metrics(yoochoose_schema, torch_yoochoose_like, weight_tying):
    input_module = torch4rec.SequentialTabularFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=64,
        masking="causal",
        )
    
    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = torch4rec.Head(body, torch4rec.NextItemPredictionTask(weight_tying=weight_tying), inputs=input_module)
    
    body_outputs = body(torch_yoochoose_like)
    outputs = head(body(torch_yoochoose_like))

    trg_flat = input_module.masking.masked_targets.flatten()
    non_pad_mask = trg_flat != input_module.masking.pad_token
    labels_all = torch.masked_select(trg_flat, non_pad_mask)
    
    loss = head.prediction_tasks['0'].compute_loss(
        inputs=body_outputs,
        targets=labels_all,
     )
    
    metrics = head.prediction_tasks['0'].calculate_metrics(
        predictions=body_outputs,
        labels=labels_all
     )
    assert all(len(m) == 3 for m in metrics.values())
    assert loss != 0
    

# Test output formats  
def test_item_prediction_HF_output(yoochoose_schema, torch_yoochoose_like):
    input_module = torch4rec.SequentialTabularFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=64,
        masking="causal",
        )
    
    body = torch4rec.SequentialBlock(input_module, torch4rec.MLPBlock([64]))
    head = torch4rec.Head(body, torch4rec.NextItemPredictionTask(weight_tying=True,
                                                                 hf_format=True),
                          inputs=input_module)
    
    body_outputs = body(torch_yoochoose_like)
    outputs = head(body(torch_yoochoose_like))
    
    assert isinstance(outputs, dict)
    assert [param in outputs for param in ['loss', 
                                           'labels',
                                           'predictions',
                                           'pred_metadata',
                                           'model_outputs']]
# Test output formats  
def test_item_prediction_with_rnn(yoochoose_schema, torch_yoochoose_like):
    input_module = torch4rec.SequentialTabularFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=64,
        masking="causal",
        )
    
    body = torch4rec.SequentialBlock(input_module,
                                     torch4rec.MLPBlock([64]), 
                                     torch4rec.Block(torch.nn.GRU(
                                         input_size=64,
                                         hidden_size=64,
                                         num_layers=2), [None, 20, 64])
                                    )
    head = torch4rec.Head(body, torch4rec.NextItemPredictionTask(weight_tying=True,
                                                                 hf_format=True),
                          inputs=input_module)
    
    body_outputs = body(torch_yoochoose_like)
    outputs = head(body(torch_yoochoose_like))
    
    assert outputs
    assert isinstance(outputs, dict)    
