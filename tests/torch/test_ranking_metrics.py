import pytest

torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")
torch_metric = pytest.importorskip("transformers4rec.torch.ranking_metric")

# fixed parameters for tests
list_metrics = list(torch_metric.ranking_metrics_registry.keys())


# Test length of output equal to number of cutoffs
@pytest.mark.parametrize("metric", list_metrics)
def test_avg_precision_at(torch_ranking_metrics_inputs, metric):
    metric = torch_metric.ranking_metrics_registry[metric]
    metric.top_ks = torch_ranking_metrics_inputs["ks"]
    result = metric(
        torch_ranking_metrics_inputs["scores"], torch_ranking_metrics_inputs["labels_one_hot"]
    )
    assert len(result) == len(torch_ranking_metrics_inputs["ks"])


# Test label one hot encoding
@pytest.mark.parametrize("metric", list_metrics)
def test_score_with_transform_onehot(torch_ranking_metrics_inputs, metric):
    metric = torch_metric.ranking_metrics_registry[metric]
    metric.top_ks = torch_ranking_metrics_inputs["ks"]
    metric.labels_onehot = True
    result = metric(torch_ranking_metrics_inputs["scores"], torch_ranking_metrics_inputs["labels"])
    assert len(result) == len(torch_ranking_metrics_inputs["ks"])


# TODO: Compare the metrics @K between pytorch and numpy
@pytest.mark.parametrize("metric", list_metrics)
def test_numpy_comparison(metric):
    pass
