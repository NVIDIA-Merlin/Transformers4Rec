import pytest

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


def test_simple_model(torch_yoochoose_tabular_features, torch_yoochoose_like):
    targets = {"target": pytorch.randint(2, (100,)).float()}

    inputs = torch_yoochoose_tabular_features
    body = torch4rec.SequentialBlock(inputs, torch4rec.MLPBlock([64]))
    model = torch4rec.BinaryClassificationTask("target").to_model(body, inputs)

    dataset = [(torch_yoochoose_like, targets)]
    losses = model.fit(dataset, num_epochs=5)

    assert len(losses) == 5
    assert all(loss.min() >= 0 and loss.max() <= 1 for loss in losses)
