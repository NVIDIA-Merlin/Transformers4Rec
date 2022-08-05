import pytest

from transformers4rec.config import transformer as tconf

pytorch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers4rec.torch")


def test_torchsciprt_not_strict(torch_yoochoose_like, yoochoose_schema):

    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        d_output=64,
        masking="causal",
    )
    prediction_task = tr.NextItemPredictionTask(hf_format=True, weight_tying=True)
    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=8, n_layer=2, total_seq_length=20
    )
    model = transformer_config.to_torch_model(input_module, prediction_task)

    _ = model(torch_yoochoose_like, training=False)

    model.eval()

    traced_model = pytorch.jit.trace(model, torch_yoochoose_like, strict=False)
    assert isinstance(traced_model, pytorch.jit.TopLevelTracedModule)
    assert pytorch.allclose(
        model(torch_yoochoose_like)["predictions"],
        traced_model(torch_yoochoose_like)["predictions"],
    )
