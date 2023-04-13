import torch

import transformers4rec.torch as tr
from transformers4rec.config import transformer as tconf


def test_torchsciprt_not_strict(torch_yoochoose_like, yoochoose_schema):
    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        d_output=64,
        masking="causal",
    )
    prediction_task = tr.NextItemPredictionTask(weight_tying=True)
    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=8, n_layer=2, total_seq_length=20
    )
    model = transformer_config.to_torch_model(input_module, prediction_task)

    _ = model(torch_yoochoose_like, training=False)

    model.eval()

    traced_model = torch.jit.trace(model, torch_yoochoose_like, strict=False)
    assert isinstance(traced_model, torch.jit.TopLevelTracedModule)
    assert torch.allclose(
        model(torch_yoochoose_like), traced_model(torch_yoochoose_like), rtol=1e-02
    )
