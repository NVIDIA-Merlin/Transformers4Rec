import pytest
import torch

from transformers4rec.config.transformer import XLNetConfig

torch4rec = pytest.importorskip("transformers4rec.torch")

# fixed parameters for tests
lm_tasks = list(torch4rec.masking.masking_registry.keys())


# Test output of XLNet with different masking taks using SequentialBlock
@pytest.mark.parametrize("task", lm_tasks)
def test_transformer_block(yoochoose_schema, torch_yoochoose_like, task):
    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    transformer_config = XLNetConfig.build(d_model=64, n_head=4, n_layer=2)

    block = (
        tab_module
        >> torch4rec.MLPBlock([64])
        >> torch4rec.TransformerBlock(masking=task, body=transformer_config)
    )

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64


# Test output of XLNet with different masking taks using pytorch-like code
@pytest.mark.parametrize("task", lm_tasks)
def test_transformer_block_torch_like(yoochoose_schema, torch_yoochoose_like, task):
    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    xlnet_model = torch4rec.TransformerBlock(
        masking=task, body="xlnet", d_model=64, n_head=4, n_layer=2
    ).link_to_input(input_module=tab_module)
    projection = torch.nn.Linear(tab_module.output_size()[-1], xlnet_model.body.d_model)
    block = torch.nn.Sequential(tab_module, projection, xlnet_model)

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64
