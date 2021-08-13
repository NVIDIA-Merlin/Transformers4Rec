import pytest
import torch

from transformers4rec.config import transformer as tconf

torch4rec = pytest.importorskip("transformers4rec.torch")

config_classes = [
    tconf.XLNetConfig,
    tconf.ElectraConfig,
    tconf.LongformerConfig,
    tconf.GPT2Config,
]

# fixed parameters for tests
lm_tasks = list(torch4rec.masking.masking_registry.keys())


# Test output of XLNet with different masking taks using SequentialBlock
@pytest.mark.parametrize("task", lm_tasks)
def test_transformer_block(yoochoose_schema, torch_yoochoose_like, task):
    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )

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
def test_xlnet_block_torch_like(yoochoose_schema, torch_yoochoose_like, task):
    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    xlnet_model = torch4rec.TransformerBlock(
        masking=task,
        body="xlnet",
        d_model=64,
        n_head=4,
        n_layer=2,
        total_seq_length=20,
    ).to_torch_module(input_module=tab_module)

    projection = torch.nn.Linear(tab_module.output_size()[-1], xlnet_model.body.config.hidden_size)
    block = torch.nn.Sequential(tab_module, projection, xlnet_model)

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64


# Test output of TransformerBlocks with CLM using pytorch-like code
@pytest.mark.parametrize("transformer_body", config_classes)
def test_transformer_block_clm(yoochoose_schema, torch_yoochoose_like, transformer_body):
    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    transformer_model = transformer_body.build(d_model=64, n_head=4, n_layer=2, total_seq_length=20)
    model = torch4rec.TransformerBlock(masking="causal", body=transformer_model).to_torch_module(
        input_module=tab_module
    )

    projection = torch.nn.Linear(tab_module.output_size()[-1], model.body.config.hidden_size)
    block = torch.nn.Sequential(tab_module, projection, model)

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64


# Test output of Reformer with clm using pytorch-like code
def test_reformer_block_clm(yoochoose_schema, torch_yoochoose_like):
    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    reformer_model = torch4rec.TransformerBlock(
        masking="causal",
        body="reformer",
        d_model=64,
        n_head=4,
        n_layer=2,
        total_seq_length=20,
    ).to_torch_module(input_module=tab_module)

    projection = torch.nn.Linear(
        tab_module.output_size()[-1], reformer_model.body.config.hidden_size
    )
    block = torch.nn.Sequential(tab_module, projection, reformer_model)

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    # 2 * hidden_size because Reformer uses reversible resnet layers
    assert outputs.shape[-1] == 64 * 2


@pytest.mark.parametrize("transformer_body", config_classes)
def test_transformer_block_torch_like_no_input(transformer_body):
    with pytest.raises(TypeError) as excinfo:
        transformer_model = transformer_body.build(
            d_model=64, n_head=4, n_layer=2, total_seq_length=20
        )
        model = torch4rec.TransformerBlock(masking="causal", body=transformer_model)
        projection = torch.nn.Linear(64, 128)
        model = torch.nn.Sequential(model, projection)

    assert "TransformerBlock is not a Module subclass" in str(excinfo.value)
