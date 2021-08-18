import pytest
import torch

from transformers4rec.config import transformer as tconf

torch4rec = pytest.importorskip("transformers4rec.torch")

config_classes = [
    tconf.XLNetConfig,
    tconf.LongformerConfig,
    tconf.GPT2Config,
]

# fixed parameters for tests
lm_tasks = list(torch4rec.masking.masking_registry.keys())
lm_tasks.remove("permutation")


# Test output of XLNet with different masking taks using SequentialBlock
@pytest.mark.parametrize("task", lm_tasks)
def test_transformer_block(yoochoose_schema, torch_yoochoose_like, task):

    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group,
        max_sequence_length=20,
        aggregation="sequential_concat",
        # d_output=64,
        masking=task,
    )

    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )

    block = (
        tab_module
        >> torch4rec.MLPBlock([64])
        >> torch4rec.TransformerBlock(transformer=transformer_config, masking=tab_module.masking)
    )

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64


# Test output of XLNet with permutation language model using SequentialBlock
def test_xlnet_with_plm(yoochoose_schema, torch_yoochoose_like):

    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group,
        max_sequence_length=20,
        aggregation="sequential_concat",
        d_output=64,
        masking="permutation",
    )

    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )

    block = (
        tab_module
        >> torch4rec.MLPBlock([64])
        >> torch4rec.TransformerBlock(transformer=transformer_config, masking=tab_module.masking)
    )

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64


# Test permutation language model with wrong transformer architecture
def test_plm_wrong_transformer(yoochoose_schema, torch_yoochoose_like):
    with pytest.raises(ValueError) as excinfo:
        col_group = yoochoose_schema
        tab_module = torch4rec.SequentialTabularFeatures.from_schema(
            col_group,
            max_sequence_length=20,
            aggregation="sequential_concat",
            d_output=64,
            masking="permutation",
        )

        transformer_config = tconf.AlbertConfig.build(
            d_model=64, n_head=4, n_layer=2, total_seq_length=20
        )

        block = (
            tab_module
            >> torch4rec.MLPBlock([64])
            >> torch4rec.TransformerBlock(
                transformer=transformer_config, masking=tab_module.masking
            )
        )

        block(torch_yoochoose_like)

    assert (
        "Permutation Language Modeling requires the parameters ['target_mapping', 'perm_mask']"
        in str(excinfo.value)
    )


# Test output of TransformerBlocks with CLM using pytorch-like code
@pytest.mark.parametrize("transformer_body", config_classes)
def test_transformer_block_clm(yoochoose_schema, torch_yoochoose_like, transformer_body):
    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group,
        max_sequence_length=20,
        aggregation="sequential_concat",
        d_output=64,
        masking="causal",
    )

    transformer_model = transformer_body.build(d_model=64, n_head=4, n_layer=2, total_seq_length=20)
    model = torch4rec.TransformerBlock(transformer=transformer_model)

    block = torch.nn.Sequential(tab_module, model)

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64


# Test output of Reformer with clm using pytorch-like code
def test_reformer_block_clm(yoochoose_schema, torch_yoochoose_like):
    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group,
        max_sequence_length=20,
        aggregation="sequential_concat",
        d_output=64,
        masking="causal",
    )

    model = torch4rec.TransformerBlock.from_registry(
        transformer="reformer", d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )

    block = torch.nn.Sequential(tab_module, model)

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    # 2 * hidden_size because Reformer uses reversible resnet layers
    assert outputs.shape[-1] == 64 * 2
