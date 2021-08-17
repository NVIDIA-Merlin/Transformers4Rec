import pytest

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
lm_tasks.remove("permutation")


# Test output of XLNet with different masking taks using SequentialBlock
@pytest.mark.parametrize("task", lm_tasks)
def test_transformer_block(yoochoose_schema, torch_yoochoose_like, task):

    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    masking_block = torch4rec.block.masking.MaskingBlock.from_registry(
        input_module=tab_module,
        masking="causal",
        hidden_size=64,
    )

    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )

    block = (
        tab_module
        >> torch4rec.MLPBlock([64])
        >> masking_block
        >> torch4rec.TransformerBlock(transformer=transformer_config)
    )

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64


# Test output of XLNet with permutation language model using SequentialBlock
def test_xlnet_with_plm(yoochoose_schema, torch_yoochoose_like):

    col_group = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    masking_block = torch4rec.block.masking.MaskingBlock(
        input_module=tab_module,
        masking=torch4rec.masking.PermutationLanguageModeling(hidden_size=64),
    )

    transformer_config = tconf.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )

    block = (
        tab_module
        >> torch4rec.MLPBlock([64])
        >> masking_block
        >> torch4rec.TransformerBlock(transformer=transformer_config, masking=masking_block)
    )

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64
