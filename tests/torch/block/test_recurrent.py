import pytest

from transformers4rec.config.transformer import XLNetConfig

torch4rec = pytest.importorskip("transformers4rec.torch")

# fixed parameters for tests
lm_tasks = list(torch4rec.masking.masking_registry.keys())


# Test output of XLNet with different masking
@pytest.mark.parametrize("task", lm_tasks)
def test_recurrent_block(yoochoose_column_group, torch_yoochoose_like, task):
    col_group = yoochoose_column_group
    tab_module = torch4rec.SequentialTabularFeatures.from_column_group(
        col_group, max_sequence_length=20, aggregation="sequential_concat"
    )

    transformer_config = XLNetConfig.for_rec(64, 4, 2)

    block = (
        tab_module
        >> torch4rec.MLPBlock([64])
        >> torch4rec.RecurrentBlock(hidden_size=64, masking=task, body=transformer_config)
    )

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64
