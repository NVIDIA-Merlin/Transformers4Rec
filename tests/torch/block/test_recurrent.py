import pytest

from transformers4rec.config.transformer import XLNetConfig

torch4rec = pytest.importorskip("transformers4rec.torch")


# TODO: Finish this test
def test_recurrent_block(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        schema, max_sequence_length=20, aggregation="sequential_concat"
    )

    transformer_config = XLNetConfig.for_rec(64, 4, 2)

    block = (
        tab_module
        >> torch4rec.MLPBlock([64])
        >> torch4rec.RecurrentBlock("causal", transformer_config)
    )

    assert block
