import pytest

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_mlp_block(yoochoose_schema, torch_yoochoose_like):
    tab_module = torch4rec.TabularFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, aggregation="concat"
    )

    block = tab_module >> torch4rec.MLPBlock([64, 32])

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 2
    assert outputs.shape[-1] == 32
