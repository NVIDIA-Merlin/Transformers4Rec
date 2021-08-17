import pytest

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_masking_block_torch_like_wrong_masking_type():
    with pytest.raises(AssertionError) as excinfo:
        torch4rec.block.masking.MaskingBlock(
            masking="clm",
            input_module=None,
        )
    assert "masking needs to be an instance of MaskSequence class" in str(excinfo.value)


def test_masking_block_output(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema

    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        schema, max_sequence_length=20, aggregation="sequential_concat"
    )

    masking_block = torch4rec.block.masking.MaskingBlock.from_registry(
        input_module=tab_module,
        masking="causal",
        hidden_size=64,
    )

    block = tab_module >> torch4rec.MLPBlock([64]) >> masking_block

    outputs = block(torch_yoochoose_like)

    assert outputs.ndim == 3
    assert outputs.shape[-1] == 64
