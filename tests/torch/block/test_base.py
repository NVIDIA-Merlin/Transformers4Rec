import pytest

torch4rec = pytest.importorskip("transformers4rec.torch")


def test_base_block(yoochoose_column_group):
    tab_module = torch4rec.TabularFeatures.from_column_group(
        yoochoose_column_group, max_sequence_length=20, aggregation="concat"
    )

    block = tab_module >> torch4rec.MLPBlock([64, 32])

    embedding_block = block.get_children_by_class_name("EmbeddingFeatures")[0]

    assert isinstance(embedding_block, torch4rec.EmbeddingFeatures)
