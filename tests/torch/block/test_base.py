import pytest

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


def test_base_block(yoochoose_schema):
    tab_module = torch4rec.TabularFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, aggregation="concat"
    )

    block = tab_module >> torch4rec.MLPBlock([64, 32])

    embedding_block = block.get_children_by_class_name(list(block), "EmbeddingFeatures")[0]

    assert isinstance(embedding_block, torch4rec.EmbeddingFeatures)


def test_sequential_block(yoochoose_schema):
    tab_module = torch4rec.TabularFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, aggregation="concat"
    )

    block = torch4rec.SequentialBlock(tab_module, torch4rec.MLPBlock([64, 32]))

    embedding_block = block.get_children_by_class_name(list(block), "EmbeddingFeatures")[0]

    assert isinstance(embedding_block, torch4rec.EmbeddingFeatures)


def test_sequential(yoochoose_schema):
    tab_module = torch4rec.TabularFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, aggregation="concat"
    )

    block = pytorch.nn.Sequential(*torch4rec.build_blocks(tab_module, torch4rec.MLPBlock([64, 32])))
    block2 = pytorch.nn.Sequential(tab_module, torch4rec.MLPBlock([64, 32]).to_module(tab_module))

    assert isinstance(block, pytorch.nn.Sequential)
    assert isinstance(block2, pytorch.nn.Sequential)
