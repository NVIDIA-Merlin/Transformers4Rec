import pytest

from transformers4rec.utils.schema import DatasetSchema
from transformers4rec.utils.tags import Tag


def test_schema_from_schema(schema_file):
    schema = DatasetSchema.from_schema(str(schema_file))

    assert len(schema.columns) == 18
    assert schema.columns[1].tags == ["list"]


def test_schema_from_yoochoose_schema(yoochoose_schema_file):
    schema = DatasetSchema.from_schema(str(yoochoose_schema_file))

    assert len(schema.columns) == 20
    assert len(schema.select_by_tag(Tag.CONTINUOUS).columns) == 6
    assert len(schema.select_by_tag(Tag.CATEGORICAL).columns) == 2


@pytest.mark.skip(reason="This test requires NVTabular installed but it is not in the CI instance")
def test_schema_embedding_sizes_nvt(yoochoose_schema_file):
    schema = DatasetSchema.from_schema(str(yoochoose_schema_file))

    assert schema.cardinalities() == {"item_id/list": 51996, "category/list": 332}
    embedding_sizes = schema.embedding_sizes_nvt(minimum_size=16, maximum_size=512)
    assert embedding_sizes == {"item_id/list": 512, "category/list": 41}


def test_schema_embedding_sizes(yoochoose_schema_file):
    schema = DatasetSchema.from_schema(str(yoochoose_schema_file))

    assert schema.cardinalities() == {"item_id/list": 51996, "category/list": 332}
    embedding_sizes = schema.embedding_sizes(multiplier=3.0)
    assert embedding_sizes == {"item_id/list": 46, "category/list": 13}
