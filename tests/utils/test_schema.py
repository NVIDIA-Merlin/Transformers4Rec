from transformers4rec.utils.schema import Schema
from transformers4rec.utils.tags import Tag


def test_schema_from_schema(schema_file):
    schema = Schema.from_schema(str(schema_file))

    assert len(schema.columns) == 18
    assert schema.columns[1].tags == ["list"]


def test_schema_from_yoochoose_schema(yoochoose_schema_file):
    schema = Schema.from_schema(str(yoochoose_schema_file))

    assert len(schema.columns) == 20
    assert len(schema.select_by_tag(Tag.CONTINUOUS).columns) == 6
    assert len(schema.select_by_tag(Tag.CATEGORICAL).columns) == 2
