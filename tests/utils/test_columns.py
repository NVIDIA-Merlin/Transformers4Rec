from transformers4rec.utils.columns import ColumnGroup
from transformers4rec.utils.tags import Tag


def test_column_group_from_schema(schema_file):
    col_group = ColumnGroup.from_schema(str(schema_file))

    assert len(col_group.columns) == 18
    assert col_group.columns[1].tags == ["list"]


def test_column_group_from_yoochoose_schema(yoochoose_schema_file):
    col_group = ColumnGroup.from_schema(str(yoochoose_schema_file))

    assert len(col_group.columns) == 20
    assert len(col_group.select_by_tag(Tag.CONTINUOUS).columns) == 6
    assert len(col_group.select_by_tag(Tag.CATEGORICAL).columns) == 2
