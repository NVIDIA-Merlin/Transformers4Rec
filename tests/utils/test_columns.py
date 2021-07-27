from transformers4rec.utils.columns import ColumnGroup


def test_column_group_from_schema(schema_file):
    col_group = ColumnGroup.from_schema(str(schema_file))

    assert len(col_group.columns) == 18
    assert col_group.columns[1].tags == ["list"]
