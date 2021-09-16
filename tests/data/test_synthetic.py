import merlin_standard_lib as msl
from merlin_standard_lib import Tag
from transformers4rec.data.synthetic import generate_session_interactions


def test_synthetic_data():
    schema = msl.Schema(
        [
            msl.ColumnSchema.create_categorical("session_id", num_items=5000, tags=["session_id"]),
            msl.ColumnSchema.create_categorical(
                "item_id",
                num_items=10000,
                tags=[Tag.ITEM_ID, Tag.LIST],
                value_count=msl.schema.ValueCount(1, 20),
            ),
            msl.ColumnSchema.create_categorical(
                "category",
                num_items=100,
                tags=[Tag.LIST, Tag.ITEM],
                value_count=msl.schema.ValueCount(1, 20),
            ),
            msl.ColumnSchema.create_continuous(
                "item_recency",
                min_value=0,
                max_value=1,
                tags=[Tag.LIST, Tag.ITEM],
                value_count=msl.schema.ValueCount(1, 20),
            ),
            msl.ColumnSchema.create_categorical("day", num_items=11, tags=[Tag.SESSION]),
            msl.ColumnSchema.create_categorical(
                "purchase", num_items=3, tags=[Tag.SESSION, Tag.BINARY_CLASSIFICATION]
            ),
            msl.ColumnSchema.create_continuous(
                "price", min_value=0, max_value=1, tags=[Tag.SESSION, Tag.REGRESSION]
            ),
        ]
    )

    data = generate_session_interactions(100, schema, 30, 5, "cpu")

    assert data is not None
