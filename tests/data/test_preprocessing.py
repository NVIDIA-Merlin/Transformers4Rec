import numpy as np
import pytest

from merlin_standard_lib import ColumnSchema, Schema, Tag
from transformers4rec.data.preprocessing import (
    add_item_first_seen_col_to_df,
    remove_consecutive_interactions,
)
from transformers4rec.data.synthetic import (
    generate_item_interactions,
    synthetic_ecommerce_data_schema,
)

pd = pytest.importorskip("pandas")


def test_remove_consecutive_interactions():
    np.random.seed(0)

    schema = synthetic_ecommerce_data_schema.remove_by_name("item_recency")
    schema += Schema([ColumnSchema.create_continuous("timestamp", tags=[Tag.SESSION])])

    interactions_df = generate_item_interactions(500, schema)
    filtered_df = remove_consecutive_interactions(interactions_df.copy())

    assert len(filtered_df) < len(interactions_df)
    assert len(filtered_df) == 499
    assert len(list(filtered_df.columns)) == len(list(interactions_df.columns))


def test_add_item_first_seen_col_to_df():
    schema = synthetic_ecommerce_data_schema.remove_by_name("item_recency")
    schema += Schema([ColumnSchema.create_continuous("timestamp", tags=[Tag.SESSION])])

    df = add_item_first_seen_col_to_df(generate_item_interactions(500, schema))

    assert len(list(df.columns)) == len(schema) + 1
    assert isinstance(df["item_ts_first"], pd.Series)


# TODO: Add test for session_aggregator when nvtabular 21.09 is released
