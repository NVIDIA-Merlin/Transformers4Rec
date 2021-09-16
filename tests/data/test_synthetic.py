import pytest

from transformers4rec.data.synthetic import (
    generate_item_interactions,
    synthetic_ecommerce_data_schema,
)

pd = pytest.importorskip("pandas")


def test_generate_item_interactions():
    data = generate_item_interactions(500, synthetic_ecommerce_data_schema)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 500
    assert list(data.columns) == [
        "session_id",
        "item_id",
        "day",
        "purchase",
        "price",
        "category",
        "item_recency",
    ]
    expected_dtypes = {
        "session_id": "int64",
        "item_id": "int64",
        "day": "int64",
        "purchase": "int64",
        "price": "float64",
        "category": "int64",
        "item_recency": "float64",
    }

    assert all(val == expected_dtypes[key] for key, val in dict(data.dtypes).items())
