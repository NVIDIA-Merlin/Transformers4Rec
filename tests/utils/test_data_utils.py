import numpy as np

from transformers4rec.utils import data_utils

np.random.seed(0)


def test_remove_consecutive_interactions(testing_data):
    df = testing_data.compute()
    df["session_id"] = np.random.randint(0, 3, len(df))

    filtered = data_utils.remove_consecutive_interactions(
        df.copy(), timestamp_col="event_timestamp"
    )

    assert len(filtered) <= len(df)
    assert len(list(filtered.columns)) == len(list(df.columns))


def test_add_item_first_seen_col_to_df(testing_data):
    df = data_utils.add_item_first_seen_col_to_df(
        testing_data.compute(), timestamp_column="event_timestamp"
    )

    assert len(list(df.columns)) == len(testing_data.schema) + 1
    assert len(df["item_ts_first"]) == 100
    assert str(df["item_ts_first"].dtype) == "int64"


# TODO: Add test for session_aggregator when nvtabular 21.09 is released
