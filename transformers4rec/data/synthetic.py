import logging

import numpy as np
import pandas as pd

import merlin_standard_lib as msl
from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.utils.proto_utils import has_field

LOG = logging.getLogger("transformers4rec")


def generate_session_interactions(
    num_interactions: int,
    schema: Schema,
    maximum_length: int = 20,
    minmum_length: int = 2,
    device: str = "gpu",
):
    """
    Util function to generate random synthetic dataset for session-based recommendation
    from a schema object. It supports the generation of session and item features.
    The item interactions are generated using `generate_item_interactions`.
    The sessions are aggregated using `session_aggregator`.

    Parameters:
    ----------
    num_interactions: int
        number of interaction rows to generate.
    schema: Schema
        The schema objects describing the features to generate.
    maximum_length: int
        Trim all sessions to a maximum length.
    minimum_length: int
        Filter out sessions shorter than minimum_length.
    device: str
        Aggregate data using `cpu` or gpu `NVTabular` workflow

    Example usage::
    >>> import merlin_standard_lib as msl
    >>> s = msl.Schema(
        [
            msl.ColumnSchema.create_categorical("session_id", num_items=5000, tags=['session_id']),
            msl.ColumnSchema.create_categorical(
                "item_id",
                num_items=10000,
                tags=[Tag.ITEM_ID, Tag.LIST],
                value_count=msl.schema.ValueCount(1, 20),
            ),
            msl.ColumnSchema.create_categorical(
                "category", num_items=100,
                tags=[Tag.LIST, Tag.ITEM], value_count=msl.schema.ValueCount(1, 20)
            ),
            msl.ColumnSchema.create_continuous(
                "item_recency", min_value=0, max_value=1,
                tags=[Tag.LIST, Tag.ITEM], value_count=msl.schema.ValueCount(1, 20)
            ),
            msl.ColumnSchema.create_categorical("day", num_items=11, tags=['session']),
            msl.ColumnSchema.create_categorical(
                "purchase", num_items=3, tags=['session', Tag.BINARY_CLASSIFICATION]),
            msl.ColumnSchema.create_continuous(
                "price", min_value=0, max_value=1 , tags=['session', Tag.REGRESSION])
        ]
    )
    >>> generate_session_interactions(100000, s, 30, 5, 'gpu')
    """
    from transformers4rec.data.preprocessing import session_aggregator  # type: ignore

    data = generate_item_interactions(num_interactions, schema)
    data = session_aggregator(
        schema, data, maximum_length=maximum_length, minimum_length=minmum_length, device=device
    )
    LOG.info(f"Data generated with {data.shape[0]} sessions")

    return data


def generate_item_interactions(num_interactions: int, schema: Schema) -> pd.DataFrame:
    """
    Util function to generate synthetic data for session-based item-interactions
    from a schema object. It supports the generation of session and item features.
    The schema should include a few tags:

    - `Tag.SESSION` for features related to sessions
    - `Tag.SESSION_ID` to tag the session-id feature
    - `Tag.ITEM` for features related to item interactions.

    Parameters:
    ----------
    num_interactions: int
        number of interaction rows to generate.
    schema: Schema
        schema object describing the columns to generate.

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe with synthetic generated data.
    """
    session_col = schema.select_by_tag(Tag.SESSION_ID).feature[0]
    data = pd.DataFrame(
        np.random.randint(1, session_col.int_domain.max, num_interactions),
        columns=[session_col.name],
    ).astype(np.int64)

    item_id_col = schema.select_by_tag(Tag.ITEM_ID).feature[0]
    data[item_id_col.name] = np.clip(
        np.random.lognormal(3.0, 1.0, num_interactions).astype(np.int32),
        1,
        item_id_col.int_domain.max,
    ).astype(np.int64)

    # get session cols
    session_features = schema.select_by_tag(Tag.SESSION).feature
    for feature in session_features:
        is_int_feature = has_field(feature, "int_domain")
        if is_int_feature:
            if Tag.BINARY_CLASSIFICATION.value in feature.tags:
                mapping_feature = dict(
                    zip(
                        data[session_col.name].unique(),
                        np.random.choice(a=[0, 1], size=(data[session_col.name].nunique())),
                    )
                )
                data[feature.name] = data[session_col.name].map(mapping_feature)
            else:
                mapping_feature = dict(
                    zip(
                        data[session_col.name].unique(),
                        np.random.randint(
                            1, feature.int_domain.max, size=(data[session_col.name].nunique())
                        ),
                    )
                )
                data[feature.name] = data[session_col.name].map(mapping_feature)

        else:
            mapping_feature = dict(
                zip(
                    data[session_col.name].unique(),
                    np.random.uniform(
                        feature.float_domain.min,
                        feature.float_domain.max,
                        size=(data[session_col.name].nunique()),
                    ),
                )
            )
            data[feature.name] = data[session_col.name].map(mapping_feature)

    # get item-id cols
    items_features = schema.select_by_tag(Tag.ITEM).feature
    for feature in items_features:
        is_int_feature = has_field(feature, "int_domain")
        if is_int_feature:
            data[feature.name] = pd.cut(
                data[item_id_col.name],
                bins=feature.int_domain.max - 1,
                labels=np.arange(1, feature.int_domain.max),
            ).astype(np.int64)
        else:
            data[feature.name] = np.random.uniform(
                feature.float_domain.min, feature.float_domain.max, num_interactions
            )

    return data


synthetic_ecommerce_data_schema = Schema(
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
