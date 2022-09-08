import numpy as np

from transformers4rec.data.dataset import ParquetDataset
from transformers4rec.data.testing.dataset import tabular_sequence_testing_data


def test_tabular_sequence_testing_data():
    assert isinstance(tabular_sequence_testing_data, ParquetDataset)
    assert tabular_sequence_testing_data.path.endswith("transformers4rec/data/testing/data.parquet")
    assert tabular_sequence_testing_data.schema_path.endswith(
        "transformers4rec/data/testing/schema.json"
    )
    assert len(tabular_sequence_testing_data.schema) == 22

    torch_yoochoose_like = tabular_sequence_testing_data.torch_synthetic_data(
        num_rows=100, min_session_length=5, max_session_length=20
    )

    t4r_yoochoose_schema = tabular_sequence_testing_data.schema

    non_matching_dtypes = {}
    for column in t4r_yoochoose_schema:
        name = column.name
        column_dtype = column.type
        schema_dtype = {0: np.float32, 2: np.int64, 3: np.float32}[column_dtype]

        value = torch_yoochoose_like[name]
        value_dtype = value.numpy().dtype

        if schema_dtype != value_dtype:
            non_matching_dtypes[name] = (column_dtype, schema_dtype, value_dtype)

    assert (
        len(non_matching_dtypes) == 0
    ), f"Found columns whose dtype does not match schema: {non_matching_dtypes}"
