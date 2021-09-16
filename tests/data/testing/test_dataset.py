from transformers4rec.data.dataset import ParquetDataset
from transformers4rec.data.testing.dataset import tabular_sequence_testing_data


def test_tabular_sequence_testing_data():
    assert isinstance(tabular_sequence_testing_data, ParquetDataset)
    assert tabular_sequence_testing_data.path.endswith("transformers4rec/data/testing/data.parquet")
    assert tabular_sequence_testing_data.schema_path.endswith(
        "transformers4rec/data/testing/schema.json"
    )
    assert len(tabular_sequence_testing_data.schema) == 22
