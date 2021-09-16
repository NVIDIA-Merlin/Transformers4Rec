from transformers4rec.data.dataset import ParquetDataset
from transformers4rec.data.testing.tabular_data.dataset import tabular_testing_data


def test_tabular_testing_data():
    assert isinstance(tabular_testing_data, ParquetDataset)
    assert tabular_testing_data.path.endswith(
        "transformers4rec/data/testing/tabular_data/data.parquet"
    )
    assert tabular_testing_data.schema_path.endswith(
        "transformers4rec/data/testing/tabular_data/schema.json"
    )
    assert len(tabular_testing_data.schema) == 11
