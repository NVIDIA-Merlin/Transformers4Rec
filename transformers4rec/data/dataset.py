import os
from typing import Optional

from merlin_standard_lib import Schema


class Dataset:
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        if self.schema_path.endswith(".json"):
            self._schema = Schema().from_json(self.schema_path)
        else:
            self._schema = Schema().from_proto_text(self.schema_path)

    @property
    def schema(self) -> Schema:
        return self._schema

    def torch_synthetic_data(self, num_rows=100, min_session_length=5, max_session_length=20):
        from transformers4rec.torch.utils import schema_utils

        return schema_utils.random_data_from_schema(
            self.schema,
            num_rows=num_rows,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
        )

    def tf_synthetic_data(self, num_rows=100, min_session_length=5, max_session_length=20):
        from transformers4rec.tf.utils import schema_utils

        return schema_utils.random_data_from_schema(
            self.schema,
            num_rows=num_rows,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
        )


class ParquetDataset(Dataset):
    def __init__(
        self,
        dir,
        parquet_file_name="data.parquet",
        schema_file_name="schema.pbtxt",
        schema_path: Optional[str] = None,
    ):
        super(ParquetDataset, self).__init__(schema_path or os.path.join(dir, schema_file_name))
        self.path = os.path.join(dir, parquet_file_name)
