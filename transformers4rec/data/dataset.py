#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import os
from typing import Optional

from merlin_standard_lib import Schema


class Dataset:
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        if self.schema_path.endswith(".pb") or self.schema_path.endswith(".pbtxt"):
            self._schema = Schema().from_proto_text(self.schema_path)
        else:
            self._schema = Schema().from_json(self.schema_path)

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
        schema_file_name="schema.json",
        schema_path: Optional[str] = None,
    ):
        super(ParquetDataset, self).__init__(schema_path or os.path.join(dir, schema_file_name))
        self.path = os.path.join(dir, parquet_file_name)
