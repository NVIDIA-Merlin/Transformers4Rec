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
from typing import Optional

from merlin_standard_lib import Schema


class SchemaMixin:
    REQUIRES_SCHEMA = False

    def set_schema(self, schema=None):
        self.check_schema(schema=schema)

        if schema and not getattr(self, "schema", None):
            self._schema = schema

        return self

    @property
    def schema(self) -> Optional[Schema]:
        return getattr(self, "_schema", None)

    @schema.setter
    def schema(self, value):
        if value:
            self.set_schema(value)
        else:
            self._schema = value

    def check_schema(self, schema=None):
        if self.REQUIRES_SCHEMA and not getattr(self, "schema", None) and not schema:
            raise ValueError(f"{self.__class__.__name__} requires a schema.")

    def __call__(self, *args, **kwargs):
        self.check_schema()

        return super().__call__(*args, **kwargs)

    def _maybe_set_schema(self, input, schema):
        if input and getattr(input, "set_schema"):
            input.set_schema(schema)

    def get_item_ids_from_inputs(self, inputs):
        return inputs[self.schema.item_id_column_name]

    def get_padding_mask_from_item_id(self, inputs, pad_token=0):
        item_id_inputs = self.get_item_ids_from_inputs(inputs)
        if len(item_id_inputs.shape) != 2:
            raise ValueError(
                "To extract the padding mask from item id tensor "
                "it is expected to have 2 dims, but it has {} dims.".format(item_id_inputs.shape)
            )
        return self.get_item_ids_from_inputs(inputs) != pad_token


def requires_schema(module):
    module.REQUIRES_SCHEMA = True

    return module
