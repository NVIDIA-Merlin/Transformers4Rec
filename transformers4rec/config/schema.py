from typing import Optional

try:
    from functools import cached_property
except ImportError:
    # polyfill cached_property for python <= 3.7 (using lru_cache which was introduced in python3.2)
    from functools import lru_cache

    cached_property = lambda func: property(lru_cache()(func))  # noqa

from transformers4rec.utils.dependencies import is_nvtabular_available

if is_nvtabular_available():
    from nvtabular.columns.schema import ColumnSchema, Schema
else:
    from ._schema import ColumnSchema, Schema

from . import schema_proto

schema_proto.Feature(
    value_count=schema_proto.ValueCount(0, 1), annotation=schema_proto.Annotation()
)


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

    @cached_property
    def item_id_column_name(self):
        item_id_col = self.schema.select_by_tag("item_id")
        if len(item_id_col.columns) == 0:
            raise ValueError("There is no column tagged as item id.")

        return item_id_col.column_names[0]

    def get_item_ids_from_inputs(self, inputs):
        return inputs[self.item_id_column_name]

    def get_mask_from_inputs(self, inputs, mask_token=0):
        return self.get_item_ids_from_inputs(inputs) != mask_token


def requires_schema(module):
    module.REQUIRES_SCHEMA = True

    return module


__all__ = ["ColumnSchema", "Schema", "SchemaMixin", "requires_schema"]
