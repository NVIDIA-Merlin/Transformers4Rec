import abc
from typing import Optional

import tensorflow as tf

from ...utils.schema import DatasetSchema


class SchemaMixin(abc.ABC):
    REQUIRES_SCHEMA = False

    def build(self, input_size, schema=None, **kwargs):
        self.check_schema(schema=schema)

        self.input_size = input_size
        if schema and not getattr(self, "schema", None):
            self.schema = schema

        return self

    @property
    def schema(self) -> Optional[DatasetSchema]:
        return getattr(self, "_schema", None)

    @schema.setter
    def schema(self, value):
        self._schema = value

    def check_schema(self, schema=None):
        if self.REQUIRES_SCHEMA and not getattr(self, "schema", None) and not schema:
            raise ValueError(f"{self.__class__.__name__} requires a schema.")

    def __call__(self, *args, **kwargs):
        self.check_schema()

        return super().__call__(*args, **kwargs)


def requires_schema(module):
    module.REQUIRES_SCHEMA = True

    return module


def get_output_sizes_from_schema(schema, batch_size=0, max_sequence_length=None):
    sizes = {}
    for feature in schema.feature:
        name = feature.name
        if feature.HasField("value_count"):
            sizes[name] = tf.TensorShape(
                [
                    batch_size,
                    max_sequence_length if max_sequence_length else feature.value_count.max,
                ]
            )
        elif feature.HasField("shape"):
            sizes[name] = tf.TensorShape([batch_size] + [d.size for d in feature.shape.dim])
        else:
            sizes[name] = tf.TensorShape([batch_size, 1])

    return sizes


def calculate_batch_size_from_input_shapes(input_shapes):
    return [i for i in input_shapes.values() if not isinstance(i, tuple)][0][0]
