from typing import Any, Optional, Tuple, Type, Union

try:
    from functools import cached_property
except ImportError:
    # polyfill cached_property for python <= 3.7 (using lru_cache which was introduced in python3.2)
    from functools import lru_cache

    cached_property = lambda func: property(lru_cache()(func))  # noqa

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

from .schema_proto import *  # noqa


def _parse_shape_and_value_count(shape, value_count) -> Dict[str, Any]:
    output = {}
    if shape:
        output["shape"] = FixedShape([FixedShapeDim(d) for d in shape])

    if value_count:
        if isinstance(value_count, ValueCount):
            output["value_count"] = value_count
        elif isinstance(value_count, ValueCountList):
            output["value_counts"] = value_count
        else:
            raise ValueError("Unknown value_count type.")

    return output


class ColumnSchema(Feature):
    @classmethod
    def create_categorical(
        cls,
        name: str,
        num_items: int,
        shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
        value_count: Optional[Union[ValueCount, ValueCountList]] = None,
        min_index: int = 0,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> "ColumnSchema":
        extra = _parse_shape_and_value_count(shape, value_count)
        int_domain = IntDomain(name=name, min=min_index, max=num_items, is_categorical=True)
        tags = list(set(tags or [] + ["categorical"]))

        return cls(name=name, int_domain=int_domain, **extra, **kwargs).with_tags(tags)

    @classmethod
    def create_continuous(
        cls,
        name: str,
        is_float: bool = True,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        disallow_nan: bool = False,
        disallow_inf: bool = False,
        is_embedding: bool = False,
        shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
        value_count: Optional[Union[ValueCount, ValueCountList]] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> "ColumnSchema":
        extra = _parse_shape_and_value_count(shape, value_count)
        if min_value is not None and max_value is not None:
            if is_float:
                extra["float_domain"] = FloatDomain(
                    name=name,
                    min=float(min_value),
                    max=float(max_value),
                    disallow_nan=disallow_nan,
                    disallow_inf=disallow_inf,
                    is_embedding=is_embedding,
                )
            else:
                extra["int_domain"] = IntDomain(
                    name=name, min=int(min_value), max=int(max_value), is_categorical=False
                )
        tags = list(set(tags or [] + ["continuous"]))

        return cls(name=name, **extra, **kwargs).with_tags(tags)

    def copy(self, **kwargs) -> "ColumnSchema":
        output = self.parse(bytes(self))
        for key, val in kwargs.items():
            setattr(output, key, val)

        return output

    def with_name(self, name: str):
        return self.copy(name=name)

    def with_tags(self, tags: List[str]) -> "ColumnSchema":
        output = self.copy()
        if self.annotation:
            self.annotation.tag = list(set(list(self.annotation.tag) + tags))
        else:
            self.annotation = Annotation(tag=tags)

        return output

    def with_properties(self, properties: Dict[str, Union[str, int, float]]) -> "ColumnSchema":
        output = self.copy()
        if self.annotation:
            if len(self.annotation.extra_metadata) > 0:
                self.annotation.extra_metadata[0].update(properties)
            else:
                self.annotation.extra_metadata = properties
        else:
            self.annotation = Annotation(extra_metadata=[properties])

        return output

    @property
    def tags(self):
        return self.annotation.tag

    @property
    def properties(self):
        return self.annotation.extra_metadata

    def __str__(self) -> str:
        return self.name

    def to_proto_text(self):
        serialized = bytes(self)

        feature = schema_pb2.Feature()
        feature.ParseFromString(serialized)

        return text_format.MessageToString(feature)


class Schema:
    """A collection of column schemas for a dataset."""

    def __init__(
        self, column_schemas: Optional[Union[List[ColumnSchema], Dict[str, ColumnSchema]]] = None
    ):
        column_schemas = column_schemas or {}

        self.column_schemas: Dict[str, ColumnSchema] = {}

        if isinstance(column_schemas, dict):
            self.column_schemas = column_schemas
        elif isinstance(column_schemas, list):
            self.column_schemas = {}
            for column_schema in column_schemas:
                if isinstance(column_schema, str):
                    column_schema = ColumnSchema(column_schema)
                self.column_schemas[column_schema.name] = column_schema
        else:
            raise TypeError("The `column_schemas` parameter must be a list or dict.")

    @property
    def column_names(self):
        return list(self.column_schemas.keys())

    def apply(self, selector):
        if selector and selector.names:
            return self.select_by_name(selector.names)
        else:
            return self

    def apply_inverse(self, selector):
        if selector:
            return self - self.select_by_name(selector.names)
        else:
            return self

    def select_by_tag(self, tags):
        if not isinstance(tags, list):
            tags = [tags]

        selected_schemas = {}

        for _, column_schema in self.column_schemas.items():
            if all(x in column_schema.tags for x in tags):
                selected_schemas[column_schema.name] = column_schema

        return Schema(selected_schemas)

    def select_by_name(self, names):
        if isinstance(names, str):
            names = [names]

        selected_schemas = {key: self.column_schemas[key] for key in names}
        return Schema(selected_schemas)

    def categorical_cardinalities(self) -> Dict[str, int]:
        outputs = {}
        for col in self:
            if col.int_domain and col.int_domain.is_categorical:
                outputs[col.name] = col.int_domain.max + 1

        return outputs

    def to_proto_text(self):
        schema = schema_pb2.Schema()
        features = []
        for col in self:
            feature = schema_pb2.Feature()
            feature.ParseFromString(bytes(col))
            features.append(feature)
        schema.feature.extend(features)

        return text_format.MessageToString(schema)

    def __iter__(self):
        return iter(self.column_schemas.values())

    def __len__(self):
        return len(self.column_schemas)

    def __repr__(self):
        return str(
            [
                col_schema.to_dict(casing=betterproto.Casing.SNAKE)
                for col_schema in self.column_schemas.values()
            ]
        )

    def __eq__(self, other):
        if not isinstance(other, Schema) or len(self.column_schemas) != len(other.column_schemas):
            return False
        return self.column_schemas == other.column_schemas

    def __add__(self, other):
        if other is None:
            return self
        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for +: 'Schema' and {type(other)}")
        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for +: 'Schema' and {type(other)}")

        return Schema({**self.column_schemas, **other.column_schemas})

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if other is None:
            return self

        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for -: 'Schema' and {type(other)}")

        result = Schema({**self.column_schemas})

        for key in other.column_schemas.keys():
            if key in self.column_schemas.keys():
                result.column_schemas.pop(key, None)

        return result

    @cached_property
    def item_id_column_name(self):
        item_id_col = self.select_by_tag("item_id")
        if len(item_id_col.column_names) == 0:
            raise ValueError("There is no column tagged as item id.")

        return item_id_col.column_names[0]
