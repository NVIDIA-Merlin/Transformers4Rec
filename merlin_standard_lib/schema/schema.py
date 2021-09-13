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


import collections
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utils import proto_utils

try:
    from functools import cached_property
except ImportError:
    # polyfill cached_property for python <= 3.7 (using lru_cache which was introduced in python3.2)
    from functools import lru_cache

    cached_property = lambda func: property(lru_cache()(func))  # noqa

import betterproto  # noqa

from ..proto.schema_bp import *  # noqa
from ..proto.schema_bp import (
    Annotation,
    Feature,
    FixedShape,
    FixedShapeDim,
    FloatDomain,
    IntDomain,
    ValueCount,
    ValueCountList,
    _Schema,
)


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
        return proto_utils.copy_better_proto_message(self, **kwargs)

    def with_name(self, name: str):
        return self.copy(name=name)

    def with_tags(self, tags: List[str]) -> "ColumnSchema":
        output = self.copy()
        if self.annotation:
            output.annotation.tag = list(set(list(self.annotation.tag) + tags))
        else:
            output.annotation = Annotation(tag=tags)

        return output

    def with_tags_based_on_properties(
        self, using_value_count=True, using_domain=True
    ) -> "ColumnSchema":
        from .tag import Tag

        extra_tags = []

        if using_value_count and proto_utils.has_field(self, "value_count"):
            extra_tags.append(str(Tag.LIST))

        if using_domain and proto_utils.has_field(self, "int_domain"):
            if self.int_domain.is_categorical:
                extra_tags.append(str(Tag.CATEGORICAL))
            else:
                extra_tags.append(str(Tag.CONTINUOUS))

        if using_domain and proto_utils.has_field(self, "float_domain"):
            extra_tags.append(str(Tag.CONTINUOUS))

        return self.with_tags(extra_tags) if extra_tags else self.copy()

    def _set_tags(self, tags: List[str]):
        if self.annotation:
            self.annotation.tag = list(set(list(self.annotation.tag) + tags))
        else:
            self.annotation = Annotation(tag=tags)

    def with_properties(self, properties: Dict[str, Union[str, int, float]]) -> "ColumnSchema":
        output = self.copy()
        if output.annotation:
            if len(output.annotation.extra_metadata) > 0:
                output.annotation.extra_metadata[0].update(properties)
            else:
                output.annotation.extra_metadata = [properties]
        else:
            output.annotation = Annotation(extra_metadata=[properties])

        return output

    @property
    def tags(self):
        return self.annotation.tag

    @property
    def properties(self) -> Dict[str, Union[str, float, int]]:
        if self.annotation.extra_metadata:
            return self.annotation.extra_metadata[0]

        return {}

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: "ColumnSchema") -> bool:
        return self.to_dict() == other.to_dict()

    def to_proto_text(self):
        from tensorflow_metadata.proto.v0 import schema_pb2

        return proto_utils.better_proto_to_proto_text(self, schema_pb2.Feature())


ColumnSchemaOrStr = Union[ColumnSchema, str]


class Schema(_Schema):
    """A collection of column schemas for a dataset."""

    feature: List["ColumnSchema"] = betterproto.message_field(1)

    @classmethod
    def create(
        cls,
        column_schemas: Optional[
            Union[List[ColumnSchemaOrStr], Dict[str, ColumnSchemaOrStr]]
        ] = None,
        **kwargs,
    ):
        column_schemas = column_schemas or []

        if isinstance(column_schemas, dict):
            column_schemas = list(column_schemas.values())

        features = []
        if isinstance(column_schemas, list):
            for column_schema in column_schemas:
                if isinstance(column_schema, str):
                    features.append(ColumnSchema(column_schema))
                else:
                    features.append(column_schema)
        else:
            raise TypeError("The `column_schemas` parameter must be a list or dict.")

        return cls(features, **kwargs)

    @property
    def column_names(self) -> List[str]:
        return [f.name for f in self.feature]

    @property
    def column_schemas(self) -> List[ColumnSchema]:
        return self.feature

    def apply(self, selector) -> "Schema":
        if selector and selector.names:
            return self.select_by_name(selector.names)
        else:
            return self

    def apply_inverse(self, selector) -> "Schema":
        if selector:
            return self - self.select_by_name(selector.names)
        else:
            return self

    def select_by_tag(self, tags) -> "Schema":
        from .. import Tag

        if isinstance(tags, (str, Tag)):
            tags = [tags]

        if isinstance(tags, (list, tuple)):

            def check_fn(column_tags):
                return all(x in column_tags for x in tags)

        elif callable(tags):
            check_fn = tags
        else:
            raise ValueError("Wrong type for tags.")

        return self._filter_by_tag_fn(check_fn)

    def remove_by_tag(self, tags) -> "Schema":
        from .. import Tag

        if isinstance(tags, (str, Tag)):
            tags = [tags]

        if isinstance(tags, (list, tuple)):

            def check_fn(column_tags):
                return not all(x in column_tags for x in tags)

        elif callable(tags):
            check_fn = tags
        else:
            raise ValueError("Wrong type for tags.")

        return self._filter_by_tag_fn(check_fn)

    def _filter_by_tag_fn(self, check_tag_fn: Callable[[List[str]], bool]):
        selected_schemas = []
        for column_schema in self.column_schemas:
            if check_tag_fn(column_schema.tags):
                selected_schemas.append(column_schema)

        return Schema(selected_schemas)

    def select_by_name(self, names) -> "Schema":
        if isinstance(names, str):
            names = [names]

        selected_schemas = [self.column_schemas[key] for key in names]
        return Schema(selected_schemas)

    def remove_by_name(self, names) -> "Schema":
        if isinstance(names, str):
            names = [names]

        selected_schemas = []

        for column_schema in self.column_schemas:
            if column_schema.name not in names:
                selected_schemas.append(column_schema)

        return Schema(selected_schemas)

    def categorical_cardinalities(self) -> Dict[str, int]:
        outputs = {}
        for col in self:
            if col.int_domain and col.int_domain.is_categorical:
                outputs[col.name] = col.int_domain.max + 1

        return outputs

    def __iter__(self):
        return iter(self.column_schemas)

    def __len__(self):
        return len(self.column_schemas)

    def __repr__(self):
        return str(
            [
                col_schema.to_dict(casing=betterproto.Casing.SNAKE)
                for col_schema in self.column_schemas
            ]
        )

    def __eq__(self, other):
        if not isinstance(other, Schema) or len(self.column_schemas) != len(other.column_schemas):
            return False

        return sorted(self.column_schemas, key=lambda x: x.name) == sorted(
            other.column_schemas, key=lambda x: x.name
        )

    def copy(self, **kwargs) -> "Schema":
        return proto_utils.copy_better_proto_message(self, **kwargs)

    def add(self, other, allow_overlap=True) -> "Schema":
        if isinstance(other, str):
            other = Schema.create([other])
        elif isinstance(other, collections.abc.Sequence):
            other = Schema(other)

        if not allow_overlap:
            # check if there are any columns with the same name in both column groups
            overlap = set(self.column_names).intersection(other.column_names)

            if overlap:
                raise ValueError(f"duplicate column names found: {overlap}")
            new_columns = self.column_schemas + other.column_schemas
        else:
            self_column_dict = {col.name: col for col in self.column_schemas}
            other_column_dict = {col.name: col for col in other.column_schemas}

            new_columns = [col for col in self.column_schemas]
            for key, val in other_column_dict.items():
                maybe_duplicate = self_column_dict.get(key, None)
                if maybe_duplicate:
                    merged_col = maybe_duplicate.with_tags(val.tags)
                    new_columns[new_columns.index(maybe_duplicate)] = merged_col
                else:
                    new_columns.append(val)

        return Schema(new_columns)

    def __add__(self, other):
        return self.add(other, allow_overlap=True)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if other is None:
            return self

        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for -: 'Schema' and {type(other)}")

        result = Schema(self.column_schemas)

        for key in other.column_schemas:
            if key in self.column_schemas:
                result.column_schemas.pop(key, None)

        return result

    @cached_property
    def item_id_column_name(self):
        item_id_col = self.select_by_tag("item_id")
        if len(item_id_col.column_names) == 0:
            raise ValueError("There is no column tagged as item id.")

        return item_id_col.column_names[0]

    def to_proto_text(self):
        from tensorflow_metadata.proto.v0 import schema_pb2

        return proto_utils.better_proto_to_proto_text(self, schema_pb2.Schema())

    def from_proto_text(self, path_or_proto_text):
        from tensorflow_metadata.proto.v0 import schema_pb2

        return proto_utils.proto_text_to_better_proto(self, path_or_proto_text, schema_pb2.Schema())
