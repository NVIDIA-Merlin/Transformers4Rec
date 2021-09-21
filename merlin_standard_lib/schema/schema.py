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
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from ..utils import proto_utils
from .tag import TagsType

try:
    from functools import cached_property  # type: ignore
except ImportError:
    # polyfill cached_property for python <= 3.7 (using lru_cache which was introduced in python3.2)
    from functools import lru_cache

    cached_property = lambda func: property(lru_cache()(func))  # noqa

import betterproto  # noqa

from ..proto.schema_bp import *  # noqa
from ..proto.schema_bp import (
    Annotation,
    Feature,
    FeatureType,
    FixedShape,
    FixedShapeDim,
    FloatDomain,
    IntDomain,
    ValueCount,
    ValueCountList,
    _Schema,
)


def _parse_shape_and_value_count(shape, value_count) -> Dict[str, Any]:
    output: Dict[str, Union[ValueCount, ValueCountList, FixedShape]] = {}
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
        tags: Optional[TagsType] = None,
        **kwargs,
    ) -> "ColumnSchema":
        _tags: List[str] = [str(t) for t in tags] if tags else []

        extra = _parse_shape_and_value_count(shape, value_count)
        int_domain = IntDomain(name=name, min=min_index, max=num_items, is_categorical=True)
        _tags = list(set(_tags + ["categorical"]))
        extra["type"] = FeatureType.INT

        return cls(name=name, int_domain=int_domain, **extra, **kwargs).with_tags(_tags)

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
        tags: Optional[TagsType] = None,
        **kwargs,
    ) -> "ColumnSchema":
        _tags: List[str] = [str(t) for t in tags] if tags else []

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
        extra["type"] = FeatureType.FLOAT if is_float else FeatureType.INT
        _tags = list(set(_tags + ["continuous"]))

        return cls(name=name, **extra, **kwargs).with_tags(_tags)

    def copy(self, **kwargs) -> "ColumnSchema":
        return proto_utils.copy_better_proto_message(self, **kwargs)

    def with_name(self, name: str):
        return self.copy(name=name)

    def with_tags(self, tags: TagsType) -> "ColumnSchema":
        tags = [str(t) for t in tags]
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

    def to_proto_text(self) -> str:
        from tensorflow_metadata.proto.v0 import schema_pb2

        return proto_utils.better_proto_to_proto_text(self, schema_pb2.Feature())

    @property
    def tags(self):
        return self.annotation.tag

    @property
    def properties(self) -> Dict[str, Union[str, float, int]]:
        if self.annotation.extra_metadata:
            properties: Dict[str, Union[str, float, int]] = self.annotation.extra_metadata[0]

            return properties

        return {}

    def _set_tags(self, tags: List[str]):
        if self.annotation:
            self.annotation.tag = list(set(list(self.annotation.tag) + tags))
        else:
            self.annotation = Annotation(tag=tags)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        if not isinstance(other, ColumnSchema):
            return NotImplemented

        return self.to_dict() == other.to_dict()


ColumnSchemaOrStr = Union[ColumnSchema, str]

FilterT = TypeVar("FilterT")


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

        features: List[ColumnSchema] = []
        if isinstance(column_schemas, list):
            for column_schema in column_schemas:
                if isinstance(column_schema, str):
                    features.append(ColumnSchema(column_schema))
                else:
                    features.append(column_schema)
        else:
            raise TypeError("The `column_schemas` parameter must be a list or dict.")

        return cls(feature=features, **kwargs)

    def with_tags_based_on_properties(self, using_value_count=True, using_domain=True) -> "Schema":
        column_schemas = []
        for column in self.column_schemas:
            column_schemas.append(
                column.with_tags_based_on_properties(
                    using_value_count=using_value_count, using_domain=using_domain
                )
            )

        return Schema(column_schemas)

    def apply(self, selector) -> "Schema":
        if selector and selector.names:
            return self.select_by_name(selector.names)
        else:
            return self

    def apply_inverse(self, selector) -> "Schema":
        if selector:
            output_schema: Schema = self - self.select_by_name(selector.names)

            return output_schema
        else:
            return self

    def filter_columns_from_dict(self, input_dict):
        filtered_dict = {}
        for key, val in input_dict.items():
            if key in self.column_names:
                filtered_dict[key] = val

        return filtered_dict

    def select_by_type(self, to_select) -> "Schema":
        if not isinstance(to_select, (list, tuple)) and not callable(to_select):
            to_select = [to_select]

        def collection_filter_fn(type):
            return type in to_select

        output: Schema = self._filter_column_schemas(
            to_select, collection_filter_fn, lambda x: x.type
        )

        return output

    def remove_by_type(self, to_remove) -> "Schema":
        if not isinstance(to_remove, (list, tuple)) and not callable(to_remove):
            to_remove = [to_remove]

        def collection_filter_fn(type):
            return type in to_remove

        output: Schema = self._filter_column_schemas(
            to_remove, collection_filter_fn, lambda x: x.type, negate=True
        )

        return output

    def select_by_tag(self, to_select) -> "Schema":
        if not isinstance(to_select, (list, tuple)) and not callable(to_select):
            to_select = [to_select]

        def collection_filter_fn(column_tags):
            return all(x in column_tags for x in to_select)

        output: Schema = self._filter_column_schemas(
            to_select, collection_filter_fn, lambda x: x.tags
        )

        return output

    def remove_by_tag(self, to_remove) -> "Schema":
        if not isinstance(to_remove, (list, tuple)) and not callable(to_remove):
            to_remove = [to_remove]

        def collection_filter_fn(column_tags):
            return all(x in column_tags for x in to_remove)

        return self._filter_column_schemas(
            to_remove, collection_filter_fn, lambda x: x.tags, negate=True
        )

    def select_by_name(self, to_select) -> "Schema":
        if not isinstance(to_select, (list, tuple)) and not callable(to_select):
            to_select = [to_select]

        def collection_filter_fn(column_name):
            return column_name in to_select

        output: Schema = self._filter_column_schemas(
            to_select, collection_filter_fn, lambda x: x.name
        )

        return output

    def remove_by_name(self, to_remove) -> "Schema":
        if not isinstance(to_remove, (list, tuple)) and not callable(to_remove):
            to_remove = [to_remove]

        def collection_filter_fn(column_name):
            return column_name in to_remove

        return self._filter_column_schemas(
            to_remove, collection_filter_fn, lambda x: x.name, negate=True
        )

    def map_column_schemas(self, map_fn: Callable[[ColumnSchema], ColumnSchema]) -> "Schema":
        output_schemas = []
        for column_schema in self.column_schemas:
            output_schemas.append(map_fn(column_schema))

        return Schema(output_schemas)

    def filter_column_schemas(
        self, filter_fn: Callable[[ColumnSchema], bool], negate=False
    ) -> "Schema":
        selected_schemas = []
        for column_schema in self.column_schemas:
            if self._check_column_schema(column_schema, filter_fn, negate=negate):
                selected_schemas.append(column_schema)

        return Schema(selected_schemas)

    def categorical_cardinalities(self) -> Dict[str, int]:
        outputs = {}
        for col in self:
            if col.int_domain and col.int_domain.is_categorical:
                outputs[col.name] = col.int_domain.max + 1

        return outputs

    @property
    def column_names(self) -> List[str]:
        return [f.name for f in self.feature]

    @property
    def column_schemas(self) -> Sequence[ColumnSchema]:
        return self.feature

    @cached_property
    def item_id_column_name(self):
        item_id_col = self.select_by_tag("item_id")
        if len(item_id_col.column_names) == 0:
            raise ValueError("There is no column tagged as item id.")

        return item_id_col.column_names[0]

    def from_json(self, value: Union[str, bytes]) -> "Schema":
        if os.path.isfile(value):
            with open(value, "rb") as f:
                value = f.read()

        return super().from_json(value)

    def to_proto_text(self) -> str:
        from tensorflow_metadata.proto.v0 import schema_pb2

        return proto_utils.better_proto_to_proto_text(self, schema_pb2.Schema())

    def from_proto_text(self, path_or_proto_text: str) -> "Schema":
        from tensorflow_metadata.proto.v0 import schema_pb2

        return proto_utils.proto_text_to_better_proto(self, path_or_proto_text, schema_pb2.Schema())

    def copy(self, **kwargs) -> "Schema":
        return proto_utils.copy_better_proto_message(self, **kwargs)

    def add(self, other, allow_overlap=True) -> "Schema":
        if isinstance(other, str):
            other = Schema.create([other])
        elif isinstance(other, collections.abc.Sequence):  # type: ignore
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

    def _filter_column_schemas(
        self,
        to_filter: Union[list, tuple, Callable[[FilterT], bool]],
        collection_filter_fn: Callable[[FilterT], bool],
        column_select_fn: Callable[[ColumnSchema], FilterT],
        negate=False,
    ) -> "Schema":
        if isinstance(to_filter, (list, tuple)):
            check_fn = collection_filter_fn
        elif callable(to_filter):
            check_fn = to_filter
        else:
            raise ValueError(f"Expected either a collection or function, got: {to_filter}.")

        selected_schemas = []
        for column_schema in self.column_schemas:
            if self._check_column_schema(column_select_fn(column_schema), check_fn, negate=negate):
                selected_schemas.append(column_schema)

        return Schema(selected_schemas)

    def _check_column_schema(
        self, inputs: FilterT, filter_fn: Callable[[FilterT], bool], negate=False
    ) -> bool:
        check = filter_fn(inputs)
        if check and not negate:
            return True
        elif not check and negate:
            return True

        return False

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
