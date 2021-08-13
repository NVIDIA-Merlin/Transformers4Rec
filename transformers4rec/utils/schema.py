import collections.abc
from dataclasses import dataclass, field
from typing import List, Optional, Text, Union

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

from .tags import DefaultTags, Tag


@dataclass(frozen=True)
class ColumnSchema:
    """ "A Column with metadata."""

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.tags, DefaultTags):
            object.__setattr__(self, "tags", self.tags.value)

    def __str__(self) -> str:
        return self.name

    def with_tags(self, tags, add=True) -> "ColumnSchema":
        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not tags:
            return self

        tags = list(set(list(self.tags) + list(tags))) if add else tags

        return ColumnSchema(self.name, tags=tags)

    def with_name(self, name) -> "ColumnSchema":
        return ColumnSchema(name, tags=self.tags)


class Schema:
    """A Schema is a group of columns that you want to apply the same transformations to.
    Schema's can be transformed by shifting operators on to them, which returns a new
    Schema with the transformations applied. This lets you define a graph of operations
    that makes up your workflow
    Parameters
    ----------
    columns: list of (str or tuple of str)
        The columns to select from the input Dataset. The elements of this list are strings
        indicating the column names in most cases, but can also be tuples of strings
        for feature crosses.
    """

    def __init__(
        self,
        columns: Union[Text, List[Text], ColumnSchema, List[ColumnSchema], "Schema"],
        tags: Optional[Union[List[Text], DefaultTags]] = None,
    ):
        if isinstance(columns, str):
            columns = [columns]

        if not tags:
            tags = []
        if isinstance(tags, DefaultTags):
            tags = tags.value

        self.tags = tags

        self.columns: List[ColumnSchema] = [_convert_col(col, tags=tags) for col in columns]
        self.set_schema(None)

    @staticmethod
    def read_schema(schema_path):
        with open(schema_path, "rb") as f:
            schema = schema_pb2.Schema()
            text_format.Parse(f.read(), schema)

        return schema

    def set_schema(self, schema):
        self._schema = schema

    @classmethod
    def from_schema(cls, schema) -> "Schema":
        if isinstance(schema, str):
            schema = cls.read_schema(schema)

        columns = []
        for feat in schema.feature:
            tags = feat.annotation.tag
            if feat.HasField("value_count"):
                tags = list(tags) + Tag.LIST.value if tags else Tag.LIST.value
            columns.append(ColumnSchema(feat.name, tags=tags))

        output = cls(columns)
        output.set_schema(schema)

        return output

    @property
    def column_names(self):
        return [col.name for col in self.columns]

    def __add__(self, other):
        """Adds columns from this Schema with another to return a new Schema
        Parameters
        -----------
        other: Schema or str or list of str
        Returns
        -------
        Schema
        """
        if isinstance(other, str):
            other = Schema([other])
        elif isinstance(other, collections.abc.Sequence):
            other = Schema(other)

        # check if there are any columns with the same name in both column groups
        overlap = set(self.column_names).intersection(other.column_names)

        if overlap:
            raise ValueError(f"duplicate column names found: {overlap}")

        child = Schema(self.columns + other.columns)
        child.parents = [self, other]
        child.kind = "+"
        return child

    # handle the "column_name" + Schema case
    __radd__ = __add__

    def __sub__(self, other):
        """Removes columns from this Schema with another to return a new Schema
        Parameters
        -----------
        other: Schema or str or list of str
            Columns to remove
        Returns
        -------
        Schema
        """
        if isinstance(other, Schema):
            to_remove = set(other.column_names)
        elif isinstance(other, str):
            to_remove = {other}
        elif isinstance(other, collections.abc.Sequence):
            to_remove = set(other)
        else:
            raise ValueError(f"Expected Schema, str, or list of str. Got {other.__class__}")
        new_columns = [c for c in self.columns if c.name not in to_remove]
        child = Schema(new_columns)

        return child

    def __getitem__(self, columns):
        return self.select_by_name(columns)

    def select_by_tag(self, tags):
        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not isinstance(tags, list):
            tags = [tags]
        output_cols = []

        for column in self.columns:
            if all(x in column.tags for x in tags):
                output_cols.append(column)

        child = Schema(output_cols, tags=tags)

        if self._schema:
            child._schema = self.filter_schema(child.column_names)

        return child

    def select_by_name(self, names):
        """Selects certain columns from this Schema, and returns a new Schema with only
        those columns
        Parameters
        -----------
        columns: str or list of str
            Columns to select
        Returns
        -------
        Schema
        """
        if isinstance(names, str):
            names = [names]

        child = Schema(names)
        child.parents = [self]
        self.children.append(child)
        child.kind = str(names)
        return child

    def embedding_sizes(self):
        if self._schema:
            cardinalities = self.cardinalities()

            from nvtabular.ops.categorify import _emb_sz_rule

            return {key: _emb_sz_rule(val) for key, val in cardinalities.items()}

    def cardinalities(self):
        if self._schema:
            outputs = {}
            for feature in self._schema.feature:
                if feature.int_domain and feature.int_domain.is_categorical:
                    outputs[feature.name] = feature.int_domain.max

            return outputs

    def filter_schema(self, columns):
        if not self._schema:
            return None

        schema = schema_pb2.Schema()

        for feat in self._schema.feature:
            if feat.name in columns:
                f = schema.feature.add()
                f.CopyFrom(feat)

        return schema


def _convert_col(col, tags=None):
    if isinstance(col, ColumnSchema):
        return col.with_tags(tags)
    elif isinstance(col, str):
        return ColumnSchema(col, tags=tags)
    elif isinstance(col, (tuple, list)):
        return [tuple(_convert_col(c, tags=tags) for c in col)]
    else:
        raise ValueError(f"Invalid column value for Schema: {col} (type: {type(col)})")
