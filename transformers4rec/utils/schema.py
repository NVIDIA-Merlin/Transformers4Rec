import collections.abc
import math
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


class DatasetSchema:
    """A collection of column schemas for a dataset."""

    def __init__(
        self,
        columns: Union[Text, List[Text], ColumnSchema, List[ColumnSchema], "DatasetSchema"],
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
    def from_schema(cls, schema) -> "DatasetSchema":
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
        """Adds columns from this DatasetSchema with another to return a new DatasetSchema
        Parameters
        -----------
        other: DatasetSchema or str or list of str
        Returns
        -------
        DatasetSchema
        """
        if isinstance(other, str):
            other = DatasetSchema([other])
        elif isinstance(other, collections.abc.Sequence):
            other = DatasetSchema(other)

        # check if there are any columns with the same name in both column groups
        overlap = set(self.column_names).intersection(other.column_names)

        if overlap:
            raise ValueError(f"duplicate column names found: {overlap}")

        new_schema = DatasetSchema(self.columns + other.columns)
        # TODO : set update method of the _schema
        # To keep it consistent over ops
        new_schema._schema = self.filter_schema(new_schema.column_names)
        return new_schema

    # handle the "column_name" + DatasetSchema case
    __radd__ = __add__

    def __sub__(self, other):
        """Removes columns from this DatasetSchema with another to return a new DatasetSchema
        Parameters
        -----------
        other: DatasetSchema or str or list of str
            Columns to remove
        Returns
        -------
        DatasetSchema
        """
        if isinstance(other, DatasetSchema):
            to_remove = set(other.column_names)
        elif isinstance(other, str):
            to_remove = {other}
        elif isinstance(other, collections.abc.Sequence):
            to_remove = set(other)
        else:
            raise ValueError(f"Expected DatasetSchema, str, or list of str. Got {other.__class__}")
        new_columns = [c for c in self.columns if c.name not in to_remove]
        new_schema = DatasetSchema(new_columns)
        # TODO : set update method of the _schema
        # To keep it consistent over ops
        new_schema._schema = self.filter_schema(new_schema.column_names)
        return new_schema

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

        child = DatasetSchema(output_cols, tags=tags)

        if self._schema:
            child._schema = self.filter_schema(child.column_names)

        return child

    def select_by_name(self, names):
        """Selects certain columns from this DatasetSchema, and returns a new DatasetSchema with only
        those columns
        Parameters
        -----------
        columns: str or list of str
            Columns to select
        Returns
        -------
        DatasetSchema
        """
        if isinstance(names, str):
            names = [names]

        output_cols = []
        for column in self.columns:
            if column.name in names:
                output_cols.append(column)

        child = DatasetSchema(output_cols)
        # TODO : set update method of the _schema
        # To keep it consistent over ops
        child._schema = self.filter_schema(child.column_names)
        return child

    def embedding_sizes(self, minimum_size=16, maximum_size=512) -> int:
        """Heuristic method to suggest the embedding sizes based on the categorical feature cardinality

        Parameters
        ----------
        minimum_size : int, optional
            Minimum embedding size, by default 16
        maximum_size : int, optional
            Minimum embedding size, by default 512

        Returns
        -------
        int
            The suggested embedding dimension
        """
        if not self._schema:
            raise ValueError(
                "The internal schema is required to retrieve "
                "the features cardinality and infer embeddings dim."
            )

        cardinalities = self.cardinalities()

        from nvtabular.ops.categorify import _emb_sz_rule

        return {
            key: _emb_sz_rule(val, minimum_size=minimum_size, maximum_size=maximum_size)[1]
            for key, val in cardinalities.items()
        }

    def embedding_sizes_v2(self, multiplier: float) -> int:
        """Heuristic method to suggest the embedding sizes based on the categorical feature cardinality

        Parameters
        ----------
        multiplier : float
            multiplier used by the heuristic to infer the embedding dimension from
            its cardinality. Generally reasonable values range between 2.0 and 10.0
        Returns
        -------
        int
            The suggested embedding dimension
        """
        if not self._schema:
            raise ValueError(
                "The internal schema is required to retrieve "
                " the features cardinality and infer embeddings dim."
            )

        if not (multiplier is not None and multiplier > 0.0):
            raise ValueError("The multiplier of the embedding size needs to be greater than 0.")

        cardinalities = self.cardinalities()
        return {
            key: DatasetSchema.get_embedding_size_from_cardinality(val, multiplier=multiplier)
            for key, val in cardinalities.items()
        }

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
                # f = f.annotation.tag
                f.CopyFrom(feat)

        return schema

    @staticmethod
    def get_embedding_size_from_cardinality(cardinality, multiplier=2.0):
        # A rule-of-thumb from Google.
        embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))
        return embedding_size


def _convert_col(col, tags=None):
    if isinstance(col, ColumnSchema):
        return col.with_tags(tags)
    elif isinstance(col, str):
        return ColumnSchema(col, tags=tags)
    elif isinstance(col, (tuple, list)):
        return [tuple(_convert_col(c, tags=tags) for c in col)]
    else:
        raise ValueError(f"Invalid column value for DatasetSchema: {col} (type: {type(col)})")
