import collections.abc
from dataclasses import dataclass, field
from typing import List, Optional, Text, Union

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

from .tags import DefaultTags, Tag


@dataclass(frozen=True)
class Column:
    """"A Column with metadata. """

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.tags, DefaultTags):
            object.__setattr__(self, "tags", self.tags.value)

    def __str__(self) -> str:
        return self.name

    def with_tags(self, tags, add=True) -> "Column":
        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not tags:
            return self

        tags = list(set(list(self.tags) + list(tags))) if add else tags

        return Column(self.name, tags=tags)

    def with_name(self, name) -> "Column":
        return Column(name, tags=self.tags)


class ColumnGroup:
    """A ColumnGroup is a group of columns that you want to apply the same transformations to.
    ColumnGroup's can be transformed by shifting operators on to them, which returns a new
    ColumnGroup with the transformations applied. This lets you define a graph of operations
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
        columns: Union[Text, List[Text], Column, List[Column], "ColumnGroup"],
        tags: Optional[Union[List[Text], DefaultTags]] = None,
    ):
        if isinstance(columns, str):
            columns = [columns]

        if not tags:
            tags = []
        if isinstance(tags, DefaultTags):
            tags = tags.value

        self.tags = tags

        self.columns: List[Column] = [_convert_col(col, tags=tags) for col in columns]

    @staticmethod
    def read_schema(schema_path):
        with open(schema_path, "rb") as f:
            schema = schema_pb2.Schema()
            text_format.Parse(f.read(), schema)

        return schema

    def set_schema(self, schema):
        self._schema = schema

    @classmethod
    def from_schema(cls, schema) -> "ColumnGroup":
        if isinstance(schema, str):
            schema = cls.read_schema(schema)

        columns = []
        for feat in schema.feature:
            tags = feat.annotation.tag
            if feat.HasField("value_count"):
                tags = list(tags) + Tag.LIST.value if tags else Tag.LIST.value
            columns.append(Column(feat.name, tags=tags))

        output = cls(columns)
        output.set_schema(schema)

        return output

    @property
    def column_names(self):
        return [col.name for col in self.columns]

    def __add__(self, other):
        """Adds columns from this ColumnGroup with another to return a new ColumnGroup
        Parameters
        -----------
        other: ColumnGroup or str or list of str
        Returns
        -------
        ColumnGroup
        """
        if isinstance(other, str):
            other = ColumnGroup([other])
        elif isinstance(other, collections.abc.Sequence):
            other = ColumnGroup(other)

        # check if there are any columns with the same name in both column groups
        overlap = set(self.column_names).intersection(other.column_names)

        if overlap:
            raise ValueError(f"duplicate column names found: {overlap}")

        child = ColumnGroup(self.columns + other.columns)
        child.parents = [self, other]
        child.kind = "+"
        return child

    # handle the "column_name" + ColumnGroup case
    __radd__ = __add__

    def __sub__(self, other):
        """Removes columns from this ColumnGroup with another to return a new ColumnGroup
        Parameters
        -----------
        other: ColumnGroup or str or list of str
            Columns to remove
        Returns
        -------
        ColumnGroup
        """
        if isinstance(other, ColumnGroup):
            to_remove = set(other.column_names)
        elif isinstance(other, str):
            to_remove = {other}
        elif isinstance(other, collections.abc.Sequence):
            to_remove = set(other)
        else:
            raise ValueError(f"Expected ColumnGroup, str, or list of str. Got {other.__class__}")
        new_columns = [c for c in self.columns if c.name not in to_remove]
        child = ColumnGroup(new_columns)

        return child

    def __getitem__(self, columns):
        return self.select_by_name(columns)

    def select_by_tag(self, tags, tags_to_filter=None):
        column_names_to_filter = (
            self.select_by_tag(tags_to_filter, output_list=True) if tags_to_filter else []
        )

        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not isinstance(tags, list):
            tags = [tags]
        output_cols = []

        for column in self.columns:
            if all(x in column.tags for x in tags):
                output_cols.append(column)

        columns = [col for col in output_cols if col.name not in column_names_to_filter]

        child = ColumnGroup(columns, tags=tags)

        return child

    def select_by_name(self, names):
        """Selects certain columns from this ColumnGroup, and returns a new Columngroup with only
        those columns
        Parameters
        -----------
        columns: str or list of str
            Columns to select
        Returns
        -------
        ColumnGroup
        """
        if isinstance(names, str):
            names = [names]

        child = ColumnGroup(names)
        child.parents = [self]
        self.children.append(child)
        child.kind = str(names)
        return child


def _convert_col(col, tags=None):
    if isinstance(col, Column):
        return col.with_tags(tags)
    elif isinstance(col, str):
        return Column(col, tags=tags)
    elif isinstance(col, (tuple, list)):
        return [tuple(_convert_col(c, tags=tags) for c in col)]
    else:
        raise ValueError(f"Invalid column value for ColumnGroup: {col} (type: {type(col)})")
