from typing import Optional, Tuple, Union

from .schema_proto import *  # noqa


class ColumnSchema(Feature):
    @classmethod
    def create_categorical(
        cls,
        name: str,
        num_items: int,
        shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
        value_count: Optional[Union[ValueCount, ValueCountList]] = None,
        min_index: int = 0,
        **kwargs,
    ) -> "ColumnSchema":
        if shape:
            kwargs["shape"] = FixedShape([FixedShapeDim(d) for d in shape])

        if value_count:
            if isinstance(value_count, ValueCount):
                kwargs["value_count"] = value_count
            elif isinstance(value_count, ValueCountList):
                kwargs["value_counts"] = value_count
            else:
                raise ValueError("Unknown value_count type.")
        int_domain = IntDomain(name=name, min=min_index, max=num_items, is_categorical=True)

        return cls(name=name, int_domain=int_domain, **kwargs)

    # @classmethod
    # def create_continuous(cls)

    def __str__(self) -> str:
        return self.name

    def copy(self, **kwargs) -> "ColumnSchema":
        output = self.from_json(self.to_json())
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

    @property
    def tags(self):
        return self.annotation.tag

    @property
    def properties(self):
        return self.annotation.extra_metadata


class Schema:
    """A collection of column schemas for a dataset."""

    def __init__(
        self, column_schemas: Optional[Union[List[ColumnSchema], Dict[str, ColumnSchema]]] = None
    ):
        column_schemas = column_schemas or {}

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

    def __iter__(self):
        return iter(self.column_schemas.values)

    def __len__(self):
        return len(self.column_schemas)

    def __repr__(self):
        return str([col_schema.__dict__ for col_schema in self.column_schemas.values()])

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
