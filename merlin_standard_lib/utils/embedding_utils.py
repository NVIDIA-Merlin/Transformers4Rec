import math

from merlin_standard_lib import Schema


def get_embedding_sizes_from_schema(schema: Schema, multiplier: float = 2.0):
    cardinalities = schema.categorical_cardinalities()

    return {
        key: get_embedding_size_from_cardinality(val, multiplier)
        for key, val in cardinalities.items()
    }


def get_embedding_size_from_cardinality(cardinality: int, multiplier: float = 2.0):
    # A rule-of-thumb from Google.
    embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))

    return embedding_size
