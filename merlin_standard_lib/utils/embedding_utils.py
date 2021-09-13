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
