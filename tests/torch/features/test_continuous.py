#
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

import pytest

from merlin_standard_lib import Tag

tr = pytest.importorskip("transformers4rec.torch")


def test_continuous_features(torch_con_features):
    features = ["con_a", "con_b"]
    con = tr.ContinuousFeatures(features)(torch_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(yoochoose_schema, torch_yoochoose_like):
    schema = yoochoose_schema
    cont_cols = schema.select_by_tag(Tag.CONTINUOUS)

    con = tr.ContinuousFeatures.from_schema(cont_cols)
    outputs = con(torch_yoochoose_like)

    assert set(outputs.keys()) == set(cont_cols.column_names)
