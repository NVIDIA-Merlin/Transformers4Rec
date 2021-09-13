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

from transformers4rec.tf.utils.testing_utils import assert_serialization

tr = pytest.importorskip("transformers4rec.tf")


def test_filter_features(tf_con_features):
    features = ["a", "b"]
    con = tr.FilterFeatures(features)(tf_con_features)

    assert list(con.keys()) == features


def test_as_tabular(tf_con_features):
    name = "tabular"
    con = tr.AsTabular(name)(tf_con_features)

    assert list(con.keys()) == [name]


def test_tabular_module(tf_con_features):
    _DummyTabular = tr.TabularBlock

    tabular = _DummyTabular()

    assert tabular(tf_con_features) == tf_con_features
    assert tabular(tf_con_features, aggregation="concat").shape[1] == 6
    assert tabular(tf_con_features, aggregation=tr.ConcatFeatures()).shape[1] == 6

    tabular_concat = _DummyTabular(aggregation="concat")
    assert tabular_concat(tf_con_features).shape[1] == 6

    tab_a = ["a"] >> _DummyTabular()
    tab_b = ["b"] >> _DummyTabular()

    assert tab_a(tf_con_features, merge_with=tab_b, aggregation="stack").shape[1] == 1


@pytest.mark.parametrize("pre", [None, "stochastic-swap-noise"])
@pytest.mark.parametrize("post", [None, "stochastic-swap-noise"])
@pytest.mark.parametrize("aggregation", [None, "concat"])
@pytest.mark.parametrize("include_schema", [True, False])
def test_serialization_continuous_features(
    yoochoose_schema, tf_yoochoose_like, pre, post, aggregation, include_schema
):
    schema = yoochoose_schema if include_schema else None
    inputs = tr.TabularBlock(pre=pre, post=post, aggregation=aggregation, schema=schema)

    copy_layer = assert_serialization(inputs)

    assert copy_layer(tf_yoochoose_like) is not None
    assert inputs.pre.__class__.__name__ == copy_layer.pre.__class__.__name__
    assert inputs.post.__class__.__name__ == copy_layer.post.__class__.__name__
    assert inputs.aggregation.__class__.__name__ == copy_layer.aggregation.__class__.__name__
    assert inputs.schema == schema
