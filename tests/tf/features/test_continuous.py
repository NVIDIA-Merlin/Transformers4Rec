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

tr = pytest.importorskip("transformers4rec.tf")
test_utils = pytest.importorskip("transformers4rec.tf.utils.testing_utils")


def test_continuous_features(tf_con_features):
    features = ["a", "b"]
    con = tr.ContinuousFeatures(features)(tf_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CONTINUOUS)

    inputs = tr.ContinuousFeatures.from_schema(schema)
    outputs = inputs(tf_yoochoose_like)

    assert list(outputs.keys()) == schema.column_names


def test_serialization_continuous_features(yoochoose_schema, tf_yoochoose_like):
    inputs = tr.ContinuousFeatures.from_schema(yoochoose_schema)

    copy_layer = test_utils.assert_serialization(inputs)

    assert inputs.filter_features.to_include == copy_layer.filter_features.to_include


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_continuous_features_yoochoose_model(yoochoose_schema, tf_yoochoose_like, run_eagerly):
    schema = yoochoose_schema.select_by_tag(Tag.CONTINUOUS)

    inputs = tr.ContinuousFeatures.from_schema(schema, aggregation="concat")
    body = tr.SequentialBlock([inputs, tr.MLPBlock([64])])

    test_utils.assert_body_works_in_model(tf_yoochoose_like, inputs, body, run_eagerly)
