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


def test_tabular_features(tabular_schema, tf_tabular_data):
    tab_module = tr.TabularFeatures.from_schema(tabular_schema)

    outputs = tab_module(tf_tabular_data)

    assert set(outputs.keys()) == set(
        tabular_schema.select_by_tag(Tag.CONTINUOUS).column_names
        + tabular_schema.select_by_tag(Tag.CATEGORICAL).column_names
    )


def test_serialization_tabular_features(tabular_schema):
    inputs = tr.TabularFeatures.from_schema(tabular_schema)

    copy_layer = test_utils.assert_serialization(inputs)

    assert list(inputs.to_merge.keys()) == list(copy_layer.to_merge.keys())


def test_tabular_features_with_projection(tabular_schema, tf_tabular_data):
    schema = tabular_schema
    tab_module = tr.TabularFeatures.from_schema(tabular_schema, continuous_projection=64)

    outputs = tab_module(tf_tabular_data)
    continuous_feature_names = schema.select_by_tag(Tag.CONTINUOUS).column_names

    assert len(set(continuous_feature_names).intersection(set(outputs.keys()))) == 0
    assert "continuous_projection" in outputs
    assert list(outputs["continuous_projection"].shape)[1] == 64


@test_utils.mark_run_eagerly_modes
@pytest.mark.parametrize("continuous_projection", [None, 128])
def test_tabular_features_yoochoose_model(
    tabular_schema, tf_tabular_data, run_eagerly, continuous_projection
):
    inputs = tr.TabularFeatures.from_schema(
        tabular_schema, continuous_projection=continuous_projection, aggregation="concat"
    )

    body = tr.SequentialBlock([inputs, tr.MLPBlock([64])])

    test_utils.assert_body_works_in_model(tf_tabular_data, inputs, body, run_eagerly)
