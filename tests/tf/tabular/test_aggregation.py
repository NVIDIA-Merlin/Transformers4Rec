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

tf = pytest.importorskip("tensorflow")
tf4rec = pytest.importorskip("transformers4rec.tf")


def test_concat_aggregation_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tf4rec.TabularFeatures.from_schema(schema)

    block = tab_module >> tf4rec.ConcatFeatures()

    out = block(tf_yoochoose_like)

    assert out.shape[-1] == 248


def test_stack_aggregation_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tf4rec.EmbeddingFeatures.from_schema(schema)

    block = tab_module >> tf4rec.StackFeatures()

    out = block(tf_yoochoose_like)

    assert out.shape[1] == 64
    assert out.shape[2] == 2


def test_element_wise_sum_features_different_shapes():
    with pytest.raises(ValueError) as excinfo:
        element_wise_op = tf4rec.ElementwiseSum()
        input = {
            "item_id/list": tf.random.uniform((10, 20)),
            "category/list": tf.random.uniform((10, 25)),
        }
        element_wise_op(input)
    assert "shapes of all input features are not equal" in str(excinfo.value)


def test_element_wise_sum_aggregation_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tf4rec.EmbeddingFeatures.from_schema(schema)

    block = tab_module >> tf4rec.ElementwiseSum()

    out = block(tf_yoochoose_like)

    assert out.shape[-1] == 64


def test_element_wise_sum_item_multi_no_col_group():
    with pytest.raises(ValueError) as excinfo:
        element_wise_op = tf4rec.ElementwiseSumItemMulti()
        element_wise_op(None)
    assert "requires a schema." in str(excinfo.value)


def test_element_wise_sum_item_multi_col_group_no_item_id(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:
        categ_schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
        # Remove the item id from col_group
        categ_schema = categ_schema.remove_by_name("item_id/list")
        element_wise_op = tf4rec.ElementwiseSumItemMulti(categ_schema)
        element_wise_op(None)
    assert "no column tagged as item id" in str(excinfo.value)


def test_element_wise_sum_item_multi_features_different_shapes(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:
        categ_schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
        element_wise_op = tf4rec.ElementwiseSumItemMulti(categ_schema)
        input = {
            "item_id/list": tf.random.uniform((10, 20)),
            "category/list": tf.random.uniform((10, 25)),
        }
        element_wise_op(input)
    assert "shapes of all input features are not equal" in str(excinfo.value)


def test_element_wise_sum_item_multi_aggregation_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tf4rec.EmbeddingFeatures.from_schema(schema)

    block = tab_module >> tf4rec.ElementwiseSumItemMulti(schema)

    out = block(tf_yoochoose_like)

    assert out.shape[-1] == 64


def test_element_wise_sum_item_multi_aggregation_registry_yoochoose(
    yoochoose_schema, tf_yoochoose_like
):
    tab_module = tf4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema.select_by_tag(Tag.CATEGORICAL), aggregation="element-wise-sum-item-multi"
    )

    out = tab_module(tf_yoochoose_like)

    assert out.shape[-1] == 64
