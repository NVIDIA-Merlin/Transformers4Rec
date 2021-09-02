import pytest

from transformers4rec.utils.tags import Tag

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
    assert "schema is necessary" in str(excinfo.value)


def test_element_wise_sum_item_multi_col_group_no_item_id(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:
        categ_schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
        # Remove the item id from col_group
        categ_schema = categ_schema - ["item_id/list"]
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


# Uncomment this when `TabularSequenceFeatures` is ported to TF.
# def test_element_wise_sum_item_multi_aggregation_registry_yoochoose(
#     yoochoose_schema, tf_yoochoose_like
# ):
#     categ_schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
#
#     tab_module = tf4rec.TabularSequenceFeatures.from_schema(
#         categ_schema, aggregation="element-wise-sum-item-multi"
#     )
#
#     out = tab_module(tf_yoochoose_like)
#
#     assert out.shape[-1] == 64
