import pytest

from tests.tf._utils import assert_body_works_in_model, assert_serialization
from transformers4rec.utils.tags import Tag

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_continuous_features(tf_con_features):
    features = ["a", "b"]
    con = tf4rec.ContinuousFeatures(features)(tf_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CONTINUOUS)

    inputs = tf4rec.ContinuousFeatures.from_schema(schema)
    outputs = inputs(tf_yoochoose_like)

    assert list(outputs.keys()) == schema.column_names


def test_serialization_continuous_features(yoochoose_schema, tf_yoochoose_like):
    inputs = tf4rec.ContinuousFeatures.from_schema(yoochoose_schema)

    copy_layer = assert_serialization(inputs)

    assert inputs.filter_features.to_include == copy_layer.filter_features.to_include


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_continuous_features_yoochoose_model(yoochoose_schema, tf_yoochoose_like, run_eagerly):
    schema = yoochoose_schema.select_by_tag(Tag.CONTINUOUS)

    inputs = tf4rec.ContinuousFeatures.from_schema(schema, aggregation="concat")
    body = tf4rec.SequentialBlock([inputs, tf4rec.MLPBlock([64])])

    assert_body_works_in_model(tf_yoochoose_like, inputs, body, run_eagerly)
