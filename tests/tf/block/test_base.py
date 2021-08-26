import pytest

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_sequential_block_yoochoose(yoochoose_schema, tf_yoochoose_like):
    inputs = tf4rec.TabularFeatures.from_schema(yoochoose_schema, aggregation="concat")

    body = tf4rec.SequentialBlock([inputs, tf4rec.MLPBlock([64])])

    outputs = body(tf_yoochoose_like)

    assert list(outputs.shape) == [100, 64]


def test_sequential_block_yoochoose_without_aggregation(yoochoose_schema, tf_yoochoose_like):
    inputs = tf4rec.TabularFeatures.from_schema(yoochoose_schema)

    with pytest.raises(TypeError) as excinfo:
        body = tf4rec.SequentialBlock([inputs, tf4rec.MLPBlock([64])])

        body(tf_yoochoose_like)

        assert "did you forget to add aggregation to TabularFeatures" in str(excinfo.value)
