import pytest

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_dlrm_block_yoochoose(yoochoose_schema, tf_yoochoose_like):
    dlrm = tf4rec.DLRMBlock.from_schema(yoochoose_schema, bottom_mlp=tf4rec.MLPBlock([64]))

    body = tf4rec.SequentialBlock([dlrm, tf4rec.MLPBlock([64])])

    outputs = body(tf_yoochoose_like)

    assert list(outputs.shape) == [100, 64]
