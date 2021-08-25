import pytest

tf = pytest.importorskip("tensorflow")
tf4rec = pytest.importorskip("transformers4rec.tf")


@pytest.mark.parametrize(
    "prediction_task", [tf4rec.BinaryClassificationTask, tf4rec.RegressionTask]
)
def test_simple_heads(tf_yoochoose_tabular_features, tf_yoochoose_like, prediction_task):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = tf4rec.SequentialBlock([tf_yoochoose_tabular_features, tf4rec.MLPBlock([64])])
    task = prediction_task("target")
    head = task.to_head(body, tf_yoochoose_tabular_features)

    body_out = body(tf_yoochoose_like)
    loss = head.compute_loss(body_out, targets)
    metrics = head.metric_results()

    assert 0 <= loss <= 1
    assert len(metrics) == len(task.metrics)
    assert all(0 <= metric <= 1 for metric in metrics.values())
