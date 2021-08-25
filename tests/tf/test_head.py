import pytest

tf = pytest.importorskip("tensorflow")
tf4rec = pytest.importorskip("transformers4rec.tf")


def assert_loss_and_metrics_are_valid(head, inputs, targets):
    loss = head.compute_loss(inputs, targets, call_body=True)
    metrics = head.metric_results()

    assert 0 <= loss <= 1
    assert len(metrics) == len(head.metrics)
    assert all(0 <= metric <= 1 for metric in metrics.values())


@pytest.mark.parametrize(
    "prediction_task", [tf4rec.BinaryClassificationTask, tf4rec.RegressionTask]
)
def test_simple_heads(tf_yoochoose_tabular_features, tf_yoochoose_like, prediction_task):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = tf4rec.SequentialBlock([tf_yoochoose_tabular_features, tf4rec.MLPBlock([64])])
    task = prediction_task("target")
    head = task.to_head(body, tf_yoochoose_tabular_features)

    assert_loss_and_metrics_are_valid(head, tf_yoochoose_like, targets)


@pytest.mark.parametrize("task", [tf4rec.BinaryClassificationTask, tf4rec.RegressionTask])
@pytest.mark.parametrize("task_block", [None, tf4rec.MLPBlock([32])])
@pytest.mark.parametrize("summary", ["last", "first", "mean", "cls_index"])
def test_simple_heads_on_sequence(
    tf_yoochoose_tabular_sequence_features, tf_yoochoose_like, task, task_block, summary
):
    inputs = tf_yoochoose_tabular_sequence_features
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = tf4rec.SequentialBlock([inputs, tf4rec.MLPBlock([64])])
    head = task("target", task_block=task_block, summary_type=summary).to_head(body, inputs)

    assert_loss_and_metrics_are_valid(head, tf_yoochoose_like, targets)
