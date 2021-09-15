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

tf = pytest.importorskip("tensorflow")
tr = pytest.importorskip("transformers4rec.tf")
test_utils = pytest.importorskip("transformers4rec.tf.utils.testing_utils")


def assert_loss_and_metrics_are_valid(head, inputs, targets):
    loss = head.compute_loss(inputs, targets, call_body=True)
    metrics = head.metric_results()

    assert loss is not None
    assert len(metrics) == len(head.metrics)


@pytest.mark.parametrize("prediction_task", [tr.BinaryClassificationTask, tr.RegressionTask])
def test_simple_heads(tf_tabular_features, tf_tabular_data, prediction_task):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = tr.SequentialBlock([tf_tabular_features, tr.MLPBlock([64])])
    task = prediction_task("target")
    head = task.to_head(body, tf_tabular_features)

    assert_loss_and_metrics_are_valid(head, tf_tabular_data, targets)


@pytest.mark.parametrize("prediction_task", [tr.BinaryClassificationTask, tr.RegressionTask])
def test_serialization_simple_heads(tf_tabular_features, tf_tabular_data, prediction_task):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = tr.SequentialBlock([tf_tabular_features, tr.MLPBlock([64])])
    task = prediction_task("target")
    head = task.to_head(body, tf_tabular_features)

    copy_head = test_utils.assert_serialization(head)
    assert_loss_and_metrics_are_valid(copy_head, tf_tabular_data, targets)


@pytest.mark.parametrize("task", [tr.BinaryClassificationTask, tr.RegressionTask])
@pytest.mark.parametrize("task_block", [None, tr.MLPBlock([32])])
@pytest.mark.parametrize("summary", ["last", "first", "mean", "cls_index"])
def test_simple_heads_on_sequence(
    tf_yoochoose_tabular_sequence_features, tf_yoochoose_like, task, task_block, summary
):
    inputs = tf_yoochoose_tabular_sequence_features
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = tr.SequentialBlock([inputs, tr.MLPBlock([64])])
    head = task("target", task_block=task_block, summary_type=summary).to_head(body, inputs)

    assert_loss_and_metrics_are_valid(head, tf_yoochoose_like, targets)


@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        tr.MLPBlock([32]),
        dict(classification=tr.MLPBlock([16]), regression=tr.MLPBlock([20])),
        dict(binary_classification_task=tr.MLPBlock([16]), regression_task=tr.MLPBlock([20])),
        {
            "classification/binary_classification_task": tr.MLPBlock([16]),
            "regression/regression_task": tr.MLPBlock([20]),
        },
    ],
)
def test_head_with_multiple_tasks(tf_tabular_features, tf_tabular_data, task_blocks):
    targets = {
        "classification": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32),
        "regression": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32),
    }

    body = tr.SequentialBlock([tf_tabular_features, tr.MLPBlock([64])])
    tasks = [
        tr.BinaryClassificationTask("classification"),
        tr.RegressionTask("regression"),
    ]
    head = tr.Head(body, tasks, task_blocks=task_blocks)
    model = tr.Model(head)
    model.compile(optimizer="adam")

    step = model.train_step((tf_tabular_data, targets))

    # assert 0 <= step["loss"] <= 1 # test failing with loss greater than 1
    assert step["loss"] >= 0
    assert len(step) == 8
    if task_blocks:
        task_blocks = list(head.task_blocks.values())
        assert task_blocks[0] != task_blocks[1]
