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


# TODO: Fix this test when `run_eagerly=False`
# @pytest.mark.parametrize("run_eagerly", [True, False])
def test_simple_model(tf_tabular_features, tf_tabular_data, run_eagerly=True):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    inputs = tf_tabular_features
    body = tr.SequentialBlock([inputs, tr.MLPBlock([64])])
    model = tr.BinaryClassificationTask("target").to_model(body, inputs)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    dataset = tf.data.Dataset.from_tensor_slices((tf_tabular_data, targets)).batch(50)

    losses = model.fit(dataset, epochs=5)
    metrics = model.evaluate(tf_tabular_data, targets, return_dict=True)

    assert len(metrics.keys()) == 7
    assert len(losses.epoch) == 5
    assert all(0 <= loss <= 1 for loss in losses.history["loss"])


@pytest.mark.parametrize("prediction_task", [tr.BinaryClassificationTask, tr.RegressionTask])
def test_serialization_model(tf_tabular_features, tf_tabular_data, prediction_task):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = tr.SequentialBlock([tf_tabular_features, tr.MLPBlock([64])])
    task = prediction_task("target")
    model = task.to_model(body, tf_tabular_features)

    copy_model = test_utils.assert_serialization(model)
    test_utils.assert_loss_and_metrics_are_valid(copy_model, tf_tabular_data, targets)
