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

import platform

import pytest

tf = pytest.importorskip("tensorflow")
tr = pytest.importorskip("transformers4rec.tf")


def mark_run_eagerly_modes(*args, **kwargs):
    modes = [True, False]

    # As of TF 2.5 there's a bug that our EmbeddingFeatures don't work on M1 Macs
    if "macOS" in platform.platform() and "arm64-arm-64bit" in platform.platform():
        modes = [True]

    return pytest.mark.parametrize("run_eagerly", modes)(*args, **kwargs)


def assert_body_works_in_model(data, inputs, body, run_eagerly):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    model = tr.BinaryClassificationTask("target").to_model(body, inputs)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    dataset = tf.data.Dataset.from_tensor_slices((data, targets)).batch(50)

    losses = model.fit(dataset, epochs=5)
    metrics = model.evaluate(data, targets, return_dict=True)

    assert len(metrics.keys()) == 7
    assert len(losses.epoch) == 5
    assert len(losses.history["loss"]) == 5


def assert_loss_and_metrics_are_valid(input, inputs, targets, call_body=True):
    loss = input.compute_loss(inputs, targets, call_body=call_body)
    metrics = input.metric_results()

    assert loss is not None
    assert len(metrics) == len(input.metrics)


def assert_serialization(layer):
    copy_layer = layer.from_config(layer.get_config())

    assert isinstance(copy_layer, layer.__class__)

    return copy_layer
