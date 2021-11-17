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
import tempfile

import pytest

from transformers4rec.config import transformer as tconf

tf = pytest.importorskip("tensorflow")
tr = pytest.importorskip("transformers4rec.tf")
test_utils = pytest.importorskip("transformers4rec.tf.utils.testing_utils")


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_simple_model(tf_tabular_features, tf_tabular_data, run_eagerly):
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


config_classes = [
    tconf.XLNetConfig,
    tconf.LongformerConfig,
    tconf.GPT2Config,
    tconf.BertConfig,
    tconf.RobertaConfig,
    tconf.AlbertConfig,
]


@pytest.mark.parametrize("config_cls", config_classes)
def test_to_tf_model_from_config(yoochoose_schema, tf_yoochoose_like, config_cls):
    transformer_config = config_cls.build(128, 4, 2, 20)

    input_module = tr.TabularSequenceFeatures.from_schema(
        yoochoose_schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=128,
        masking="causal",
    )
    task = tr.BinaryClassificationTask("classification")
    model = transformer_config.to_tf_model(input_module, task)

    out = model(tf_yoochoose_like)

    assert tf.shape(out)[0] == 100


@pytest.mark.parametrize("d_model", [32, 64, 128])
def test_with_d_model_different_from_item_dim(
    tf_yoochoose_like, tf_next_item_prediction_model, d_model
):

    model = tf_next_item_prediction_model(d_output=d_model)

    assert model(tf_yoochoose_like).shape[-1] == 51997


@pytest.mark.parametrize("masking", ["causal", "mlm"])
def test_output_shape_mode_eval(tf_yoochoose_like, tf_next_item_prediction_model, masking):

    model = tf_next_item_prediction_model(masking=masking)

    out = model(tf_yoochoose_like, training=False)
    assert out.shape[0] == tf_yoochoose_like["item_id/list"].shape[0]


@pytest.mark.parametrize("masking", ["causal", "mlm"])
@pytest.mark.parametrize("run_eagerly", [True, False])
def test_next_item_fit_model(
    tf_yoochoose_like, tf_next_item_prediction_model, masking, run_eagerly
):

    model = tf_next_item_prediction_model(masking=masking)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf_yoochoose_like, tf_yoochoose_like["item_id/list"])
    ).batch(50)
    losses = model.fit(dataset, epochs=5)
    metrics = model.evaluate(tf_yoochoose_like, tf_yoochoose_like["item_id/list"], return_dict=True)

    assert len(metrics.keys()) == 9
    assert len(losses.epoch) == 5
    assert all(loss >= 0 for loss in losses.history["loss"])
    assert losses.history["loss"][-1] < losses.history["loss"][0]


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("config", config_classes)
def test_save_and_load_transformer_model(
    tf_next_item_prediction_model,
    tf_yoochoose_like,
    config,
    run_eagerly,
):
    model = tf_next_item_prediction_model(config=config)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf_yoochoose_like, tf_yoochoose_like["item_id/list"])
    ).batch(50)

    inputs = next(iter(dataset))[0]
    model._set_inputs(inputs)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        model = tf.keras.models.load_model(tmpdir)

    assert tf.shape(model(inputs))[1] == 51997
