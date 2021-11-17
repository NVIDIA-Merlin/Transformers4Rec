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
tfr = pytest.importorskip("tensorflow_ranking")
test_utils = pytest.importorskip("transformers4rec.tf.utils.testing_utils")

# fixed parameters for tests
list_metrics = list(tr.ranking_metric.ranking_metrics_registry.keys())


# Test length of output equal to number of cutoffs
@pytest.mark.parametrize("metric", list_metrics)
def test_metrics_shape(tf_ranking_metrics_inputs, metric):
    metric = tr.ranking_metric.ranking_metrics_registry[metric](
        top_ks=tf_ranking_metrics_inputs["ks"]
    )
    result = metric(
        y_pred=tf_ranking_metrics_inputs["scores"],
        y_true=tf_ranking_metrics_inputs["labels_one_hot"],
    )
    assert tf.shape(result)[0] == len(tf_ranking_metrics_inputs["ks"])


# Test label one hot encoding
@pytest.mark.parametrize("metric", list_metrics)
def test_score_with_transform_onehot(tf_ranking_metrics_inputs, metric):
    metric = tr.ranking_metric.ranking_metrics_registry[metric](
        top_ks=tf_ranking_metrics_inputs["ks"], labels_onehot=True
    )
    result = metric(
        y_pred=tf_ranking_metrics_inputs["scores"], y_true=tf_ranking_metrics_inputs["labels"]
    )
    assert len(result) == len(tf_ranking_metrics_inputs["ks"])


@pytest.mark.parametrize("metric", list_metrics)
def test_score_different_from_zero(tf_ranking_metrics_inputs, metric):
    metric = tr.ranking_metric.ranking_metrics_registry[metric](top_ks=[20], labels_onehot=True)
    result = metric(
        y_pred=tf_ranking_metrics_inputs["scores"], y_true=tf_ranking_metrics_inputs["labels"]
    )
    assert all(e > 0 for e in result)


@pytest.mark.parametrize("metric", list_metrics)
def test_score_different_between_thresholds(tf_ranking_metrics_inputs, metric):
    metric = tr.ranking_metric.ranking_metrics_registry[metric](top_ks=[5, 20], labels_onehot=True)
    result = metric(
        y_pred=tf_ranking_metrics_inputs["scores"], y_true=tf_ranking_metrics_inputs["labels"]
    )
    assert result[-1] != result[0]


# compare implemented metrics w.r.t tensorflow_ranking
metrics_to_compare = [
    (tfr.keras.metrics.RecallMetric, "recall"),
    (tfr.keras.metrics.PrecisionMetric, "precision"),
    (tfr.keras.metrics.DCGMetric, "dcg"),
]


@pytest.mark.parametrize("metric", metrics_to_compare)
def test_compare_with_tfr(tf_ranking_metrics_inputs, metric):
    np = pytest.importorskip("numpy")
    tfr_metric = [metric[0](topn=topn) for topn in tf_ranking_metrics_inputs["ks"]]
    results_tfr = [
        metric(
            y_pred=tf_ranking_metrics_inputs["scores"],
            y_true=tf.cast(tf_ranking_metrics_inputs["labels_one_hot"], tf.float32),
        ).numpy()
        for metric in tfr_metric
    ]

    tr_metric = tr.ranking_metric.ranking_metrics_registry[metric[1]](
        top_ks=tf_ranking_metrics_inputs["ks"]
    )

    results_tr = tr_metric(
        y_pred=tf_ranking_metrics_inputs["scores"],
        y_true=tf_ranking_metrics_inputs["labels_one_hot"],
    ).numpy()

    assert np.allclose(results_tr, np.array(results_tfr), rtol=1e-04, atol=1e-08)


# Test serialization of ranking metrics
@pytest.mark.parametrize("metric", list_metrics)
def test_serialization_ranking_metric(tf_ranking_metrics_inputs, metric):
    metric = tr.ranking_metric.ranking_metrics_registry[metric](
        top_ks=tf_ranking_metrics_inputs["ks"]
    )
    copy_metric = test_utils.assert_serialization(metric)

    result = copy_metric(
        y_pred=tf_ranking_metrics_inputs["scores"],
        y_true=tf_ranking_metrics_inputs["labels_one_hot"],
    )

    assert all(copy_metric.top_ks == metric.top_ks)
    assert tf.shape(result)[0] == len(tf_ranking_metrics_inputs["ks"])
