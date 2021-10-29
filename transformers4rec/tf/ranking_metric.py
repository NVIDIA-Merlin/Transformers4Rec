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

# Adapted from source code: https://github.com/karlhigley/ranking-metrics-torch
from abc import abstractmethod

import tensorflow as tf
from keras import backend

from merlin_standard_lib import Registry

from .utils import tf_utils

ranking_metrics_registry = Registry("tf.ranking_metrics")

METRIC_PARAMETERS_DOCSTRING = """
    ks : tf.Tensor
        tensor of cutoffs.
    scores : tf.Tensor
        scores of predicted item-ids.
    labels : tf.Tensor
        true item-ids labels.
"""


class RankingMetric(tf.keras.metrics.Metric):
    """
    Metric wrapper for computing ranking metrics@K for session-based task.

    Parameters
    ----------
    top_ks : list, default [2, 5])
        list of cutoffs
    labels_onehot : bool
        Enable transform the encoded labels to one-hot representation
    """

    def __init__(self, name=None, dtype=None, top_ks=[2, 5], labels_onehot=False, **kwargs):
        super(RankingMetric, self).__init__(name=name, **kwargs)
        self.top_ks = top_ks
        self.labels_onehot = labels_onehot
        # Store the mean vector of the batch metrics (for each cut-off at topk) in ListWrapper
        self.metric_mean = []
        self.built = False

    def _build(self, shape):
        with tf.init_scope():
            # This should only be necessary for handling v1 behavior. In v2, AUC
            # should be initialized outside of any tf.functions, and therefore in
            # eager mode.
            if not tf.executing_eagerly():
                backend._initialize_variables(backend._get_session())

        if not self.built:
            self.accumulator = tf.Variable(
                tf.zeros(shape=[1, len(self.top_ks)]),
                shape=tf.TensorShape([None, tf.compat.v1.Dimension(len(self.top_ks))]),
            )

        bs = shape[0]
        variable_shape = [bs, tf.compat.v1.Dimension(len(self.top_ks))]
        self.accumulator.assign(tf.zeros(variable_shape))
        self.built = True

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs):
        # Computing the metrics at different cut-offs
        # init batch accumulator
        self._build(shape=tf.shape(y_pred))
        if self.labels_onehot:
            y_true = tf_utils.tranform_label_to_onehot(y_true, tf.shape(y_pred)[-1])

        metric = self._metric(
            tf.convert_to_tensor(self.top_ks),
            tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]]),
            y_true,
        )
        self.metric_mean.append(metric)

    def result(self):
        # Computing the mean of the batch metrics (for each cut-off at topk)
        return tf.reduce_mean(tf.concat(self.metric_mean, axis=0), axis=0)

    def reset_state(self):
        self.metric_mean = []

    @abstractmethod
    def _metric(self, ks: tf.Tensor, preds: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """
        Compute ranking metric over predictions and one-hot targets for different cut-offs.
        This method should be overridden by subclasses.

        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        raise NotImplementedError


@ranking_metrics_registry.register_with_multiple_names("precision_at", "precision")
class PrecisionAt(RankingMetric):
    def __init__(self, top_ks=None, labels_onehot=False):
        super(PrecisionAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)

    def _metric(self, ks: tf.Tensor, scores: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        Compute precision@K for each provided cutoff in ks

        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """

        ks, scores, labels = check_inputs(ks, scores, labels)
        _, _, topk_labels = tf_utils.extract_topk(ks, scores, labels)
        bs = tf.shape(scores)[0]

        for index in range(int(tf.shape(ks)[0])):
            k = ks[index]
            rows_ids = tf.range(bs, dtype=tf.int64)
            indices = tf.concat(
                [
                    tf.expand_dims(rows_ids, 1),
                    tf.cast(index, tf.int64) * tf.ones([bs, 1], dtype=tf.int64),
                ],
                axis=1,
            )

            self.accumulator.scatter_nd_update(
                indices=indices, updates=tf.reduce_sum(topk_labels[:, : int(k)], axis=1) / float(k)
            )

        return self.accumulator


@ranking_metrics_registry.register_with_multiple_names("recall_at", "recall")
class RecallAt(RankingMetric):
    def __init__(self, top_ks=None, labels_onehot=False):
        super(RecallAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)

    def _metric(self, ks: tf.Tensor, scores: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        Compute recall@K for each provided cutoff in ks

        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """

        ks, scores, labels = check_inputs(ks, scores, labels)
        _, _, topk_labels = tf_utils.extract_topk(ks, scores, labels)

        # Compute recalls at K
        num_relevant = tf.reduce_sum(labels, axis=-1)
        rel_indices = tf.where(num_relevant != 0)
        rel_count = tf.gather_nd(num_relevant, rel_indices)

        if tf.shape(rel_indices)[0] > 0:
            for index in range(int(tf.shape(ks)[0])):
                k = ks[index]
                rel_labels = tf.cast(
                    tf.gather_nd(topk_labels, rel_indices)[:, : int(k)], tf.float32
                )
                batch_recall_k = tf.cast(
                    tf.reshape(
                        tf.math.divide(tf.reduce_sum(rel_labels, axis=-1), rel_count),
                        (len(rel_indices), 1),
                    ),
                    tf.float32,
                )
                # Ensuring type is double, because it can be float if --fp16

                update_indices = tf.concat(
                    [
                        rel_indices,
                        tf.expand_dims(
                            tf.cast(index, tf.int64) * tf.ones(tf.shape(rel_indices)[0], tf.int64),
                            -1,
                        ),
                    ],
                    axis=1,
                )
                self.accumulator.scatter_nd_update(
                    indices=update_indices, updates=tf.reshape(batch_recall_k, (-1,))
                )

        return self.accumulator


@ranking_metrics_registry.register_with_multiple_names("avg_precision_at", "avg_precision", "map")
class AvgPrecisionAt(RankingMetric):
    def __init__(self, top_ks=[1], labels_onehot=False):
        super(AvgPrecisionAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)
        max_k = tf.reduce_max(self.top_ks)
        self.precision_at = PrecisionAt(top_ks=1 + tf.range(max_k))

    def _metric(self, ks: tf.Tensor, scores: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        Compute average precision @K for provided cutoff in ks

        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        ks, scores, labels = check_inputs(ks, scores, labels)
        topk_scores, _, topk_labels = tf_utils.extract_topk(ks, scores, labels)

        num_relevant = tf.reduce_sum(labels, axis=-1)
        max_k = tf.reduce_max(ks)

        bs = tf.shape(scores)[0]
        self.precision_at._build(shape=tf.shape(scores))
        precisions = self.precision_at._metric(1 + tf.range(max_k), topk_scores, topk_labels)
        rel_precisions = precisions * topk_labels

        for index in range(int(tf.shape(ks)[0])):
            k = ks[index]
            tf_total_prec = tf.reduce_sum(rel_precisions[:, :k], axis=1)
            clip_value = tf.clip_by_value(
                num_relevant, clip_value_min=1, clip_value_max=tf.cast(k, tf.float32)
            )

            rows_ids = tf.range(bs, dtype=tf.int64)
            indices = tf.concat(
                [
                    tf.expand_dims(rows_ids, 1),
                    tf.cast(index, tf.int64) * tf.ones([bs, 1], dtype=tf.int64),
                ],
                axis=1,
            )
            self.accumulator.scatter_nd_update(indices=indices, updates=tf_total_prec / clip_value)
            # Ensuring type is double, because it can be float if --fp16
        return self.accumulator


@ranking_metrics_registry.register_with_multiple_names("dcg_at", "dcg")
class DCGAt(RankingMetric):
    def __init__(self, top_ks=[1], labels_onehot=False):
        super(DCGAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)

    def _metric(
        self, ks: tf.Tensor, scores: tf.Tensor, labels: tf.Tensor, log_base: int = 2
    ) -> tf.Tensor:

        """
        Compute discounted cumulative gain @K for each provided cutoff in ks
        (ignoring ties)

        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        ks, scores, labels = check_inputs(ks, scores, labels)
        _, _, topk_labels = tf_utils.extract_topk(ks, scores, labels)

        # Compute discounts
        max_k = tf.reduce_max(ks)
        discount_positions = tf.cast(tf.range(max_k), tf.float32)
        discount_log_base = tf.math.log(tf.convert_to_tensor([log_base], dtype=tf.float32))

        discounts = 1 / (tf.math.log(discount_positions + 2) / discount_log_base)
        bs = tf.shape(scores)[0]
        # Compute DCGs at K
        for index in range(len(self.top_ks)):
            k = ks[index]
            m = topk_labels[:, :k] * tf.repeat(
                tf.expand_dims(discounts[:k], 0), tf.shape(topk_labels)[0], axis=0
            )
            rows_ids = tf.range(bs, dtype=tf.int64)
            indices = tf.concat(
                [
                    tf.expand_dims(rows_ids, 1),
                    tf.cast(index, tf.int64) * tf.ones([bs, 1], dtype=tf.int64),
                ],
                axis=1,
            )

            self.accumulator.scatter_nd_update(
                indices=indices, updates=tf.cast(tf.reduce_sum(m, axis=1), tf.float32)
            )
            # Ensuring type is double, because it can be float if --fp16

        return self.accumulator


@ranking_metrics_registry.register_with_multiple_names("ndcg_at", "ndcg")
class NDCGAt(RankingMetric):
    def __init__(self, top_ks=[1], labels_onehot=False):
        super(NDCGAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)
        self.dcg_at = DCGAt(top_ks)

    def _metric(
        self, ks: tf.Tensor, scores: tf.Tensor, labels: tf.Tensor, log_base: int = 2
    ) -> tf.Tensor:

        """
        Compute normalized discounted cumulative gain @K for each provided cutoffs in ks
        (ignoring ties)

        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        ks, scores, labels = check_inputs(ks, scores, labels)
        topk_scores, _, topk_labels = tf_utils.extract_topk(ks, scores, labels)

        # Compute discounted cumulative gains
        self.dcg_at._build(shape=tf.shape(scores))
        gains = self.dcg_at._metric(ks, topk_scores, topk_labels, log_base=log_base)
        # Re-init states of dcg_at accumulator
        self.dcg_at._build(shape=tf.shape(scores))
        normalizing_gains = self.dcg_at._metric(ks, topk_labels, topk_labels, log_base=log_base)

        # Prevent divisions by zero
        relevant_pos = tf.where(normalizing_gains != 0)
        tf.where(normalizing_gains == 0, 0.0, gains)

        updates = tf.gather_nd(gains, relevant_pos) / tf.gather_nd(normalizing_gains, relevant_pos)
        gains.scatter_nd_update(relevant_pos, updates)

        return gains


def check_inputs(ks, scores, labels):
    if len(ks.shape) > 1:
        raise ValueError("ks should be a 1-dimensional tensor")

    if len(scores.shape) != 2:
        raise ValueError("scores must be a 2-dimensional tensor")

    if len(labels.shape) != 2:
        raise ValueError("labels must be a 2-dimensional tensor")

    scores.get_shape().assert_is_compatible_with(labels.get_shape())

    return (tf.cast(ks, tf.int32), tf.cast(scores, tf.float32), tf.cast(labels, tf.float32))
