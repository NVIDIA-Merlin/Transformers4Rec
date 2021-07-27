from collections import defaultdict
from typing import Dict, List, Optional, Text

import tensorflow as tf

from ..types import ColumnGroup
from ..utils.columns import Tag


class TaskMixin:
    pass


class PredictionTask(TaskMixin, tf.keras.layers.Layer):
    def __init__(
        self,
        loss: Optional[tf.keras.losses.Loss] = None,
        pre: Optional[tf.keras.layers.Layer] = None,
        body: Optional[tf.keras.layers.Layer] = None,
        metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        name: Optional[Text] = None,
    ) -> None:
        """Initializes the task.
        Args:
          loss: Loss function. Defaults to BinaryCrossentropy.
          metrics: List of Keras metrics to be evaluated.
          prediction_metrics: List of Keras metrics used to summarize the
            predictions.
          label_metrics: List of Keras metrics used to summarize the labels.
          loss_metrics: List of Keras metrics used to summarize the loss.
          name: Optional task name.
        """

        super().__init__(name=name)

        self.body = body
        self.pre = pre

        self.loss = loss if loss is not None else tf.keras.losses.BinaryCrossentropy()
        self.eval_metrics = metrics or []
        self.prediction_metrics = prediction_metrics or []
        self.label_metrics = label_metrics or []
        self.loss_metrics = loss_metrics or []

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        if self.body:
            x = self.body(x)
        if self.pre:
            x = self.pre(x)

        return x

    def compute_loss(
        self,
        inputs,
        targets,
        training: bool = False,
        compute_metrics=True,
        sample_weight: Optional[tf.Tensor] = None,
        **kwargs
    ) -> tf.Tensor:
        predictions = self(inputs, training=training, **kwargs)
        loss = self.loss(y_true=targets, y_pred=predictions, sample_weight=sample_weight)

        if compute_metrics:
            update_ops = self.calculate_metrics(predictions, targets, forward=False, loss=loss)

            update_ops = [x for x in update_ops if x is not None]

            with tf.control_dependencies(update_ops):
                return tf.identity(loss)

        return loss

    def repr_add(self):
        return [("loss", self.loss)]

    def calculate_metrics(self, predictions, labels, sample_weight=None, forward=True, loss=None):
        if forward:
            predictions = self(predictions)

        update_ops = []

        for metric in self.eval_metrics:
            update_ops.append(
                metric.update_state(y_true=labels, y_pred=predictions, sample_weight=sample_weight)
            )

        for metric in self.prediction_metrics:
            update_ops.append(metric.update_state(predictions, sample_weight=sample_weight))

        for metric in self.label_metrics:
            update_ops.append(metric.update_state(labels, sample_weight=sample_weight))

        for metric in self.loss_metrics:
            if not loss:
                loss = self.loss(y_true=labels, y_pred=predictions, sample_weight=sample_weight)
            update_ops.append(metric.update_state(loss, sample_weight=sample_weight))

        return update_ops

    @classmethod
    def binary_classification(cls, metrics=None, add_logit_layer=True, name=None):

        metrics = metrics or [
            tf.keras.metrics.Precision(name=name_fn(name, "precision")),
            tf.keras.metrics.Recall(name=name_fn(name, "recall")),
            tf.keras.metrics.BinaryAccuracy(name=name_fn(name, "accuracy")),
            tf.keras.metrics.AUC(name=name_fn(name, "auc")),
        ]

        return cls(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics,
            pre=tf.keras.layers.Dense(1, activation="sigmoid", name=name_fn(name, "logit"))
            if add_logit_layer
            else None,
        )

    @classmethod
    def regression(cls, metrics=None, add_logit_layer=True, name=None):
        metrics = metrics or [tf.keras.metrics.RootMeanSquaredError(name=name_fn(name, "rmse"))]

        return cls(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=metrics,
            pre=tf.keras.layers.Dense(1, name=name_fn(name, "logit")) if add_logit_layer else None,
        )


class Head(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks = {}
        self._task_weights = defaultdict(lambda: 1)

    @classmethod
    def from_column_group(cls, column_group: ColumnGroup, add_logits=True, task_weights=None):
        if task_weights is None:
            task_weights = {}
        to_return = cls()

        for binary_target in column_group.select_by_tag(Tag.TARGETS_BINARY).column_names:
            to_return = to_return.add_binary_classification_task(
                binary_target,
                add_logit_layer=add_logits,
                task_weight=task_weights.get(binary_target, 1),
            )

        for regression_target in column_group.select_by_tag(Tag.TARGETS_REGRESSION).column_names:
            to_return = to_return.add_regression_task(
                regression_target,
                add_logit_layer=add_logits,
                task_weight=task_weights.get(regression_target, 1),
            )

        # TODO: Add multi-class classification here. Figure out how to get number of classes

        return to_return

    def add_task(self, target_name, task: PredictionTask, task_weight=1):
        self.tasks[target_name] = task
        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def add_binary_classification_task(self, target_name, add_logit_layer=True, task_weight=1):
        self.tasks[target_name] = PredictionTask.binary_classification(
            add_logit_layer=add_logit_layer, name=target_name
        )
        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def add_regression_task(self, target_name, add_logit_layer=True, task_weight=1):
        self.tasks[target_name] = PredictionTask.regression(
            add_logit_layer=add_logit_layer, name=target_name
        )
        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def pop_labels(self, inputs: Dict[Text, tf.Tensor]):
        outputs = {}
        for name in self.tasks.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def call(self, logits: tf.Tensor, **kwargs):
        outputs = {}

        for name, task in self.tasks.items():
            outputs[name] = task(logits, **kwargs)

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(self, block_outputs, targets, training=False, **kwargs) -> tf.Tensor:
        losses = []

        for name, task in self.tasks.items():
            task_target = targets[name] if isinstance(targets, dict) else targets
            loss = task.compute_loss(block_outputs, task_target, training=training, **kwargs)
            losses.append(loss * self._task_weights[name])

        return tf.reduce_sum(losses)


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None
