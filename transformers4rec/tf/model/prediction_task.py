from types import SimpleNamespace
from typing import List, Optional, Text, Type, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import generic_utils
from transformers.modeling_tf_utils import TFSequenceSummary

from ..typing import Head, Model
from ..utils.tf_utils import LossMixin, MetricsMixin, maybe_serialize_keras_objects


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None


MetricOrMetricClass = Union[tf.keras.metrics.Metric, Type[tf.keras.metrics.Metric]]


class PredictionTask(Layer, LossMixin, MetricsMixin):
    def __init__(
        self,
        loss: Optional[tf.keras.losses.Loss],
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        metrics: Optional[List[MetricOrMetricClass]] = None,
        pre: Optional[Layer] = None,
        task_block: Optional[Layer] = None,
        prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        name: Optional[Text] = None,
        summary_type="last",
    ) -> None:
        """Initializes the task.

        Parameters
        ----------
        loss:
            Loss function. Defaults to BinaryCrossentropy.
        metrics:
            List of Keras metrics to be evaluated.
        prediction_metrics:
            List of Keras metrics used to summarize the predictions.
        label_metrics:
            List of Keras metrics used to summarize the labels.
        loss_metrics:
            List of Keras metrics used to summarize the loss.
        name:
            Optional task name.
        """

        super().__init__(name=name)
        self.target_name = target_name
        self.sequence_summary = TFSequenceSummary(
            SimpleNamespace(summary_type=summary_type)
        )  # noqa
        self.pre = pre
        self.task_block = task_block
        self.loss = loss
        self._task_name = task_name

        create_metrics = self._create_metrics
        self.eval_metrics = create_metrics(metrics) if metrics else []
        self.prediction_metrics = create_metrics(prediction_metrics) if prediction_metrics else []
        self.label_metrics = create_metrics(label_metrics) if label_metrics else []
        self.loss_metrics = create_metrics(loss_metrics) if loss_metrics else []

    def call(self, inputs, training=False, **kwargs):
        x = inputs

        if len(x.shape) == 3:
            x = self.sequence_summary(x)

        if self.task_block:
            x = self.task_block(x)

        if self.pre:
            x = self.pre(x)

        return x

    def _create_metrics(self, metrics: List[MetricOrMetricClass]) -> List[tf.keras.metrics.Metric]:
        outputs = []
        for metric in metrics:
            if not isinstance(metric, tf.keras.metrics.Metric):
                metric = metric(name=self.child_name(generic_utils.to_snake_case(metric.__name__)))
            outputs.append(metric)

        return outputs

    @property
    def task_name(self):
        if self._task_name:
            return self._task_name

        base_name = generic_utils.to_snake_case(self.__class__.__name__)

        return name_fn(self.target_name, base_name) if self.target_name else base_name

    def child_name(self, name):
        return name_fn(self.task_name, name)

    def compute_loss(
        self,
        inputs,
        targets,
        training: bool = False,
        compute_metrics=True,
        sample_weight: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> tf.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

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

    def calculate_metrics(self, predictions, targets, sample_weight=None, forward=True, loss=None):
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        if forward:
            predictions = self(predictions)

        update_ops = []

        for metric in self.eval_metrics:
            update_ops.append(
                metric.update_state(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
            )

        for metric in self.prediction_metrics:
            update_ops.append(metric.update_state(predictions, sample_weight=sample_weight))

        for metric in self.label_metrics:
            update_ops.append(metric.update_state(targets, sample_weight=sample_weight))

        for metric in self.loss_metrics:
            if not loss:
                loss = self.loss(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
            update_ops.append(metric.update_state(loss, sample_weight=sample_weight))

        return update_ops

    def metric_results(self, mode: str = None):
        return {metric.name: metric.result() for metric in self.metrics}

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def to_head(self, body, inputs=None, **kwargs) -> Head:
        from .head import Head as _Head

        return _Head(body, self, inputs=inputs, **kwargs)

    def to_model(self, body, inputs=None, **kwargs) -> Model:
        from .head import Head as _Head
        from .model import Model as _Model

        return _Model(_Head(body, self, inputs=inputs, **kwargs), **kwargs)

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, ["loss", "pre"])
        config = maybe_serialize_keras_objects(
            self,
            config,
            ["metrics", "prediction_metrics", "label_metrics", "loss_metrics"],
        )

        config["summary_type"] = self.sequence_summary.summary_type
        if self.target_name:
            config["target_name"] = self.target_name
        if self._task_name:
            config["task_name"] = self._task_name

        return config


class BinaryClassificationTask(PredictionTask):
    DEFAULT_LOSS = tf.keras.losses.BinaryCrossentropy()
    DEFAULT_METRICS = (
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.AUC,
    )

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics: List[MetricOrMetricClass] = DEFAULT_METRICS,
        summary_type="first",
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
        )
        self.pre = tf.keras.layers.Dense(1, activation="sigmoid", name=self.child_name("logit"))


class RegressionTask(PredictionTask):
    DEFAULT_LOSS = tf.keras.losses.MeanSquaredError()
    DEFAULT_METRICS = (tf.keras.metrics.RootMeanSquaredError,)

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        summary_type="first",
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
        )
        self.pre = tf.keras.layers.Dense(1, name=self.child_name("logit"))
