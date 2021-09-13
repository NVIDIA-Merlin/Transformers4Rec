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

from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Text, Type, Union

import tensorflow as tf
from tensorflow.python.keras.utils import generic_utils
from transformers.modeling_tf_utils import TFSequenceSummary

from merlin_standard_lib import Schema, Tag


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None


class TaskMixin:
    pass


Layer = tf.keras.layers.Layer
MetricOrMetricClass = Union[tf.keras.metrics.Metric, Type[tf.keras.metrics.Metric]]


class PredictionTask(TaskMixin, Layer):
    def __init__(
        self,
        loss: Optional[tf.keras.losses.Loss],
        target_name=None,
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

    def metric_results(self):
        return {metric.name: metric.result() for metric in self.metrics}

    def to_head(self, body, inputs=None, **kwargs) -> "Head":
        return Head(body, self, inputs=inputs, **kwargs)

    def to_model(self, body, inputs=None, **kwargs):
        from .model import Model

        return Model(Head(body, self, inputs=inputs, **kwargs), **kwargs)


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
        target_name=None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics: List[MetricOrMetricClass] = DEFAULT_METRICS,
        summary_type="first",
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            summary_type=summary_type,
            task_block=task_block,
        )
        self.pre = tf.keras.layers.Dense(1, activation="sigmoid", name=self.child_name("logit"))


class RegressionTask(PredictionTask):
    DEFAULT_LOSS = tf.keras.losses.MeanSquaredError()
    DEFAULT_METRICS = (tf.keras.metrics.RootMeanSquaredError,)

    def __init__(
        self,
        target_name=None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        summary_type="first",
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            summary_type=summary_type,
            task_block=task_block,
        )
        self.pre = tf.keras.layers.Dense(1, name=self.child_name("logit"))


class Head(tf.keras.layers.Layer):
    def __init__(
        self,
        body: tf.keras.layers.Layer,
        prediction_tasks: Optional[Union[List[PredictionTask], PredictionTask]] = None,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weights=None,
        loss_reduction=tf.reduce_mean,
        inputs=None,
    ):
        super().__init__()
        self.body = body
        self.loss_reduction = loss_reduction
        self.prediction_tasks = {}
        if prediction_tasks:
            if not isinstance(prediction_tasks, list):
                prediction_tasks = [prediction_tasks]
            for task in prediction_tasks:
                self.prediction_tasks[task.task_name] = task

        self._task_weights = defaultdict(lambda: 1)
        if task_weights:
            for key, val in task_weights.items():
                self._task_weights[key] = val

        self._set_task_blocks(task_blocks)

    @classmethod
    def from_schema(cls, schema: Schema, body, task_weights=None):
        if task_weights is None:
            task_weights = {}
        to_return = cls(body)

        for binary_target in schema.select_by_tag(Tag.TARGETS_BINARY).column_names:
            to_return = to_return.add_task(
                BinaryClassificationTask(binary_target),
                task_weight=task_weights.get(binary_target, 1),
            )

        for regression_target in schema.select_by_tag(Tag.TARGETS_REGRESSION).column_names:
            to_return = to_return.add_task(
                RegressionTask(regression_target),
                task_weight=task_weights.get(regression_target, 1),
            )

        # TODO: Add multi-class classification here. Figure out how to get number of classes

        return to_return

    def _set_task_blocks(self, task_blocks):
        if not task_blocks:
            return

        if isinstance(task_blocks, dict):
            tasks_multi_names = self._prediction_tasks_multi_names()
            for key, task_block in task_blocks.items():
                if key in tasks_multi_names:
                    tasks = tasks_multi_names[key]
                    if len(tasks) == 1:
                        self.prediction_tasks[tasks[0].task_name].task_block = task_block
                    else:
                        raise ValueError(
                            f"Ambiguous name: {key}, can't resolve it to a task "
                            "because there are multiple tasks that contain the key: "
                            f"{', '.join([task.task_name for task in tasks])}"
                        )
                else:
                    raise ValueError(
                        f"Couldn't find {key} in prediction_tasks, "
                        f"only found: {', '.join(list(self.prediction_tasks.keys()))}"
                    )
        elif isinstance(task_blocks, Layer):
            for key, val in self.prediction_tasks.items():
                task_block = task_blocks.from_config(task_blocks.get_config())
                val.task_block = task_block
        else:
            raise ValueError("`task_blocks` must be a Layer or a Dict[str, Layer]")

    def _prediction_tasks_multi_names(self) -> Dict[str, List[PredictionTask]]:
        prediction_tasks_multi_names = {name: [val] for name, val in self.prediction_tasks.items()}
        for name, value in self.prediction_tasks.items():
            name_parts = name.split("/")
            for name_part in name_parts:
                if name_part in prediction_tasks_multi_names:
                    prediction_tasks_multi_names[name_part].append(value)
                else:
                    prediction_tasks_multi_names[name_part] = [value]

        return prediction_tasks_multi_names

    def add_task(self, task: PredictionTask, task_weight=1):
        key = task.target_name
        self.prediction_tasks[key] = task
        if task_weight:
            self._task_weights[key] = task_weight

        return self

    def pop_labels(self, inputs: Dict[Text, tf.Tensor]):
        outputs = {}
        for name in self.prediction_tasks.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, body_outputs: tf.Tensor, call_body=False, always_output_dict=False, **kwargs):
        outputs = {}

        if call_body:
            body_outputs = self.body(body_outputs)

        for name, task in self.prediction_tasks.items():
            outputs[name] = task(body_outputs, **kwargs)

        if len(outputs) == 1 and not always_output_dict:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(
        self, body_outputs, targets, training=False, call_body=False, **kwargs
    ) -> tf.Tensor:
        losses = []

        if call_body:
            body_outputs = self.body(body_outputs)

        for name, task in self.prediction_tasks.items():
            loss = task.compute_loss(body_outputs, targets, training=training, **kwargs)
            losses.append(loss * self._task_weights[name])

        return self.loss_reduction(losses)

    def metric_results(self, mode=None):
        def name_fn(x):
            return "_".join([mode, x]) if mode else x

        metrics = {
            name_fn(name): task.metric_results() for name, task in self.prediction_tasks.items()
        }

        return _output_metrics(metrics)

    def reset_metrics(self):
        for task in self.prediction_tasks.values():
            task.reset_metrics()

    @property
    def task_blocks(self) -> Dict[str, Optional[Layer]]:
        return {name: task.task_block for name, task in self.prediction_tasks.items()}

    @property
    def metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        outputs = {}
        for name, task in self.prediction_tasks.items():
            outputs.update({metric.name: metric for metric in task.metrics})

        return outputs


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics
