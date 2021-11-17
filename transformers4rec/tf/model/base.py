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
import abc
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Text, Type, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import generic_utils
from transformers import TFSequenceSummary

from merlin_standard_lib import Schema, Tag

from ..features.sequence import TabularFeaturesType
from ..ranking_metric import process_metrics
from ..utils.tf_utils import (
    LossMixin,
    MetricsMixin,
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None


MetricOrMetricClass = Union[tf.keras.metrics.Metric, Type[tf.keras.metrics.Metric]]


class PredictionTask(Layer, LossMixin, MetricsMixin):
    def __init__(
        self,
        loss: tf.keras.losses.Loss,
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
        **kwargs,
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

        super().__init__(name=name, **kwargs)
        self.target_name = target_name
        self.sequence_summary = TFSequenceSummary(
            SimpleNamespace(summary_type=summary_type)  # type: ignore
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

    def compute_loss(  # type: ignore
        self,
        inputs,
        targets,
        training: bool = False,
        call_task: bool = True,
        compute_metrics=True,
        sample_weight: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> tf.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]
        predictions = inputs
        if call_task:
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
            metric.reset_state()

    def to_head(self, body, inputs=None, **kwargs) -> "Head":
        return Head(body, self, inputs=inputs, **kwargs)

    def to_model(self, body, inputs=None, **kwargs) -> "Model":
        return Model(Head(body, self, inputs=inputs, **kwargs), **kwargs)

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(
            config,
            {
                "pre": tf.keras.layers.deserialize,
                "loss": tf.keras.losses.deserialize,
                "metrics": tf.keras.metrics.deserialize,
                "prediction_metrics": tf.keras.metrics.deserialize,
                "label_metrics": tf.keras.metrics.deserialize,
                "loss_metrics": tf.keras.metrics.deserialize,
            },
        )

        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self,
            config,
            ["metrics", "prediction_metrics", "label_metrics", "loss_metrics", "loss", "pre"],
        )

        config["summary_type"] = self.sequence_summary.summary_type
        if self.target_name:
            config["target_name"] = self.target_name
        if self._task_name:
            config["task_name"] = self._task_name

        return config


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class Head(tf.keras.layers.Layer):
    def __init__(
        self,
        body: tf.keras.layers.Layer,
        prediction_tasks: Union[List[PredictionTask], PredictionTask],
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weights: Optional[List[float]] = None,
        loss_reduction=tf.reduce_mean,
        inputs: Optional[TabularFeaturesType] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.body = body
        self.inputs = inputs
        self.loss_reduction = loss_reduction

        self.prediction_tasks = prediction_tasks
        self.task_weights = task_weights

        self.prediction_task_dict = {}
        if prediction_tasks:
            if not isinstance(prediction_tasks, list):
                prediction_tasks = [prediction_tasks]
            for task in prediction_tasks:
                self.prediction_task_dict[task.task_name] = task

        self._task_weight_dict = defaultdict(lambda: 1.0)
        if task_weights:
            for task, val in zip(prediction_tasks, task_weights):
                self._task_weight_dict[task.task_name] = val

        self._set_task_blocks(task_blocks)

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        body: Layer,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weight_dict: Optional[Dict[str, float]] = None,
        loss_reduction=tf.reduce_mean,
        inputs: Optional[TabularFeaturesType] = None,
        **kwargs,
    ) -> "Head":
        task_weight_dict = task_weight_dict or {}

        tasks: List[PredictionTask] = []
        task_weights = []

        from .prediction_task import BinaryClassificationTask, RegressionTask

        for binary_target in schema.select_by_tag(Tag.BINARY_CLASSIFICATION).column_names:
            tasks.append(BinaryClassificationTask(binary_target))
            task_weights.append(task_weight_dict.get(binary_target, 1.0))

        for regression_target in schema.select_by_tag(Tag.REGRESSION).column_names:
            tasks.append(RegressionTask(regression_target))
            task_weights.append(task_weight_dict.get(regression_target, 1.0))

        # TODO: Add multi-class classification here. Figure out how to get number of classes

        return cls(
            body,
            tasks,
            task_blocks=task_blocks,
            task_weights=task_weights,
            loss_reduction=loss_reduction,
            inputs=inputs,
            **kwargs,
        )

    def _set_task_blocks(self, task_blocks):
        if not task_blocks:
            return

        if isinstance(task_blocks, dict):
            tasks_multi_names = self._prediction_tasks_multi_names()
            for key, task_block in task_blocks.items():
                if key in tasks_multi_names:
                    tasks = tasks_multi_names[key]
                    if len(tasks) == 1:
                        self.prediction_task_dict[tasks[0].task_name].task_block = task_block
                    else:
                        raise ValueError(
                            f"Ambiguous name: {key}, can't resolve it to a task "
                            "because there are multiple tasks that contain the key: "
                            f"{', '.join([task.task_name for task in tasks])}"
                        )
                else:
                    raise ValueError(
                        f"Couldn't find {key} in prediction_tasks, "
                        f"only found: {', '.join(list(self.prediction_task_dict.keys()))}"
                    )
        elif isinstance(task_blocks, Layer):
            for key, val in self.prediction_task_dict.items():
                task_block = task_blocks.from_config(task_blocks.get_config())
                val.task_block = task_block
        else:
            raise ValueError("`task_blocks` must be a Layer or a Dict[str, Layer]")

    def _prediction_tasks_multi_names(self) -> Dict[str, List[PredictionTask]]:
        prediction_tasks_multi_names = {
            name: [val] for name, val in self.prediction_task_dict.items()
        }
        for name, value in self.prediction_task_dict.items():
            name_parts = name.split("/")
            for name_part in name_parts:
                if name_part in prediction_tasks_multi_names:
                    prediction_tasks_multi_names[name_part].append(value)
                else:
                    prediction_tasks_multi_names[name_part] = [value]

        return prediction_tasks_multi_names

    def add_task(self, task: PredictionTask, task_weight=1):
        key = task.target_name
        self.prediction_task_dict[key] = task
        if task_weight:
            self._task_weight_dict[key] = task_weight

        return self

    def pop_labels(self, inputs: Dict[Text, tf.Tensor]):
        outputs = {}
        for name in self.prediction_task_dict.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def build(self, input_shape):
        from .prediction_task import NextItemPredictionTask

        self.body.build(input_shape)
        input_shape = self.body.compute_output_shape(input_shape)

        for task in self.prediction_task_dict.values():
            if isinstance(task, NextItemPredictionTask):
                task.build(input_shape, self.body, inputs=self.inputs)
        return super().build(input_shape)

    def call(self, body_outputs: tf.Tensor, call_body=True, always_output_dict=False, **kwargs):
        outputs = {}

        if call_body:
            body_outputs = self.body(body_outputs)

        for name, task in self.prediction_task_dict.items():
            outputs[name] = task(body_outputs, **kwargs)

        if len(outputs) == 1 and not always_output_dict:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(
        self, body_outputs, targets, training=False, call_body=True, compute_metrics=True, **kwargs
    ) -> tf.Tensor:
        losses = []

        predictions = self(body_outputs, call_body=call_body, always_output_dict=True)
        for name, task in self.prediction_task_dict.items():
            loss = task.compute_loss(
                predictions[name],
                targets,
                call_task=False,
                training=training,
                compute_metrics=compute_metrics,
                **kwargs,
            )
            losses.append(loss * self._task_weight_dict[name])

        return self.loss_reduction(losses)

    def metric_results(self, mode=None):
        def name_fn(x):
            return "_".join([mode, x]) if mode else x

        metrics = {
            name_fn(name): task.metric_results() for name, task in self.prediction_task_dict.items()
        }

        return _output_metrics(metrics)

    def reset_metrics(self):
        for task in self.prediction_task_dict.values():
            task.reset_metrics()

    @property
    def task_blocks(self) -> Dict[str, Optional[Layer]]:
        return {name: task.task_block for name, task in self.prediction_task_dict.items()}

    @property
    def metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        outputs = {}
        for name, task in self.prediction_task_dict.items():
            outputs.update({metric.name: metric for metric in task.metrics})

        return outputs

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(
            config, ["body", "prediction_tasks", "task_weights"]
        )

        config["loss_reduction"] = getattr(tf, config["loss_reduction"])

        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self, config, ["body", "loss_reduction", "prediction_tasks"]
        )
        if self.task_weights:
            config["task_weights"] = self.task_weights

        return config


class BaseModel(tf.keras.Model, LossMixin, abc.ABC):
    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            inputs, targets = inputs
            loss = self.compute_loss(inputs, targets, training=True)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = process_metrics(self.metrics, prefix="train_")
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        loss = self.compute_loss(*inputs, training=False)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = process_metrics(self.metrics, prefix="eval_")

        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class Model(BaseModel):
    def __init__(
        self, *head: Head, head_weights: Optional[List[float]] = None, name=None, **kwargs
    ):
        if head_weights:
            if not isinstance(head_weights, (list, tuple)):
                raise ValueError("`head_weights` must be a list or tuple")
            if not len(head_weights) == len(head):
                raise ValueError(
                    "`head_weights` needs to have the same length " "as the number of heads"
                )

        super().__init__(name=name, **kwargs)

        self.heads = head
        self.head_weights = tuple(head_weights or [1.0] * len(head))

    def call(self, inputs, **kwargs):
        # TODO: Optimize this
        outputs = {}
        for head in self.heads:
            outputs.update(head(inputs, call_body=True, always_output_dict=True))

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(  # type: ignore
        self, inputs, targets, training: bool = False, compute_metrics=True, **kwargs
    ) -> tf.Tensor:
        losses = tuple(
            [
                head.compute_loss(
                    inputs,
                    targets,
                    call_body=kwargs.pop("call_body", True),
                    compute_metrics=compute_metrics,
                    **kwargs,
                )
                for head in self.heads
            ]
        )
        with ops.name_scope("merge_losses", values=losses + self.head_weights):
            weighted_losses = []
            for loss, head_weight in zip(losses, self.head_weights):
                weighted_losses.append(tf.math.multiply(loss, head_weight))

            return tf.add_n(weighted_losses)

    def metric_results(self, mode=None):
        outputs = []

        for head in self.heads:
            outputs.append(head.metric_results(mode=mode))

        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    @classmethod
    def from_config(cls, config, custom_objects=None):
        heads = [tf.keras.utils.deserialize_keras_object(h) for h in config.pop("heads")]

        return cls(*heads, **config)

    def get_config(self):
        return {
            "head_weights": self.head_weights,
            "heads": [tf.keras.utils.serialize_keras_object(h) for h in self.heads],
        }


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics
