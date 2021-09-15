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
from typing import Dict, List, Optional, Text, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin_standard_lib import Schema, Tag

from ..typing import TabularFeaturesType
from ..utils.tf_utils import maybe_deserialize_keras_objects, maybe_serialize_keras_objects
from .prediction_task import BinaryClassificationTask, PredictionTask, RegressionTask


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
        tasks, task_weights = [], []

        for binary_target in schema.select_by_tag(Tag.TARGETS_BINARY).column_names:
            tasks.append(BinaryClassificationTask(binary_target))
            task_weights.append(task_weight_dict.get(binary_target, 1.0))

        for regression_target in schema.select_by_tag(Tag.TARGETS_REGRESSION).column_names:
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
        return super().build(input_shape)

    def call(self, body_outputs: tf.Tensor, call_body=False, always_output_dict=False, **kwargs):
        outputs = {}

        if call_body:
            body_outputs = self.body(body_outputs)

        for name, task in self.prediction_task_dict.items():
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

        for name, task in self.prediction_task_dict.items():
            loss = task.compute_loss(body_outputs, targets, training=training, **kwargs)
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
            self, config, ["body", "loss_reduction", "prediction_tasks", "task_weights"]
        )

        return config


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics
