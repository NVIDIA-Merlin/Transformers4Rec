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
from typing import Dict, List, Optional, Union

import torch

from merlin_standard_lib import Schema, Tag

from ..typing import BlockOrModule, BlockType, Model, TabularData, TabularFeaturesType
from ..utils.torch_utils import LossMixin, MetricsMixin
from .prediction_task import BinaryClassificationTask, PredictionTask, RegressionTask


class Head(torch.nn.Module, LossMixin, MetricsMixin):
    """Head of a Model, a head has a single body but could have multiple prediction-tasks.

    Parameters
    ----------
    body: Block
        TODO
    prediction_tasks: Union[List[PredictionTask], PredictionTask], optional
        TODO
    task_blocks
        TODO
    task_weights: List[float], optional
        TODO
    loss_reduction: str, default="mean"
        TODO
    inputs: TabularFeaturesType, optional
        TODO
    """

    def __init__(
        self,
        body: BlockType,
        prediction_tasks: Union[List[PredictionTask], PredictionTask],
        task_blocks: Optional[Union[BlockType, Dict[str, BlockType]]] = None,
        task_weights: Optional[List[float]] = None,
        loss_reduction: str = "mean",
        inputs: Optional[TabularFeaturesType] = None,
    ):
        super().__init__()
        self.body = body
        self.loss_reduction = loss_reduction
        self.prediction_tasks = torch.nn.ModuleDict()
        if prediction_tasks:
            if not isinstance(prediction_tasks, list):
                prediction_tasks = [prediction_tasks]
            for i, task in enumerate(prediction_tasks):
                self.prediction_tasks[task.task_name] = task

        self._task_weights = defaultdict(lambda: 1)
        if task_weights:
            for key, val in task_weights.items():
                self._task_weights[key] = val

        self.build(inputs=inputs, task_blocks=task_blocks)

    def build(self, inputs=None, device=None, task_blocks=None):
        """Build each prediction task that's part of the head.

        Parameters
        ----------
        body
        inputs
        device
        task_blocks
        """
        if not getattr(self.body, "output_size", lambda: None)():
            raise ValueError(
                "Can't infer output-size of the body, please provide  "
                "a `Block` with a output-size. You can wrap any torch.Module in a Block."
            )

        input_size = self.body.output_size()

        if device:
            self.to(device)

        for name, task in self.prediction_tasks.items():
            task_block = task_blocks
            if task_blocks and isinstance(task_blocks, dict) and name in task_blocks:
                task_block = task_blocks[name]
            task.build(self.body, input_size, inputs=inputs, device=device, task_block=task_block)
        self.input_size = input_size

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        body: BlockType,
        task_blocks: Optional[Union[BlockType, Dict[str, BlockType]]] = None,
        task_weight_dict: Optional[Dict[str, float]] = None,
        loss_reduction: str = "mean",
        inputs: Optional[TabularFeaturesType] = None,
    ) -> "Head":
        """Instantiate a Head from a Schema through tagged targets.

        Parameters
        ----------
        schema: DatasetSchema
            Schema to use for inferring all targets based on the tags.
        body
        task_blocks
        task_weight_dict
        loss_reduction
        inputs

        Returns
        -------
        Head
        """
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
        )

    def pop_labels(self, inputs: TabularData) -> TabularData:
        """Pop the labels from the different prediction_tasks from the inputs.

        Parameters
        ----------
        inputs: TabularData
            Input dictionary containing all targets.

        Returns
        -------
        TabularData
        """
        outputs = {}
        for name in self.prediction_tasks.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def forward(
        self,
        body_outputs: Union[torch.Tensor, TabularData],
        training: bool = True,
        call_body: bool = False,
        always_output_dict: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, TabularData]:
        outputs = {}

        if call_body:
            body_outputs = self.body(body_outputs, training=training)

        for name, task in self.prediction_tasks.items():
            outputs[name] = task(body_outputs, **kwargs)

        if len(outputs) == 1 and not always_output_dict:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(
        self,
        body_outputs: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
        training: bool = True,
        compute_metrics: bool = True,
        call_body: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        losses = []

        if call_body:
            body_outputs = self.body(body_outputs, training=training)

        for name, task in self.prediction_tasks.items():
            loss = task.compute_loss(
                body_outputs, targets, compute_metrics=compute_metrics, **kwargs
            )
            losses.append(loss * self._task_weights[name])

        loss_tensor = torch.stack(losses)

        return getattr(loss_tensor, self.loss_reduction)()

    def calculate_metrics(
        self,
        body_outputs: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
        mode: str = "val",
        forward=True,
        call_body=False,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        metrics = {}

        if call_body:
            body_outputs = self.body(body_outputs, training=False)

        for name, task in self.prediction_tasks.items():
            metrics.update(
                task.calculate_metrics(body_outputs, targets, mode=mode, forward=forward, **kwargs)
            )

        return _output_metrics(metrics)

    def compute_metrics(self, mode: str = None) -> Dict[str, Union[float, torch.Tensor]]:
        def name_fn(x):
            return "_".join([mode, x]) if mode else x

        metrics = {
            name_fn(name): task.compute_metrics() for name, task in self.prediction_tasks.items()
        }

        return _output_metrics(metrics)

    def reset_metrics(self):
        """"""
        for task in self.prediction_tasks.values():
            task.reset_metrics()

    @property
    def task_blocks(self) -> Dict[str, Optional[BlockOrModule]]:
        return {name: task.task_block for name, task in self.prediction_tasks.items()}

    def to_model(self, **kwargs) -> Model:
        """Convert the head to a Model.

        Returns
        -------
        Model
        """
        from .model import Model as _Model

        return _Model(self, **kwargs)


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics
