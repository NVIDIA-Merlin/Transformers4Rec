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
import copy
import inspect
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Type, Union, cast

import numpy as np
import torch
import torchmetrics as tm
from tqdm import tqdm
from transformers.modeling_utils import SequenceSummary

from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.registry import camelcase_to_snakecase

from ..block.base import BlockBase, BlockOrModule, BlockType
from ..features.base import InputBlock
from ..features.sequence import TabularFeaturesType
from ..typing import TabularData, TensorOrTabularData
from ..utils.torch_utils import LossMixin, MetricsMixin


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None


class PredictionTask(torch.nn.Module, LossMixin, MetricsMixin):
    """Individual prediction-task of a model.

    Parameters
    ----------
    loss: torch.nn.Module
        The loss to use during training of this task.
    metrics: torch.nn.Module
        The metrics to calculate during training & evaluation.
    target_name: str, optional
        Name of the target, this is needed when there are multiple targets.
    task_name: str, optional
        Name of the prediction task, if not provided a name will be automatically constructed based
        on the target-name & class-name.
    forward_to_prediction_fn: Callable[[torch.Tensor], torch.Tensor]
        Function to apply before the prediction
    task_block: BlockType
        Module to transform input tensor before computing predictions.
    pre: BlockType
        Module to compute the predictions probabilities.
    summary_type: str
        This is used to summarize a sequence into a single tensor. Accepted values are:
            - `"last"` -- Take the last token hidden state (like XLNet)
            - `"first"` -- Take the first token hidden state (like Bert)
            - `"mean"` -- Take the mean of all tokens hidden states
            - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
            - `"attn"` -- Not implemented now, use multi-head attention
    """

    def __init__(
        self,
        loss: torch.nn.Module,
        metrics: Iterable[tm.Metric] = None,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        forward_to_prediction_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        task_block: Optional[BlockType] = None,
        pre: Optional[BlockType] = None,
        summary_type: str = "last",
    ):
        super().__init__()
        self.sequence_summary = SequenceSummary(
            SimpleNamespace(summary_type=summary_type)  # type: ignore
        )  # noqa
        self.target_name = target_name
        self.forward_to_prediction_fn = forward_to_prediction_fn
        self.set_metrics(metrics)
        self.loss = loss
        self.pre = pre
        self.task_block = task_block
        self._task_name = task_name

    def build(
        self,
        body: BlockType,
        input_size,
        inputs: Optional[InputBlock] = None,
        device=None,
        task_block: Optional[BlockType] = None,
        pre=None,
    ):
        """
        The method will be called when block is converted to a model,
        i.e when linked to prediction head.

        Parameters
        ----------
        block:
            the model block to link with head
        device:
            set the device for the metrics and layers of the task
        """

        if task_block:
            # TODO: What to do when `self.task_block is not None`?
            self.task_block = task_block
        if pre:
            # TODO: What to do when `self.pre is not None`?
            self.pre = pre

        # Build task block
        pre_input_size = input_size
        if self.task_block:
            if isinstance(self.task_block, torch.nn.Module):
                self.task_block = copy.deepcopy(self.task_block)
            else:
                self.task_block = self.task_block.build(input_size)
            pre_input_size = self.task_block.output_size()  # type: ignore

        if self.pre:
            if isinstance(self.pre, torch.nn.Module):
                self.pre = copy.deepcopy(self.pre)
            else:
                self.pre = self.pre.build(pre_input_size)

        if device:
            self.to(device)
            for metric in self.metrics:
                metric.to(device)
        self.built = True

    def forward(self, inputs, **kwargs):
        x = inputs

        if len(x.size()) == 3:
            x = self.sequence_summary(x)

        if self.task_block:
            x = self.task_block(x)

        if self.pre:
            x = self.pre(x)

        return x

    @property
    def task_name(self):
        if self._task_name:
            return self._task_name

        base_name = camelcase_to_snakecase(self.__class__.__name__)

        return name_fn(self.target_name, base_name) if self.target_name else base_name

    def child_name(self, name):
        return name_fn(self.task_name, name)

    def set_metrics(self, metrics):
        self.metrics = torch.nn.ModuleList(metrics)

    def compute_loss(
        self,
        inputs: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
        compute_metrics: bool = True,
        training: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        predictions = self(inputs, training=training)
        loss = self.loss(predictions, targets)

        if compute_metrics:
            self.calculate_metrics(predictions, targets, mode="train", forward=False)

            return loss

        return loss

    def calculate_metrics(  # type: ignore
        self,
        predictions: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
        mode: str = "val",
        forward: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        outputs = {}
        if forward:
            predictions = self(predictions)
        predictions = self.forward_to_prediction_fn(cast(torch.Tensor, predictions))

        from .prediction_task import BinaryClassificationTask

        for metric in self.metrics:
            if isinstance(metric, tuple(type(x) for x in BinaryClassificationTask.DEFAULT_METRICS)):
                targets = cast(torch.Tensor, targets).int()
            outputs[self.metric_name(metric)] = metric(predictions, targets)

        return outputs

    def compute_metrics(self, **kwargs):
        return {self.metric_name(metric): metric.compute() for metric in self.metrics}

    def metric_name(self, metric: tm.Metric) -> str:
        return self.child_name(camelcase_to_snakecase(metric.__class__.__name__))

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def to_head(self, body, inputs=None, **kwargs) -> "Head":
        return Head(body, self, inputs=inputs, **kwargs)

    def to_model(self, body, inputs=None, **kwargs) -> "Model":
        return Model(Head(body, self, inputs=inputs, **kwargs), **kwargs)


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
        body: BlockBase,
        prediction_tasks: Union[List[PredictionTask], PredictionTask],
        task_blocks: Optional[Union[BlockType, Dict[str, BlockType]]] = None,
        task_weights: Optional[List[float]] = None,
        loss_reduction: str = "mean",
        inputs: Optional[TabularFeaturesType] = None,
    ):
        super().__init__()
        self.body = body
        self.loss_reduction = loss_reduction
        self.prediction_task_dict = torch.nn.ModuleDict()
        if prediction_tasks:
            if not isinstance(prediction_tasks, list):
                prediction_tasks = [prediction_tasks]
            for i, task in enumerate(prediction_tasks):
                self.prediction_task_dict[task.task_name] = task

        self._task_weights = defaultdict(lambda: 1.0)
        if task_weights:
            for task, val in zip(cast(List[PredictionTask], prediction_tasks), task_weights):
                self._task_weights[task.task_name] = val

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

        for name, task in self.prediction_task_dict.items():
            task_block = task_blocks
            if task_blocks and isinstance(task_blocks, dict) and name in task_blocks:
                task_block = task_blocks[name]
            task.build(self.body, input_size, inputs=inputs, device=device, task_block=task_block)
        self.input_size = input_size

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        body: BlockBase,
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
        for name in self.prediction_task_dict.keys():
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

        for name, task in self.prediction_task_dict.items():
            outputs[name] = task(body_outputs, **kwargs)

        if len(outputs) == 1 and not always_output_dict:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(  # type: ignore
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

        for name, task in self.prediction_task_dict.items():
            loss = task.compute_loss(
                body_outputs, targets, compute_metrics=compute_metrics, **kwargs
            )
            losses.append(loss * self._task_weights[name])

        loss_tensor = torch.stack(losses)

        return getattr(loss_tensor, self.loss_reduction)()

    def calculate_metrics(  # type: ignore
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

        for name, task in self.prediction_task_dict.items():
            metrics.update(
                task.calculate_metrics(body_outputs, targets, mode=mode, forward=forward, **kwargs)
            )

        return _output_metrics(metrics)

    def compute_metrics(self, mode: str = None) -> Dict[str, Union[float, torch.Tensor]]:
        def name_fn(x):
            return "_".join([mode, x]) if mode else x

        metrics = {
            name_fn(name): task.compute_metrics()
            for name, task in self.prediction_task_dict.items()
        }

        return _output_metrics(metrics)

    def reset_metrics(self):
        """"""
        for task in self.prediction_task_dict.values():
            task.reset_metrics()

    @property
    def task_blocks(self) -> Dict[str, Optional[BlockOrModule]]:
        return {name: task.task_block for name, task in self.prediction_task_dict.items()}

    def to_model(self, **kwargs) -> "Model":
        """Convert the head to a Model.

        Returns
        -------
        Model
        """

        return Model(self, **kwargs)


class Model(torch.nn.Module, LossMixin, MetricsMixin):
    """Model class that can aggregate one of multiple heads.

    Parameters
    ----------
    head: Head
        One or more heads of the model.
    head_weights: List[float], optional
        Weight-value to use for each head.
    head_reduction: str, optional
        How to reduce the losses into a single tensor when multiple heads are used.
    optimizer: Type[torch.optim.Optimizer]
        Optimizer-class to use during fitting
    name: str, optional
        Name of the model.
    """

    def __init__(
        self,
        *head: Head,
        head_weights: Optional[List[float]] = None,
        head_reduction: str = "mean",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        name=None,
    ):
        """
        #TODO
        """
        if head_weights:
            if not isinstance(head_weights, list):
                raise ValueError("`head_weights` must be a list")
            if not len(head_weights) == len(head):
                raise ValueError(
                    "`head_weights` needs to have the same length " "as the number of heads"
                )

        super().__init__()

        self.name = name
        self.heads = torch.nn.ModuleList(head)
        self.head_weights = head_weights or [1.0] * len(head)
        self.head_reduction = head_reduction
        self.optimizer = optimizer

    def forward(self, inputs: TensorOrTabularData, training=True, **kwargs):
        # TODO: Optimize this
        outputs = {}
        for head in self.heads:
            outputs.update(head(inputs, call_body=True, training=training, always_output_dict=True))

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(self, inputs, targets, compute_metrics=True, **kwargs) -> torch.Tensor:
        losses = []

        for i, head in enumerate(self.heads):
            loss = head.compute_loss(
                inputs, targets, call_body=True, compute_metrics=compute_metrics, **kwargs
            )
            losses.append(loss * self.head_weights[i])

        loss_tensor = torch.stack(losses)

        return getattr(loss_tensor, self.head_reduction)()

    def calculate_metrics(  # type: ignore
        self, inputs, targets, mode="val", call_body=True, forward=True, **kwargs
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        outputs = {}
        for head in self.heads:
            outputs.update(
                head.calculate_metrics(
                    inputs, targets, mode=mode, call_body=call_body, forward=forward, **kwargs
                )
            )

        return outputs

    def compute_metrics(self, mode=None) -> Dict[str, Union[float, torch.Tensor]]:
        metrics = {}
        for head in self.heads:
            metrics.update(head.compute_metrics(mode=mode))

        return metrics

    def reset_metrics(self):
        for head in self.heads:
            head.reset_metrics()

    def to_lightning(self):
        import pytorch_lightning as pl

        parent_self = self

        class BlockWithHeadLightning(pl.LightningModule):
            def __init__(self):
                super(BlockWithHeadLightning, self).__init__()
                self.parent = parent_self

            def forward(self, inputs, *args, **kwargs):
                return self.parent(inputs, *args, **kwargs)

            def training_step(self, batch, batch_idx):
                loss = self.parent.compute_loss(*batch)
                self.log("train_loss", loss)

                return loss

            def configure_optimizers(self):
                optimizer = self.parent.optimizer(self.parent.parameters(), lr=1e-3)

                return optimizer

        return BlockWithHeadLightning()

    def fit(
        self,
        dataloader,
        optimizer=torch.optim.Adam,
        eval_dataloader=None,
        num_epochs=1,
        amp=False,
        train=True,
        verbose=True,
    ):
        if isinstance(dataloader, torch.utils.data.DataLoader):
            dataset = dataloader.dataset
        else:
            dataset = dataloader

        if inspect.isclass(optimizer):
            optimizer = optimizer(self.parameters())

        self.train(mode=train)
        epoch_losses = []
        with torch.set_grad_enabled(mode=train):
            for epoch in range(num_epochs):
                losses = []
                batch_iterator = enumerate(iter(dataset))
                if verbose:
                    batch_iterator = tqdm(batch_iterator)
                for batch_idx, (x, y) in batch_iterator:
                    if amp:
                        with torch.cuda.amp.autocast():
                            loss = self.compute_loss(x, y)
                    else:
                        loss = self.compute_loss(x, y)

                    losses.append(float(loss))

                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                if verbose:
                    print(self.compute_metrics(mode="train"))
                    if eval_dataloader:
                        print(self.evaluate(eval_dataloader, verbose=False))
                epoch_losses.append(np.mean(losses))

        return np.array(epoch_losses)

    def evaluate(self, dataloader, verbose=True, mode="eval"):
        if isinstance(dataloader, torch.utils.data.DataLoader):
            dataset = dataloader.dataset
        else:
            dataset = dataloader

        batch_iterator = enumerate(iter(dataset))
        if verbose:
            batch_iterator = tqdm(batch_iterator)
        self.reset_metrics()
        for batch_idx, (x, y) in batch_iterator:
            self.calculate_metrics(x, y, mode=mode)

        return self.compute_metrics(mode=mode)

    def _get_name(self):
        if self.name:
            return self.name

        return super(Model, self)._get_name()


def _output_metrics(metrics):
    # If there is only a single head with metrics, returns just those metrics
    if len(metrics) == 1 and isinstance(metrics[list(metrics.keys())[0]], dict):
        return metrics[list(metrics.keys())[0]]

    return metrics
