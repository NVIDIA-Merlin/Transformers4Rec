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
import os
import pathlib
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Type, Union, cast

import numpy as np
import torch
import torchmetrics as tm
from merlin.models.utils.registry import camelcase_to_snakecase
from merlin.schema import ColumnSchema
from merlin.schema import Schema as Core_Schema
from merlin.schema import Tags
from tqdm import tqdm
from transformers.modeling_utils import SequenceSummary

from merlin_standard_lib import Schema

from ..block.base import BlockBase, BlockOrModule, BlockType
from ..features.base import InputBlock
from ..features.sequence import TabularFeaturesType
from ..typing import TabularData
from ..utils.padding import pad_inputs
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
        self.summary_type = summary_type
        self.sequence_summary = SequenceSummary(
            SimpleNamespace(summary_type=self.summary_type)  # type: ignore
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

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor = None,
        training: bool = False,
        testing: bool = False,
    ):
        x = inputs

        if len(x.size()) == 3 and self.summary_type:
            x = self.sequence_summary(x)

        if self.task_block:
            x = self.task_block(x)  # type: ignore

        if self.pre:
            x = self.pre(x)  # type: ignore

        if training or testing:
            # add support of computing the loss inside the forward
            # and return a dictionary as standard output
            if self.summary_type is None:
                if targets.dim() != 2:
                    raise ValueError(
                        "If `summary_type==None`, targets are expected to be a 2D tensor, "
                        f"but got a tensor with shape {targets.shape}"
                    )

            loss = self.loss(x, target=targets)
            return {"loss": loss, "labels": targets, "predictions": x}

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

    def calculate_metrics(  # type: ignore
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = {}

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

        for binary_target in schema.select_by_tag([Tags.BINARY, Tags.CLASSIFICATION]).column_names:
            tasks.append(BinaryClassificationTask(binary_target))
            task_weights.append(task_weight_dict.get(binary_target, 1.0))

        for regression_target in schema.select_by_tag(Tags.REGRESSION).column_names:
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
        training: bool = False,
        testing: bool = False,
        targets: Union[torch.Tensor, TabularData] = None,
        call_body: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, TabularData]:
        outputs = {}

        if call_body:
            body_outputs = self.body(body_outputs, training=training, testing=testing, **kwargs)

        if training or testing:
            losses = []
            labels = {}
            predictions = {}
            for name, task in self.prediction_task_dict.items():
                if isinstance(targets, dict):
                    label = targets.get(task.target_name, None)
                else:
                    label = targets
                if label is not None:
                    label = label.float()
                task_output = task(
                    body_outputs, targets=label, training=training, testing=testing, **kwargs
                )
                labels[name] = task_output["labels"]
                predictions[name] = task_output["predictions"]
                losses.append(task_output["loss"] * self._task_weights[name])
            loss_tensor = torch.stack(losses)
            loss = getattr(loss_tensor, self.loss_reduction)()
            outputs = {"loss": loss, "labels": labels, "predictions": predictions}
        else:
            for name, task in self.prediction_task_dict.items():
                outputs[name] = task(
                    body_outputs, targets=targets, training=training, testing=testing, **kwargs
                )

        return outputs

    def calculate_metrics(  # type: ignore
        self,
        predictions: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Calculate metrics of the task(s) set in the Head instance.
        Parameters
        ----------
        predictions: Union[torch.Tensor, TabularData]
            The predictions tensors to use for calculate metrics.
            They can be either a torch.Tensor if a single task is used or
            a dictionary of torch.Tensor if multiple tasks are used. In the
            second case, the dictionary is indexed by the tasks names.
        targets:
            The tensor or dictionary of targets to use for computing the metrics of
            one or multiple tasks.
        """
        metrics = {}

        for name, task in self.prediction_task_dict.items():
            label = targets
            output = predictions
            if isinstance(targets, dict):
                # The labels are retrieved from the task's output
                # and indexed by the task name.
                label = targets[name]
            if isinstance(predictions, dict):
                output = predictions[name]

            metrics.update(
                task.calculate_metrics(
                    predictions=output,
                    targets=label,
                )
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
    def __init__(
        self,
        *head: Head,
        head_weights: Optional[List[float]] = None,
        head_reduction: str = "mean",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        name: str = None,
        max_sequence_length: Optional[int] = None,
    ):
        """Model class that can aggregate one or multiple heads.
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
        max_sequence_length : int, optional
            The maximum sequence length supported by the model.
            Used to truncate sequence inputs longer than this value.
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
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs):
        # Convert inputs to float32 which is the default type, expected by PyTorch
        for name, val in inputs.items():
            if torch.is_floating_point(val):
                inputs[name] = val.to(torch.float32)

        # pad ragged inputs
        inputs = pad_inputs(inputs, self.max_sequence_length)

        if isinstance(targets, dict) and len(targets) == 0:
            # `pyarrow`` dataloader is returning {} instead of None
            # TODO remove this code when `PyarraowDataLoader` is dropped
            targets = None

        # TODO: Optimize this
        if training or testing:
            losses = []
            labels = {}
            predictions = {}
            for i, head in enumerate(self.heads):
                head_output = head(
                    inputs,
                    call_body=True,
                    targets=targets,
                    training=training,
                    testing=testing,
                    **kwargs,
                )
                labels.update(head_output["labels"])
                predictions.update(head_output["predictions"])
                losses.append(head_output["loss"] * self.head_weights[i])
            loss_tensor = torch.stack(losses)
            loss = getattr(loss_tensor, self.head_reduction)()
            if len(labels) == 1:
                labels = list(labels.values())[0]
                predictions = list(predictions.values())[0]
            return {"loss": loss, "labels": labels, "predictions": predictions}
        else:
            outputs = {}
            for head in self.heads:
                outputs.update(
                    head(
                        inputs,
                        call_body=True,
                        targets=targets,
                        training=training,
                        testing=testing,
                        **kwargs,
                    )
                )
            if len(outputs) == 1:
                return list(outputs.values())[0]

        return outputs

    def calculate_metrics(  # type: ignore
        self,
        predictions: Union[torch.Tensor, TabularData],
        targets: Union[torch.Tensor, TabularData],
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Calculate metrics of the task(s) set in the Head instance.
        Parameters
        ----------
        predictions: Union[torch.Tensor, TabularData]
            The predictions tensors returned by the model.
            They can be either a torch.Tensor if a single task is used or
            a dictionary of torch.Tensor if multiple heads/tasks are used. In the
            second case, the dictionary is indexed by the tasks names.
        targets:
            The tensor or dictionary of targets returned by the model.
            They are used for computing the metrics of one or multiple tasks.
        """
        outputs = {}
        for head in self.heads:
            outputs.update(
                head.calculate_metrics(
                    predictions,
                    targets,
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

            def forward(self, inputs, targets=None, training=False, testing=False, *args, **kwargs):
                return self.parent(
                    inputs, targets=targets, training=training, testing=testing, *args, **kwargs
                )

            def training_step(self, batch, batch_idx, targets=None, training=True, testing=False):
                loss = self.parent(*batch, targets=targets, training=training, testing=testing)[
                    "loss"
                ]
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
        compute_metric=True,
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
                            output = self(x, targets=y, training=True)
                    else:
                        output = self(x, targets=y, training=True)
                    losses.append(float(output["loss"]))
                    if compute_metric:
                        self.calculate_metrics(
                            output["predictions"],
                            targets=output["labels"],
                        )
                    if train:
                        optimizer.zero_grad()
                        output["loss"].backward()
                        optimizer.step()
                if verbose:
                    print(self.compute_metrics(mode="train"))
                    if eval_dataloader:
                        print(self.evaluate(eval_dataloader, verbose=False))
                epoch_losses.append(np.mean(losses))

        return np.array(epoch_losses)

    def evaluate(
        self, dataloader, targets=None, training=False, testing=True, verbose=True, mode="eval"
    ):
        if isinstance(dataloader, torch.utils.data.DataLoader):
            dataset = dataloader.dataset
        else:
            dataset = dataloader

        batch_iterator = enumerate(iter(dataset))
        if verbose:
            batch_iterator = tqdm(batch_iterator)
        self.reset_metrics()
        for batch_idx, (x, y) in batch_iterator:
            output = self(x, targets=y, training=training, testing=testing)
            self.calculate_metrics(
                output["predictions"],
                targets=output["labels"],
            )

        return self.compute_metrics(mode=mode)

    def _get_name(self):
        if self.name:
            return self.name

        return super(Model, self)._get_name()

    @property
    def input_schema(self):
        # return the input schema given by the model
        # loop over the heads to get input schemas
        schemas = []
        for head in self.heads:
            schemas.append(head.body.inputs.schema)
        if all(isinstance(s, Core_Schema) for s in schemas):
            return sum(schemas, Core_Schema())

        model_schema = sum(schemas, Schema())

        # TODO: rework T4R to use Merlin Schemas.
        # In the meantime, we convert model_schema to merlin core schema
        core_schema = Core_Schema()
        for column in model_schema:
            name = column.name

            dtype = {0: np.float32, 2: np.int64, 3: np.float32}[column.type]
            tags = column.tags
            dims = None
            if column.value_count.max > 0:
                dims = (None, (column.value_count.min, column.value_count.max))
            int_domain = {"min": column.int_domain.min, "max": column.int_domain.max}
            properties = {
                "int_domain": int_domain,
            }

            col_schema = ColumnSchema(
                name, dtype=dtype, tags=tags, properties=properties, dims=dims
            )
            core_schema[name] = col_schema
        return core_schema

    @property
    def output_schema(self):
        from .prediction_task import BinaryClassificationTask, RegressionTask

        # if the model has one head with one task, the output is a tensor
        # if multiple heads and/or multiple prediction task, the output is a dictionary
        output_cols = []
        for head in self.heads:
            dims = None
            for name, task in head.prediction_task_dict.items():
                target_dim = task.target_dim
                int_domain = {"min": target_dim, "max": target_dim}
                if (
                    isinstance(task, (BinaryClassificationTask, RegressionTask))
                    and not task.summary_type
                ):
                    dims = (None, (1, None))
                elif (
                    isinstance(task, (BinaryClassificationTask, RegressionTask))
                    and task.summary_type
                ):
                    dims = (None,)
                else:
                    dims = (None, task.target_dim)
                properties = {
                    "int_domain": int_domain,
                }
                col_schema = ColumnSchema(name, dtype=np.float32, properties=properties, dims=dims)
                output_cols.append(col_schema)

        return Core_Schema(output_cols)

    @property
    def prediction_tasks(self):
        return [task for head in self.heads for task in list(head.prediction_task_dict.values())]

    def save(self, path: Union[str, os.PathLike], model_name="t4rec_model_class"):
        """Saves the model to f"{export_path}/{model_name}.pkl" using `cloudpickle`
        Parameters
        ----------
        path : Union[str, os.PathLike]
            Path to the directory where the T4Rec model should be saved.
        model_name : str, optional
           the name given to the pickle file storing the T4Rec model,
            by default 't4rec_model_class'
        """
        try:
            import cloudpickle
        except ImportError:
            raise ValueError("cloudpickle is required to save model class")

        export_path = pathlib.Path(path)
        export_path.mkdir(exist_ok=True)

        model_name = model_name + ".pkl"
        export_path = export_path / model_name
        with open(export_path, "wb") as out:
            cloudpickle.dump(self, out)

    @classmethod
    def load(cls, path: Union[str, os.PathLike], model_name="t4rec_model_class") -> "Model":
        """Loads a T4Rec model that was saved with `model.save()`.
        Parameters
        ----------
        path : Union[str, os.PathLike]
            Path to the directory where the T4Rec model is saved.
        model_name : str, optional
           the name given to the pickle file storing the T4Rec model,
            by default 't4rec_model_class'.
        """
        try:
            import cloudpickle
        except ImportError:
            raise ValueError("cloudpickle is required to load T4Rec model")

        export_path = pathlib.Path(path)
        model_name = model_name + ".pkl"
        export_path = export_path / model_name
        return cloudpickle.load(open(export_path, "rb"))


def _output_metrics(metrics):
    # If there is only a single head with metrics, returns just those metrics
    if len(metrics) == 1 and isinstance(metrics[list(metrics.keys())[0]], dict):
        return metrics[list(metrics.keys())[0]]

    return metrics
