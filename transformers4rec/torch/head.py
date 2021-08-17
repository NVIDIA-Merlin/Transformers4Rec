from collections import defaultdict
from typing import Dict, List, Optional, Text, Union

import torch
import torchmetrics as tm

from transformers4rec.torch.typing import BuildableBlock

from ..types import DatasetSchema
from ..utils.tags import Tag
from .features.embedding import EmbeddingFeatures
from .tabular import MergeTabular


class PredictionTask(torch.nn.Module):
    def __init__(
        self,
        loss,
        metrics=None,
        target_name=None,
        forward_to_prediction_fn=lambda x: x,
        pre: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.target_name = target_name
        self.forward_to_prediction_fn = forward_to_prediction_fn
        self.set_metrics(metrics)
        self.loss = loss
        self.pre = pre

    def build(self, block, input_size, device=None):
        """
        The method will be called when block is convert to_model,
        i.e when linked to prediction head
        Inputs:
            block: (BlockType) the model block to link with head
            device: set the device for the metrics and layers of the task
        """
        if device:
            self.to(device)
            for metric in self.metrics:
                metric.to(device)
        self.built = True

    def forward(self, inputs, **kwargs):
        x = inputs
        if self.pre:
            x = self.pre(x)

        return x

    def set_metrics(self, metrics):
        self.metrics = torch.nn.ModuleList(metrics)

    def compute_loss(
        self, inputs, targets, training: bool = False, compute_metrics=True
    ) -> torch.Tensor:
        predictions = self(inputs)
        loss = self.loss(predictions, targets)

        if compute_metrics:
            self.calculate_metrics(predictions, targets, mode="train", forward=False)

            return loss

        return loss

    def calculate_metrics(
        self, predictions, labels, mode="val", forward=True
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        if forward:
            predictions = self(predictions)
        predictions = self.forward_to_prediction_fn(predictions)
        for metric in self.metrics:
            if isinstance(metric, tuple(type(x) for x in self.binary_classification_metrics())):
                labels = labels.int()
            outputs[f"{mode}_{metric.__class__.__name__.lower()}"] = metric(predictions, labels)

        return outputs

    def compute_metrics(self):
        return {f"{metric.__class__.__name__.lower()}": metric.compute() for metric in self.metrics}

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()


class BinaryClassificationTask(PredictionTask):
    DEFAULT_LOSS = torch.nn.BCELoss()
    DEFAULT_METRICS = (
        tm.Precision(num_classes=2),
        tm.Recall(num_classes=2),
        tm.Accuracy(),
        # tm.AUC()
    )

    def __init__(self, loss=DEFAULT_LOSS, metrics=DEFAULT_METRICS, target_name=None):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            forward_to_prediction_fn=lambda x: torch.round(x).int(),
        )

    def build(self, block, input_size, device=None):
        super().build(block, input_size, device=device)
        self.pre = torch.nn.Sequential(
            torch.nn.Linear(input_size[-1], 1, bias=False),
            torch.nn.Sigmoid(),
            LambdaModule(lambda x: x.view(-1)),
        )


class RegressionTask(PredictionTask):
    DEFAULT_LOSS = torch.nn.MSELoss()
    DEFAULT_METRICS = tm.regression.MeanSquaredError()

    def __init__(self, loss=DEFAULT_LOSS, metrics=DEFAULT_METRICS, target_name=None):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            forward_to_prediction_fn=lambda x: torch.round(x).int(),
        )

    def build(self, block, input_size, device=None):
        super().build(block, input_size, device=device)
        self.pre = torch.nn.Sequential(
            torch.nn.Linear(input_size[-1], 1), LambdaModule(lambda x: x.view(-1))
        )


class SequentialPredictionTask(PredictionTask):
    def __init__(
        self,
        loss,
        metrics=None,
        body: Optional[torch.nn.Module] = None,
        pre: Optional[torch.nn.Module] = None,
        forward_to_prediction_fn=lambda x: x,
        mf_constrained_embeddings: bool = True,
        item_id_name: str = None,
        item_embedding_table: Optional[torch.nn.Module] = None,
        output_layer_bias: Optional[torch.nn.Parameter] = None,
        input_size: int = None,
        vocab_size: int = None,
    ):
        super(SequentialPredictionTask, self).__init__(
            loss=loss,
            metrics=metrics,
            body=body,
            forward_to_prediction_fn=forward_to_prediction_fn,
        )

        self.mf_constrained_embeddings = mf_constrained_embeddings
        self.vocab_size = vocab_size
        self.input_size = input_size

        self.item_embedding_table = item_embedding_table
        self.output_layer_bias = output_layer_bias

        self.pre = pre
        self.itemid_name = item_id_name

    def build(self, block, device=None):
        # Retrieve item embedding table
        for layer in block.children():
            if isinstance(layer, MergeTabular):
                for feature in layer.to_merge:
                    if isinstance(feature, EmbeddingFeatures):
                        if self.itemid_name in feature.embedding_tables:
                            item_embedding_table = feature.embedding_tables[self.itemid_name]
        if not self.vocab_size:
            self.vocab_size = item_embedding_table.weight.size(0)
        self.item_embedding_table = item_embedding_table
        self.output_layer_bias = torch.nn.Parameter(torch.Tensor(self.vocab_size))
        torch.nn.init.zeros_(self.output_layer_bias)
        super(SequentialPredictionTask, self).build(block)

    def forward(self, inputs, **kwargs):
        x = inputs.float()
        if self.body:
            x = self.body(x)

        if self.mf_constrained_embeddings:
            x = torch.nn.functional.linear(
                x,
                weight=self.item_embedding_table.weight.float(),
                bias=self.output_layer_bias,
            )

        if self.pre:
            x = self.pre(x)
        return x


class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types

        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Head(torch.nn.Module):
    def __init__(
        self,
        body,
        prediction_tasks: Optional[Union[List[PredictionTask], PredictionTask]] = None,
        task_towers: Optional[Union[BuildableBlock, Dict[str, BuildableBlock]]] = None,
        task_weights=None,
        body_output_size=None,
    ):
        super().__init__()
        if isinstance(body_output_size, int):
            body_output_size = [body_output_size]
        self.body_output_size = body_output_size
        self.body = body
        self.prediction_tasks = torch.nn.ModuleDict()
        if prediction_tasks:
            if not isinstance(prediction_tasks, list):
                prediction_tasks = [prediction_tasks]
            for i, task in enumerate(prediction_tasks):
                self.prediction_tasks[task.target_name or str(i)] = task

        self._task_weights = defaultdict(lambda: 1)

    def build(self, input_size, device=None):
        if device:
            self.to(device)
            for task in self.tasks.values():
                task.build(input_size, device=device)
        self.input_size = input_size

    @classmethod
    def from_schema(
        cls, schema: DatasetSchema, add_logits=True, task_weights=None, input_size=None
    ):
        if task_weights is None:
            task_weights = {}
        to_return = cls(input_size=input_size)

        for binary_target in schema.select_by_tag(Tag.TARGETS_BINARY).column_names:
            to_return = to_return.add_binary_classification_task(
                binary_target,
                add_logit_layer=add_logits,
                task_weight=task_weights.get(binary_target, 1),
            )

        for regression_target in schema.select_by_tag(Tag.TARGETS_REGRESSION).column_names:
            to_return = to_return.add_regression_task(
                regression_target,
                add_logit_layer=add_logits,
                task_weight=task_weights.get(regression_target, 1),
            )

        # TODO: Add multi-class classification here. Figure out how to get number of classes

        return to_return

    def add_task(
        self,
        target_name,
        task: PredictionTask,
        pre: Optional[torch.nn.Module] = None,
        task_weight=1,
    ):
        self.tasks[target_name] = task
        if pre:
            self._tasks_prepares[target_name] = pre
        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def add_binary_classification_task(self, target_name, add_logit_layer=True, task_weight=1):
        self.tasks[target_name] = PredictionTask.binary_classification()

        if add_logit_layer:
            self.tasks[target_name].pre = torch.nn.Sequential(
                torch.nn.Linear(self.input_size[-1], 1, bias=False),
                torch.nn.Sigmoid(),
                LambdaModule(lambda x: x.view(-1)),
            )

        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def add_regression_task(self, target_name, add_logit_layer=True, task_weight=1):
        self.tasks[target_name] = PredictionTask.regression()

        if add_logit_layer:
            self.tasks[target_name].pre = torch.nn.Sequential(
                torch.nn.Linear(self.input_size[-1], 1), LambdaModule(lambda x: x.view(-1))
            )
        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def pop_labels(self, inputs: Dict[Text, torch.Tensor]):
        outputs = {}
        for name in self.tasks.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def forward(self, logits: torch.Tensor, **kwargs):
        outputs = {}

        for name, task in self.tasks.items():
            outputs[name] = task(logits, **kwargs)

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(self, block_outputs, targets, **kwargs) -> torch.Tensor:
        losses = []

        for name, task in self.tasks.items():
            loss = task.compute_loss(block_outputs, targets, **kwargs)
            losses.append(loss * self._task_weights[name])

        return torch.sum(*losses)

    def calculate_metrics(
        self, block_outputs, targets, mode="val"
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        metrics = {}

        for name, task in self.tasks.items():
            metrics[name] = task.calculate_metrics(block_outputs, targets, mode=mode)

        return _output_metrics(metrics)

    def compute_metrics(self, mode=None):
        def name_fn(x):
            return "_".join([mode, x]) if mode else x

        metrics = {name_fn(name): task.compute_metrics() for name, task in self.tasks.items()}

        return _output_metrics(metrics)

    def reset_metrics(self):
        for task in self.tasks.values():
            task.reset_metrics()


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics
