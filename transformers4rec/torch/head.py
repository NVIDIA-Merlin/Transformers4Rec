from collections import defaultdict
from typing import Dict, Optional, Text, Union

import torch
import torchmetrics as tm

from ..types import ColumnGroup
from ..utils.columns import Tag


class PredictionTask(torch.nn.Module):
    def __init__(
        self,
        loss,
        metrics=None,
        body: Optional[torch.nn.Module] = None,
        forward_to_prediction_fn=lambda x: x,
        pre: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.forward_to_prediction_fn = forward_to_prediction_fn
        self.set_metrics(metrics)
        self.loss = loss
        self.body = body
        self.pre = pre

    def build(self, input_size, device=None):
        if device:
            self.to(device)
            for metric in self.metrics:
                metric.to(device)
        self.input_size = input_size

    def forward(self, inputs, **kwargs):
        x = inputs
        if self.body:
            x = self.body(x)
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
            if isinstance(metric, tuple([type(x) for x in self.binary_classification_metrics()])):
                labels = labels.int()
            outputs[f"{mode}_{metric.__class__.__name__.lower()}"] = metric(predictions, labels)

        return outputs

    def compute_metrics(self):
        return {f"{metric.__class__.__name__.lower()}": metric.compute() for metric in self.metrics}

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    @staticmethod
    def binary_classification_metrics():
        return [
            tm.Precision(num_classes=2),
            tm.Recall(num_classes=2),
            tm.Accuracy(),
            # tm.AUC()
        ]

    @classmethod
    def binary_classification(cls, metrics=None):
        metrics = metrics or cls.binary_classification_metrics()

        return cls(
            loss=torch.nn.BCELoss(),
            forward_to_prediction_fn=lambda x: torch.round(x).int(),
            metrics=metrics,
        )

    @staticmethod
    def regression_metrics():
        return [tm.regression.MeanSquaredError()]

    @classmethod
    def regression(cls, metrics=None):
        metrics = metrics or [tm.regression.MeanSquaredError()]

        return cls(loss=torch.nn.MSELoss(), metrics=metrics)


class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types

        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Head(torch.nn.Module):
    def __init__(self, input_size=None):
        super().__init__()
        if isinstance(input_size, int):
            input_size = [input_size]
        self.input_size = input_size
        self.tasks = torch.nn.ModuleDict()
        self._task_weights = defaultdict(lambda: 1)

    def build(self, input_size, device=None):
        if device:
            self.to(device)
            for task in self.tasks.values():
                task.build(input_size, device=device)
        self.input_size = input_size

    @classmethod
    def from_column_group(
        cls, column_group: ColumnGroup, add_logits=True, task_weights=None, input_size=None
    ):
        if task_weights is None:
            task_weights = {}
        to_return = cls(input_size=input_size)

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
