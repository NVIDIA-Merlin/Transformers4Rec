from collections import defaultdict
from typing import Dict, List, Optional, Text, Union

import torch
import torchmetrics as tm

from transformers4rec.torch.typing import BuildableBlock, SequentialBlock

from ..types import DatasetSchema
from ..utils.tags import Tag


class PredictionTask(torch.nn.Module):
    def __init__(
        self,
        loss,
        metrics=None,
        target_name=None,
        forward_to_prediction_fn=lambda x: x,
        pre: Optional[torch.nn.Module] = None,
        summary_type="first",
    ):
        super().__init__()
        self.summary_type = summary_type
        self.target_name = target_name
        self.forward_to_prediction_fn = forward_to_prediction_fn
        self.set_metrics(metrics)
        self.loss = loss
        self.pre = pre

    def build(self, block, input_size, inputs=None, device=None):
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
            if len(x.size()) == 3:
                # TODO: Implement this properly
                pass
            x = self.pre(x)

        return x

    def set_metrics(self, metrics):
        self.metrics = torch.nn.ModuleList(metrics)

    def compute_loss(
        self, inputs, targets, training: bool = False, compute_metrics=True
    ) -> torch.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

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
            if isinstance(metric, tuple(type(x) for x in BinaryClassificationTask.DEFAULT_METRICS)):
                labels = labels.int()
            outputs[f"{mode}_{metric.__class__.__name__.lower()}"] = metric(predictions, labels)

        return outputs

    def compute_metrics(self):
        return {f"{metric.__class__.__name__.lower()}": metric.compute() for metric in self.metrics}

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def to_head(self, body, inputs=None, **kwargs) -> "Head":
        return Head(body, self, inputs=inputs, **kwargs)


class BinaryClassificationTask(PredictionTask):
    DEFAULT_LOSS = torch.nn.BCELoss()
    DEFAULT_METRICS = (
        tm.Precision(num_classes=2),
        tm.Recall(num_classes=2),
        tm.Accuracy(),
        # tm.AUC()
    )

    def __init__(
        self, target_name=None, loss=DEFAULT_LOSS, metrics=DEFAULT_METRICS, summary_type="first"
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            summary_type=summary_type,
            forward_to_prediction_fn=lambda x: torch.round(x).int(),
        )

    def build(self, block, input_size, inputs=None, device=None):
        super().build(block, input_size, device=device)
        self.pre = torch.nn.Sequential(
            torch.nn.Linear(input_size[-1], 1, bias=False),
            torch.nn.Sigmoid(),
            LambdaModule(lambda x: x.view(-1)),
        )


class RegressionTask(PredictionTask):
    DEFAULT_LOSS = torch.nn.MSELoss()
    DEFAULT_METRICS = (tm.regression.MeanSquaredError(),)

    def __init__(
        self, target_name=None, loss=DEFAULT_LOSS, metrics=DEFAULT_METRICS, summary_type="first"
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            summary_type=summary_type,
            forward_to_prediction_fn=lambda x: torch.round(x).int(),
        )

    def build(self, block, input_size, inputs=None, device=None):
        super().build(block, input_size, device=device)
        self.pre = torch.nn.Sequential(
            torch.nn.Linear(input_size[-1], 1), LambdaModule(lambda x: x.view(-1))
        )


class NextItemPredictionTask(PredictionTask):
    def __init__(
        self,
        loss=torch.nn.NLLLoss(ignore_index=0),
        metrics=None,
        pre: Optional[torch.nn.Module] = None,
        weight_tying: bool = False,
        softmax_temperature: float = 1,
        pad_token: int = 0,
        target_dim: int = None,
        hf_format=False,
    ):
        """
        Class to support Next Item prediction task:
        Parameters:
        ----------
            loss : the loss function to use.
            metrics: list of ranking metrics to use for evaluation.
            body: network to transform input tensor before computing predictions.
            pre: classifier network to compute the item predictions probabilities.
            weight_tying: the item id embedding table weights are shared
                        with the prediction network layer.
            softmax_temperature: Softmax temperature, used to reduce model overconfidence,
                        so that softmax(logits / T). Value 1.0 reduces to regular softmax.
            pad_token: pad token id.
            target_dim: vocabulary size of item ids
            HF_format: output the dictionary of outputs needed by RecSysTrained,
                        if set to False, return the predictions tensor.
        """
        super().__init__(loss=loss, metrics=metrics, pre=pre)
        self.pre = pre
        self.softmax_temperature = softmax_temperature
        self.weight_tying = weight_tying
        self.pad_token = pad_token
        self.target_dim = target_dim
        self.hf_format = hf_format

        self.item_embedding_table = None
        self.masking = None

    def build(self, body, input_size, device=None, inputs=None):
        # Retrieve the embedding module to get the name of itemid col and its related table

        # TODO: retrieve embeddings
        if not inputs:
            inputs = body.inputs
        if not getattr(inputs, "item_id", None):
            raise ValueError(
                "For Item Prediction task a categorical_module "
                "including an item_id column is required."
            )
        embeddings = inputs.categorical_module
        if not self.target_dim:
            self.target_dim = embeddings.item_embedding_table.num_embeddings
        if self.weight_tying:
            self.item_embedding_table = embeddings.item_embedding_table

        # Retrieve the masking if used in the model block
        self.masking = inputs.masking
        if self.masking:
            self.pad_token = self.masking.pad_token

        self.pre = _ItemPredictionTask(
            input_size=input_size,
            target_dim=self.target_dim,
            weight_tying=self.weight_tying,
            item_embedding_table=self.item_embedding_table,
            softmax_temperature=self.softmax_temperature,
        )

        super().build(body, input_size, device=device, inputs=inputs)

    def forward(self, inputs, **kwargs):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]
        x = inputs.float()

        # Retrieve labels either from masking or input module
        if self.masking:
            labels = self.masking.masked_targets
        else:
            labels = self.embeddings.item_seq

        # remove padded items
        trg_flat = labels.flatten()
        non_pad_mask = trg_flat != self.pad_token
        labels_all = torch.masked_select(trg_flat, non_pad_mask)
        x = self.remove_pad_3d(x, non_pad_mask)

        # Compute predictions probs
        x = self.pre(x)

        # prepare outputs for HF trainer
        if self.hf_format:
            loss = self.loss(x, labels_all)
            return {
                "loss": loss,
                "labels": labels_all,
                "predictions": x,
                "pred_metadata": {},
                "model_outputs": [],
            }
            # TODO: Add model_outputs and metadata

        return x

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(
            inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor)
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))
        return out_tensor


class _ItemPredictionTask(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
    ):
        """
        Predict the interacted item-id probabilities.
        - During inference, the task consists of predicting the next item.
        - During training, the class supports the following Language modeling tasks:
            Causal LM, Masked LM, Permutation LM and Replacement Token Detection
        Parameters:
        -----------
            input_size:
            target_dim:
            weight_tying:
            item_embedding_table:
            softmax_temperature:
        """
        super().__init__()
        self.input_size = input_size
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        if self.weight_tying:
            self.output_layer_bias = torch.nn.Parameter(torch.Tensor(self.target_dim))
            torch.nn.init.zeros_(self.output_layer_bias)
        else:
            self.output_layer = torch.nn.Linear(self.input_size[-1], self.target_dim)

    def forward(self, inputs):
        if self.weight_tying:
            logits = torch.nn.functional.linear(
                inputs,
                weight=self.item_embedding_table.weight,
                bias=self.output_layer_bias,
            )
        else:
            logits = self.output_layer(inputs)

        if self.softmax_temperature:
            # Softmax temperature to reduce model overconfidence
            # and better calibrate probs and accuracy
            logits = torch.div(logits, self.softmax_temperature)

        predictions = self.log_softmax(logits)

        return predictions

    def _get_name(self):
        return "ItemPredictionTask"


class Head(torch.nn.Module):
    def __init__(
        self,
        body: SequentialBlock,
        prediction_tasks: Optional[Union[List[PredictionTask], PredictionTask]] = None,
        task_towers: Optional[Union[BuildableBlock, Dict[str, BuildableBlock]]] = None,
        task_weights=None,
        body_output_size=None,
        inputs=None,
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
        self.build(body_output_size or body.output_size(), inputs=inputs)

    def build(self, input_size, inputs=None, device=None):
        if device:
            self.to(device)
        for task in self.prediction_tasks.values():
            task.build(self.body, input_size, inputs=inputs, device=device)
        self.input_size = input_size

    @classmethod
    def from_schema(cls, schema: DatasetSchema, body, task_weights=None, input_size=None):
        if task_weights is None:
            task_weights = {}
        to_return = cls(body, body_output_size=input_size)

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

    def add_task(self, task: PredictionTask, task_weight=1):
        key = task.target_name
        self.prediction_tasks[key] = task
        if task_weight:
            self._task_weights[key] = task_weight

        return self

    def pop_labels(self, inputs: Dict[Text, torch.Tensor]):
        outputs = {}
        for name in self.prediction_tasks.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def forward(self, logits: torch.Tensor, **kwargs):
        outputs = {}

        for name, task in self.prediction_tasks.items():
            outputs[name] = task(logits, **kwargs)

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(self, body_outputs, targets, **kwargs) -> torch.Tensor:
        losses = []

        for name, task in self.prediction_tasks.items():
            loss = task.compute_loss(body_outputs, targets, **kwargs)
            losses.append(loss * self._task_weights[name])

        return torch.stack(losses).mean()

    def calculate_metrics(
        self, block_outputs, targets, mode="val"
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        metrics = {}

        for name, task in self.prediction_tasks.items():
            metrics[name] = task.calculate_metrics(block_outputs, targets, mode=mode)

        return _output_metrics(metrics)

    def compute_metrics(self, mode=None):
        def name_fn(x):
            return "_".join([mode, x]) if mode else x

        metrics = {
            name_fn(name): task.compute_metrics() for name, task in self.prediction_tasks.items()
        }

        return _output_metrics(metrics)

    def reset_metrics(self):
        for task in self.prediction_tasks.values():
            task.reset_metrics()


class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types

        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics
