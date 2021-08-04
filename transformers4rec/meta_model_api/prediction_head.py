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
from typing import Any, Callable, Dict, List, Optional

import torch
from feature_process import FeatureGroupProcess
from torch import nn
from torch.nn import functional as F


class ItemPrediction(torch.nn.Module):
    def __init__(
        self,
        task,
        loss,
        metrics=None,
        body: Optional[torch.nn.Module] = None,
        forward_to_prediction_fn=lambda x: x,
        pre: Optional[torch.nn.Module] = None,
        mf_constrained_embeddings: bool = True,
        feature_process: FeatureGroupProcess = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.forward_to_prediction_fn = forward_to_prediction_fn
        self.set_metrics(metrics)
        self.loss = loss
        self.body = body
        self.pre = pre
        self.mf_constrained_embeddings = mf_constrained_embeddings
        self.target_dim = task.dimension
        self.item_name = task.label_column
        self.device = device

        if self.mf_constrained_embeddings:
            self.output_layer_bias = nn.Parameter(torch.Tensor(self.target_dim)).to(self.device)
            nn.init.zeros_(self.output_layer_bias)

            categorical_item = [x for x in feature_process.categoricals if x.name == self.item_name]
            if not categorical_item:
                raise ValueError(
                    "When mf_constrained_embeddings is enbaled, feature_process class have to contain itemid column "
                )
            self.item_embedding_table = categorical_item[0].table

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

        if self.mf_constrained_embeddings:
            x = F.linear(
                x,
                weight=self.item_embedding_table.weight,
                bias=self.output_layer_bias,
            )
        if self.pre:
            x = self.pre(x)
        return x

    def set_metrics(self, metrics):
        self.metrics = torch.nn.ModuleList(metrics)

    def compute_loss(
        self, inputs, targets, training: bool = False, compute_metrics=False
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
        raise NotImplementedError

    def compute_metrics(self):
        return {f"{metric.__class__.__name__.lower()}": metric.compute() for metric in self.metrics}

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()
