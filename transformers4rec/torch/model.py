from typing import Dict, Union

import torch

from transformers4rec.torch.typing import TensorOrTabularData


class Model(torch.nn.Module):
    def __init__(self, *head, **kwargs):
        super().__init__()
        self.heads = head

    def forward(self, inputs: TensorOrTabularData, **kwargs):
        raise NotImplementedError
        # TODO: Optimize this
        # for head in self.heads:
        #     outputs = head(head.body(inputs))

    def compute_loss(self, block_outputs, targets, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def calculate_metrics(
        self, block_outputs, targets, mode="val"
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        raise NotImplementedError

    def compute_metrics(self, mode=None):
        raise NotImplementedError

    def reset_metrics(self):
        raise NotImplementedError
