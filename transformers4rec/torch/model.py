from typing import Dict, Union

import torch


class Model(torch.nn.Module):
    def __init__(self, *head, **kwargs):
        super().__init__()
        self.heads = head

    def forward(self, logits: torch.Tensor, **kwargs):
        raise NotImplementedError

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
