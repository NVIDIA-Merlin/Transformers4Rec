from typing import Union

import torch
from transformers import PreTrainedModel

from ...config.transformer import T4RecConfig
from .. import masking
from ..typing import MaskSequence

# from .base import BuildableBlock, SequentialBlock

RecurrentBody = Union[str, PreTrainedModel, T4RecConfig, torch.nn.Module]


class RecurrentBlock(torch.nn.Module):
    def __init__(
        self, masking: Union[str, MaskSequence] = "clm", body: RecurrentBody = "xlnet"
    ) -> None:
        super().__init__()
        if isinstance(body, T4RecConfig):
            body = body.to_torch_model()

        self.masking = masking
        self.body = body

    @property
    def masking(self):
        return self._masking

    @masking.setter
    def masking(self, value):
        if value:
            self._masking = masking.masking_registry.parse(value)
        else:
            self._masking = None

    def forward(self, inputs, training=True, **kwargs):
        # mask_out = self.masking(inputs, itemid_seq, training=training)
        pass

    def build(self, input_shape) -> torch.nn.Module:
        if isinstance(self.body, torch.nn.Module):
            return self.body
