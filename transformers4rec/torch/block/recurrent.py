from typing import Union

import torch
from torch import nn
from transformers import PreTrainedModel

from ..transformer import T4RecConfig
from ..typing import MaskedOutput, MaskSequence, ProcessedSequence
from .base import BuildableBlock, SequentialBlock

RecurrentBody = Union[str, PreTrainedModel, T4RecConfig, nn.Module]


class RecurrentBlock(BuildableBlock):
    def __init__(
        self, masking: Union[str, MaskSequence] = "clm", body: RecurrentBody = "xlnet"
    ) -> None:
        super().__init__()
        if isinstance(body, T4RecConfig):
            body = body.to_model()

        self.masking = masking
        self.body = body

    def build(self, input_shape) -> "_RecurrentBlock":
        pass


class _RecurrentBlock(SequentialBlock):
    def parse_inputs(self, inputs):
        if isinstance(inputs, ProcessedSequence):
            parsed = inputs.values
        elif isinstance(inputs, MaskedOutput):
            parsed = inputs.masked_input
        elif isinstance(inputs, torch.Tensor):
            parsed = inputs
        else:
            raise ValueError("Unrecognized inputs")

    def forward(self, inputs):
        pass
