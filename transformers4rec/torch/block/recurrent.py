from typing import Union

from transformers import PreTrainedModel
from torch import nn

from .base import BuildableBlock
from ..masking import MaskSequence
from ..transformer import T4RecConfig

RecurrentBody = Union[str, PreTrainedModel, T4RecConfig, nn.Module]


class RecurrentBlock(BuildableBlock):
    def __init__(self,
                 masking: Union[str, MaskSequence] = "clm",
                 body: RecurrentBody = "xlnet") -> None:
        super().__init__()
        if isinstance(body, T4RecConfig):
            body = body.to_model()

        self.masking = masking
        self.body = body

    def build(self, input_shape) -> "RecurrentBlock":
        pass
