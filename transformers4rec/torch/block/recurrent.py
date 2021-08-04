from typing import Union

import torch
from transformers import PreTrainedModel

from ...config.transformer import T4RecConfig
from .. import masking
from ..features.embedding import EmbeddingFeatures
from ..typing import MaskSequence
from .base import BuildableBlock, SequentialBlock

RecurrentBody = Union[str, PreTrainedModel, T4RecConfig, torch.nn.Module]


class RecurrentBlock(BuildableBlock):
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

    def build(self, input_shape, parents) -> torch.nn.Module:
        embeddings = SequentialBlock.get_children_by_class_name(
            parents, "EmbeddingFeatures", "SequentialEmbeddingFeatures"
        )

        return _RecurrentBlock(self.masking, self.body, embeddings)


class _RecurrentBlock(torch.nn.Module):
    def __init__(self, masking, body, embedding_module: EmbeddingFeatures):
        super().__init__()
        self.embedding_module = embedding_module
        self.body = body
        self.masking = masking

    def forward(self, inputs, training=True, **kwargs):
        item_seq = self.embedding_module[0].item_seq
        self.masking.set_masking_schema(item_seq, training=training)
        inputs = self.masking(inputs, for_inputs=True)
        # TODO Call body here

        return inputs

    @property
    def masking(self):
        return self._masking

    @masking.setter
    def masking(self, value):
        if value:
            self._masking = masking.masking_registry.parse(value)
        else:
            self._masking = None

    def _get_name(self):
        return "RecurrentBlock"

    # TODO: Implement output-size based on the body
    # def output_size(self):
    #     if len(input_shape) == 3:
    #         return torch.Size([input_shape[0], input_shape[1], dense_output_size])
    #
    #     return torch.Size([input_shape[0], dense_output_size])
    #
    # def forward_output_size(self, input_size):
    #     if len(input_size) == 3:
    #         return torch.Size([input_size[0], input_size[1], dense_output_size])
    #
    #     return torch.Size([input_size[0], dense_output_size])
