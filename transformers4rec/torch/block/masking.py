from typing import Union

import torch

from ..features.sequential import SequentialEmbeddingFeatures, SequentialTabularFeatures
from ..masking import MaskSequence, masking_registry
from .base import Block, SequentialBlock


class MaskingBlock(Block):
    """
    Class to create masked inputs and targets for Language-Modeling tasks adapted to
    sequential-based recommendation models.

    Params:
    ------
        - masking: MaskSequence class that computes the mask schema
                    and apply it to inputs and targets
        - input_module: Tabular module that processes the inputs containing item-id column
    """

    def __init__(
        self, masking, input_module: Union[SequentialEmbeddingFeatures, SequentialTabularFeatures]
    ):
        super().__init__()

        assert isinstance(
            masking, MaskSequence
        ), "masking needs to be an instance of MaskSequence class"
        self.masking = masking

        embeddings = SequentialBlock.get_children_by_class_name(
            [input_module], "EmbeddingFeatures", "SequentialEmbeddingFeatures"
        )

        assert embeddings, "The input_module should includes `SequentialEmbeddingFeatures`"
        self.embeddings = embeddings[0]

        if not self.embeddings.item_id:
            raise ValueError(
                "Please provide the tagged item-id column in the schema of the input module."
                " The sequence of item-ids is necessary to apply masking"
            )

    @classmethod
    def from_registry(
        cls,
        input_module,
        masking: str,
        hidden_size: int,
        device: Union[str, torch.device] = "cpu",
        pad_token: int = 0,
        **kwargs
    ):
        masking = masking_registry.parse(masking)(
            hidden_size=hidden_size, pad_token=pad_token, device=device, **kwargs
        )
        return cls(masking, input_module)

    def forward(self, inputs, training=True):
        return self.masking(inputs, item_ids=self.embeddings.item_seq, training=training)
