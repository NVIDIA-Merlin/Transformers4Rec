import inspect
from typing import Optional, Union

import torch
from transformers import GPT2Model, PreTrainedModel

from ...config.transformer import T4RecConfig, transformer_registry
from ..masking import PermutationLanguageModeling
from .base import Block
from .masking import MaskingBlock

TransformerBody = Union[str, PreTrainedModel, T4RecConfig]


class TransformerBlock(Block):
    """
    Class to support HF Transformers for session-based and sequential-based recommendation models.

    Parameters:
    -----------
        transformer: TransformerBody to set the HF model.
        The model returns the hidden representation of the sequence of interaction embeddings

        masking: Optional MaskingBlock, the block is required when :
                    - The HF model signature needs additional masks
                    -> e.g: For PLM task, XLNet needs target_mapping and perm_mask.

    """

    def __init__(
        self,
        transformer: TransformerBody = "xlnet",
        masking: Optional[MaskingBlock] = None,
    ):
        super().__init__()

        if isinstance(transformer, T4RecConfig):
            transformer = transformer.to_torch_model()

        self.transformer = transformer
        self.masking = masking

    @classmethod
    def from_registry(
        cls,
        transformer: str,
        d_model: int,
        n_head: int,
        n_layer: int,
        total_seq_length: int,
        masking: Optional[MaskingBlock] = None,
    ):
        transformer = transformer_registry.parse(transformer).build(
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            total_seq_length=total_seq_length,
        )

        return cls(transformer, masking)

    def forward(self, inputs, **kwargs):
        """
        Transformer Models
        """
        if type(self.transformer) is GPT2Model:
            seq_len = inputs.shape[1]
            # head_mask has shape n_layer x batch x n_heads x N x N
            head_mask = (
                torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=self.device))
                .view(1, 1, 1, seq_len, seq_len)
                .repeat(self.transformer.config.num_hidden_layers, 1, 1, 1, 1)
            )

            model_outputs = self.transformer(
                inputs_embeds=inputs,
                head_mask=head_mask,
            )

        elif isinstance(self.masking, PermutationLanguageModeling):

            check = all(
                param in inspect.signature(self.transformer.forward).parameters
                for param in ["target_mapping", "perm_mask"]
            )
            if not check:
                raise ValueError(
                    "Permutation Language Modeling requires the parameters"
                    "['target_mapping', 'perm_mask'] in the transformer signature"
                )

            model_outputs = self.transformer(
                inputs_embeds=inputs,
                target_mapping=self.masking_block.masking.plm_target_mapping,
                perm_mask=self.masking_block.masking.plm_perm_mask,
            )

        else:
            model_outputs = self.transformer(inputs_embeds=inputs)

        pos_emb_hidden = model_outputs[0]

        # TODO: store the attention outputs for meta-data logging
        return pos_emb_hidden

    def _get_name(self):
        return "TansformerBlock"

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
