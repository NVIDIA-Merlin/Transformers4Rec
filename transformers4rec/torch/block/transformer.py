from typing import Union

import torch
from transformers import GPT2Model, PreTrainedModel, XLNetModel

from ...config.transformer import T4RecConfig, transformer_registry
from .. import masking
from ..features.embedding import EmbeddingFeatures
from ..masking import PermutationLanguageModeling
from ..typing import MaskSequence
from .base import BuildableBlock, SequentialBlock

TransformerBody = Union[str, PreTrainedModel, T4RecConfig]


class TransformerBlock(BuildableBlock):
    """
    Class to support HF Transformers for session-based and sequential-based recommendation models.
    """

    def __init__(
        self,
        masking: Union[str, MaskSequence] = "clm",
        body: TransformerBody = "xlnet",
        d_model: int = None,
        n_head: int = None,
        n_layer: int = None,
        total_seq_length: int = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        if isinstance(body, T4RecConfig):
            body = body.to_torch_model()

        elif isinstance(body, str):
            # check that required arguments to build the model are not None
            assert all(
                arg is not None for arg in [d_model, n_head, n_layer, total_seq_length]
            ), "Please provide all required arguments to load the model '{}'" ": {}".format(
                body,
                {
                    "d_model": d_model,
                    "n_head": n_head,
                    "n_layer": n_layer,
                    "total_seq_length": total_seq_length,
                },
            )
            body = (
                transformer_registry.parse(body)
                .build(
                    d_model=d_model,
                    n_head=n_head,
                    n_layer=n_layer,
                    total_seq_length=total_seq_length,
                )
                .to_torch_model()
            )

        self.hidden_size = body.config.hidden_size
        self.input_shape = self.hidden_size
        self.n_layer = n_layer
        self.device = device
        self.masking = masking
        self.body = body

    @property
    def masking(self):
        return self._masking

    @masking.setter
    def masking(self, value):
        if value:
            self._masking = masking.masking_registry.parse(value)(
                hidden_size=self.hidden_size, device=self.device
            )
        else:
            self._masking = None

    def build(self, input_shape, parents) -> torch.nn.Module:
        embeddings = SequentialBlock.get_children_by_class_name(
            parents, "EmbeddingFeatures", "SequentialEmbeddingFeatures"
        )
        return _TransformerBlock(self.masking, self.body, embeddings)

    def to_torch_module(self, input_module):
        return self.build(input_shape=self.hidden_size, parents=[input_module])


class _TransformerBlock(torch.nn.Module):
    def __init__(self, masking, body, embedding_module: EmbeddingFeatures):
        super().__init__()
        self.embedding_module = embedding_module
        self.body = body
        self.masking = masking
        self.device = self.masking.device

    def forward(self, inputs, training=True, **kwargs):
        if self.masking:
            item_seq = self.embedding_module[0].item_seq
            self.masking.set_masking_targets(item_ids=item_seq, training=training)
            inputs = self.masking(inputs)

        """
        Transformer Models
        """
        if type(self.body) is GPT2Model:
            seq_len = inputs.shape[1]
            # head_mask has shape n_layer x batch x n_heads x N x N
            head_mask = (
                torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=self.device))
                .view(1, 1, 1, seq_len, seq_len)
                .repeat(self.body.config.num_hidden_layers, 1, 1, 1, 1)
            )

            model_outputs = self.body(
                inputs_embeds=inputs,
                head_mask=head_mask,
            )

        elif type(self.body) is XLNetModel and isinstance(
            self.masking, PermutationLanguageModeling
        ):
            model_outputs = self.body(
                inputs_embeds=inputs,
                target_mapping=self.masking.plm_target_mapping,
                perm_mask=self.masking.plm_perm_mask,
            )

        else:
            model_outputs = self.body(inputs_embeds=inputs)

        pos_emb_hidden = model_outputs[0]

        # TO-ADD: store the attention outputs for meta-data logging
        return pos_emb_hidden

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
