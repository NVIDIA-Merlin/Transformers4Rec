from typing import Union

import torch
from transformers import GPT2Model, PreTrainedModel, XLNetModel

from ...config.transformer import T4RecConfig
from .. import masking
from ..features.embedding import EmbeddingFeatures
from ..masking import PermutationLanguageModeling
from ..typing import MaskSequence
from .base import BuildableBlock, SequentialBlock

RecurrentBody = Union[str, PreTrainedModel, T4RecConfig, torch.nn.Module]


class RecurrentBlock(BuildableBlock):
    def __init__(
        self, hidden_size, masking: Union[str, MaskSequence] = "clm", body: RecurrentBody = "xlnet"
    ) -> None:
        super().__init__()
        if isinstance(body, T4RecConfig):
            body = body.to_torch_model()
        # TODO: remove the hidden_size from paramters and directly compute it from body class
        self.hidden_size = hidden_size
        self.masking = masking
        self.body = body

    @property
    def masking(self):
        return self._masking

    @masking.setter
    def masking(self, value):
        if value:
            self._masking = masking.masking_registry.parse(value)(hidden_size=self.hidden_size)
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
        if self.masking:
            item_seq = self.embedding_module[0].item_seq
            self.masking.set_masking_targets(item_ids=item_seq, training=training)
            inputs = self.masking(inputs)

        if not isinstance(self.body, PreTrainedModel):  # Checks if its not a transformer
            # compute output through RNNs
            results = self.body(input=inputs)
            if type(results) is tuple or type(results) is list:
                pos_emb_hidden = results[0]
            else:
                pos_emb_hidden = results
            model_outputs = (None,)

        else:
            """
            Transformer Models
            """
            if type(self.body) is GPT2Model:
                seq_len = inputs.shape[1]
                # head_mask has shape n_layer x batch x n_heads x N x N
                head_mask = (
                    torch.tril(
                        torch.ones((seq_len, seq_len), dtype=torch.uint8, device=self.device)
                    )
                    .view(1, 1, 1, seq_len, seq_len)
                    .repeat(self.n_layer, 1, 1, 1, 1)
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

        # TODO: store the attention outputs for meta-data logging
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
