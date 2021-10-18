#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect
from typing import Any, Dict, Optional, Type, Union

import torch
import transformers
from transformers import GPT2Model, PretrainedConfig, PreTrainedModel

from ...config.transformer import T4RecConfig, transformer_registry
from ..masking import MaskSequence
from ..utils.torch_utils import MappingTransformerMasking
from .base import BlockBase

TransformerBody = Union[PreTrainedModel, PretrainedConfig]


class TransformerPrepare(torch.nn.Module):
    def __init__(self, transformer, masking):
        super().__init__()
        self.transformer = transformer
        self.masking = masking

    def forward(self, inputs_embeds) -> Dict[str, Any]:
        raise NotImplementedError()


class GPT2Prepare(TransformerPrepare):
    def forward(self, inputs_embeds) -> Dict[str, Any]:
        seq_len = inputs_embeds.shape[1]
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = (
            torch.tril(
                torch.ones((seq_len, seq_len), dtype=torch.uint8, device=inputs_embeds.device)
            )
            .view(1, 1, 1, seq_len, seq_len)
            .repeat(self.transformer.config.num_hidden_layers, 1, 1, 1, 1)
        )

        return {"inputs_embeds": inputs_embeds, "head_mask": head_mask}


class TransformerBlock(BlockBase):
    """
    Class to support HF Transformers for session-based and sequential-based recommendation models.

    Parameters
    ----------
    transformer: TransformerBody
        The T4RecConfig or a pre-trained HF object related to specific transformer architecture.
    masking:
        Needed when masking is applied on the inputs.
    """

    TRANSFORMER_TO_PREPARE: Dict[Type[PreTrainedModel], Type[TransformerPrepare]] = {
        GPT2Model: GPT2Prepare
    }

    def __init__(
        self,
        transformer: TransformerBody,
        masking: Optional[MaskSequence] = None,
        prepare_module: Optional[Type[TransformerPrepare]] = None,
        output_fn=lambda model_outputs: model_outputs[0],
    ):
        super().__init__()

        self.transformer: PreTrainedModel
        if isinstance(transformer, T4RecConfig):
            self.transformer = transformer.to_huggingface_torch_model()
        elif isinstance(transformer, PretrainedConfig):
            model_cls = transformers.MODEL_MAPPING[transformer.__class__]
            self.transformer = model_cls(transformer)
        else:
            self.transformer = transformer

        if masking:
            # check for the four default masking
            if (masking.__class__ in MappingTransformerMasking.DEFAULT_MASKING) and (
                masking.__class__
                not in getattr(
                    MappingTransformerMasking,
                    self.transformer.config_class.__name__,  # type: ignore
                    [masking.__class__],
                )
            ):
                raise ValueError(
                    f"{masking.__class__.__name__} is not supported by: "  # type: ignore
                    f"the {self.transformer.config_class.__name__} architecture"  # type: ignore
                )

            required = list(masking.transformer_required_arguments().keys())
            check = all(
                param in inspect.signature(self.transformer.forward).parameters
                for param in required
            )
            if not check:
                raise ValueError(
                    f"{masking.__class__.__name__} requires the parameters: "
                    f"{', '.join(required)} "
                    f"in the {type(self.transformer)} signature"
                )

        self.masking = masking
        self.prepare_module: Optional[TransformerPrepare] = None
        if not prepare_module and type(self.transformer) in self.TRANSFORMER_TO_PREPARE:
            prepare_module = self.TRANSFORMER_TO_PREPARE[type(self.transformer)]
        if prepare_module:
            self.prepare_module = prepare_module(self.transformer, self.masking)
        self.output_fn = output_fn

    @classmethod
    def from_registry(
        cls,
        transformer: str,
        d_model: int,
        n_head: int,
        n_layer: int,
        total_seq_length: int,
        masking: Optional[MaskSequence] = None,
    ):
        """
        Load the HF transformer architecture based on its name

        Parameters
        ----------
        transformer: str
            Name of the Transformer to use. Possible values are :
            ["reformer", "gtp2", "longformer", "electra", "albert", "xlnet"]
        d_model: int
            size of hidden states for Transformers
        n_head:
            Number of attention heads for Transformers
        n_layer: int
            Number of layers for RNNs and Transformers"
        total_seq_length: int
            The maximum sequence length
        """
        _transformer = transformer_registry.parse(transformer).build(
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            total_seq_length=total_seq_length,
        )

        return cls(_transformer, masking)

    def forward(self, inputs_embeds, **kwargs):
        """
        Transformer Models
        """
        transformer_kwargs = {"inputs_embeds": inputs_embeds}
        if self.prepare_module:
            transformer_kwargs = self.prepare_module(inputs_embeds)
        if self.masking:
            masking_kwargs = self.masking.transformer_arguments
            if masking_kwargs:
                transformer_kwargs.update(masking_kwargs)

        filtered_transformer_kwargs = {}
        for param in inspect.signature(self.transformer.forward).parameters:
            if param in transformer_kwargs:
                filtered_transformer_kwargs[param] = transformer_kwargs[param]
        model_outputs = self.transformer(**filtered_transformer_kwargs)
        outputs = self.output_fn(model_outputs)

        # TODO: store the attention outputs for meta-data logging
        return outputs

    def _get_name(self):
        return "TansformerBlock"

    def forward_output_size(self, input_size):
        assert len(input_size) == 3
        return torch.Size([input_size[0], input_size[1], self.transformer.config.hidden_size])
