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

import tensorflow as tf
import transformers
from transformers import PretrainedConfig, TFPreTrainedModel

from ...config.transformer import T4RecConfig, transformer_registry
from ..masking import MaskSequence
from ..utils.tf_utils import (
    get_tf_main_layer,
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)
from .base import Block

TransformerBody = Union[TFPreTrainedModel, PretrainedConfig, tf.keras.layers.Layer]


class TransformerPrepare(tf.keras.layers.Layer):
    def __init__(self, transformer, masking, **kwargs):
        super().__init__(**kwargs)
        self.transformer = transformer
        self.masking = masking

    def call(self, inputs_embeds, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class TransformerBlock(Block):
    """
    Class to support HF Transformers for session-based and sequential-based recommendation models.

    Parameters
    ----------
    transformer: TransformerBody
        The T4RecConfig, The pre-trained HF model or the custom keras layer TF*MainLayer,
        related to specific transformer architecture.
    masking:
        Needed when masking is applied on the inputs.
    """

    TRANSFORMER_TO_PREPARE: Dict[Type[TFPreTrainedModel], Type[TransformerPrepare]] = {}

    # TODO: Add {GPT2Model: GPT2Prepare}

    def __init__(
        self,
        transformer: TransformerBody,
        masking: Optional[MaskSequence] = None,
        prepare_module: Optional[Type[TransformerPrepare]] = None,
        output_fn=lambda model_outputs: model_outputs[0],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.transformer: TFPreTrainedModel
        if isinstance(transformer, T4RecConfig):
            self.transformer = get_tf_main_layer(transformer.to_huggingface_tf_model())
        elif isinstance(transformer, PretrainedConfig):
            model_cls = transformers.TF_MODEL_MAPPING[transformer.__class__]
            self.transformer = get_tf_main_layer(model_cls(transformer))
        elif isinstance(transformer, TFPreTrainedModel):
            self.transformer = get_tf_main_layer(transformer)
        else:
            self.transformer = transformer

        if masking:
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
            self.prepare_module = prepare_module(transformer, masking)
        self.output_fn = output_fn

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self, config, ["transformer", "prepare_module", "masking"]
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(
            config, ["transformer", "prepare_module", "masking"]
        )

        return super().from_config(config)

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
        _transformer: TFPreTrainedModel = transformer_registry.parse(transformer).build(
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            total_seq_length=total_seq_length,
        )

        return cls(_transformer, masking=masking)

    def call(self, inputs_embeds: tf.Tensor, **kwargs):
        """

        Parameters
        ----------
        inputs_embeds `tf.Tensor` of shape ({0}, hidden_size)`
            An embedded representation of a sequence.

        Returns
        -------
        `tf.Tensor`
        """
        transformer_kwargs = {"inputs_embeds": inputs_embeds}
        if self.prepare_module:
            transformer_kwargs = self.prepare_module(inputs_embeds)
        if self.masking:
            masking_kwargs = self.masking.transformer_arguments
            if masking_kwargs:
                transformer_kwargs.update(masking_kwargs)

        filtered_transformer_kwargs = {}
        for param in inspect.signature(self.transformer.call).parameters:
            if param in transformer_kwargs:
                filtered_transformer_kwargs[param] = transformer_kwargs[param]

        # In HF the call accept inputs as a dictionary containing all needed tensors
        model_outputs = self.transformer(filtered_transformer_kwargs)
        outputs = self.output_fn(model_outputs)

        # TODO: store the attention outputs for meta-data logging
        return outputs
