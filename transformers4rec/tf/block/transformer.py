import inspect
from typing import Any, Dict, Optional, Type, Union

import tensorflow as tf
from transformers import PretrainedConfig, TFPreTrainedModel

from ...config.transformer import T4RecConfig, transformer_registry
from ..masking import MaskSequence
from .base import Block

TransformerBody = Union[TFPreTrainedModel, T4RecConfig]


class TransformerPrepare(tf.keras.layers.Layer):
    def __init__(self, transformer, masking, **kwargs):
        super().__init__(**kwargs)
        self.transformer = transformer
        self.masking = masking

    def call(self, inputs_embeds, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()


class TransformerBlock(Block):
    TRANSFORMER_TO_PREPARE: Dict[PretrainedConfig, Type[TransformerPrepare]] = {}

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

        if isinstance(transformer, T4RecConfig):
            transformer = transformer.to_huggingface_tf_model()

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
        self.prepare_module = prepare_module
        if not prepare_module and self.transformer in self.TRANSFORMER_TO_PREPARE:
            prepare_module = self.TRANSFORMER_TO_PREPARE[self.transformer]
        if prepare_module:
            self.prepare_module = prepare_module(transformer, masking)
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
        transformer = transformer_registry.parse(transformer).build(
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            total_seq_length=total_seq_length,
        )

        return cls(transformer, masking)

    def call(self, inputs_embeds, **kwargs):
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
