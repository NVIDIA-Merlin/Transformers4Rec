from dataclasses import dataclass

import tensorflow as tf

from ..utils.masking import MaskSequence as _MaskSequence
from ..utils.registry import Registry

masking_registry = Registry("tf.masking")

MaskingSchema = tf.Tensor
MaskedTargets = tf.Tensor


@dataclass
class MaskedSequence:
    """
    Class to store the masked inputs, labels and boolean masking scheme
    resulting from the related LM task.

    Parameters
    ----------
        masked_input: the masked interactions tensor
        masked_label: the masked sequence of item ids
        mask_schema: the boolean mask indicating the position of masked items
        plm_target_mapping: boolean mapping needed by XLNET-PLM
        plm_perm_mask:  boolean mapping needed by XLNET-PLM
    """

    masked_input: tf.Tensor
    masked_label: tf.Tensor
    mask_schema: tf.Tensor
    plm_target_mapping: tf.Tensor = None
    plm_perm_mask: tf.Tensor = None


class MaskSequence(_MaskSequence, tf.keras.layers.Layer):
    def __init__(self, pad_token: int = 0, **kwargs):
        super().__init__(None, pad_token)
        tf.keras.layers.Layer.__init__(**kwargs)
        self.schema = None

    def _compute_masked_targets(self, item_ids: tf.Tensor, training=False) -> MaskingSchema:
        # TODO: assert inputs has 3 dims
        raise NotImplementedError

    def compute_masked_targets(self, item_ids: tf.Tensor, training=False, return_targets=False):
        self.mask_schema, self.masked_targets = self._compute_masked_targets(
            item_ids, training=training
        )
        if return_targets:
            return self.masked_targets

    def apply_mask_to_inputs(self, inputs: tf.Tensor, schema: MaskingSchema):
        raise NotImplementedError

    def apply_mask_to_targets(self, target: tf.Tensor, schema: MaskingSchema):
        raise NotImplementedError

    def call(self, inputs: tf.Tensor, item_ids: tf.Tensor, training=False, **kwargs) -> tf.Tensor:
        self.compute_masked_targets(item_ids=item_ids, training=training)
        return self.apply_mask_to_inputs(inputs, self.mask_schema)

    def transformer_required_arguments(self):
        return {}

    def transformer_optional_arguments(self):
        return {}

    def transformer_arguments(self):
        return {**self.transformer_required_arguments(), **self.transformer_optional_arguments()}

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        return super().build(input_shape)


@masking_registry.register_with_multiple_names("clm", "causal")
class CausalLanguageModeling(MaskSequence):
    pass


@masking_registry.register_with_multiple_names("mlm", "masked")
class MaskedLanguageModeling(MaskSequence):
    pass


@masking_registry.register_with_multiple_names("mlm", "permutation")
class PermutationLanguageModeling(MaskSequence):
    pass


@masking_registry.register_with_multiple_names("mlm", "replacement")
class ReplacementLanguageModeling(MaskSequence):
    pass
