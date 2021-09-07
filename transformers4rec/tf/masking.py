import tensorflow as tf

from ..utils.registry import Registry

masking_registry = Registry("tf.masking")

MaskingSchema = tf.Tensor
MaskedTargets = tf.Tensor


class MaskSequence(tf.keras.layers.Layer):
    def __init__(self, padding_idx: int = 0, **kwargs):
        self.padding_idx = padding_idx
        super(MaskSequence, self).__init__(**kwargs)
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


@masking_registry.register_with_multiple_names("plm", "permutation")
class PermutationLanguageModeling(MaskSequence):
    pass


@masking_registry.register_with_multiple_names("rtd", "replacement")
class ReplacementLanguageModeling(MaskSequence):
    pass
