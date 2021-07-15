from dataclasses import dataclass

import tensorflow as tf

from ..utils.masking import MaskSequence as _MaskSequence
from ..utils.registry import Registry

masking_tasks = Registry("tf.masking")


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
    def __init__(self, hidden_size: int, pad_token: int = 0, **kwargs):
        super().__init__(hidden_size, pad_token)
        tf.keras.layers.Layer.__init__(**kwargs)

    def call(self, pos_emb, itemid_seq, training, **kwargs) -> MaskedSequence:
        raise NotImplementedError()


@masking_tasks.register_with_multiple_names("clm", "causal")
class CausalLanguageModeling(MaskSequence):
    pass


@masking_tasks.register_with_multiple_names("mlm", "masked")
class MaskedLanguageModeling(MaskSequence):
    pass


@masking_tasks.register_with_multiple_names("mlm", "permutation")
class PermutationLanguageModeling(MaskSequence):
    pass


@masking_tasks.register_with_multiple_names("mlm", "replacement")
class ReplacementLanguageModeling(MaskSequence):
    pass
