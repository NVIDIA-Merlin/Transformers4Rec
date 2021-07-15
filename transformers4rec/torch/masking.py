from dataclasses import dataclass

import torch
from torch import nn

from ..utils.masking import MaskSequence as _MaskSequence
from ..utils.registry import Registry

masking_tasks = Registry("torch.masking")


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
    masked_input: torch.tensor
    masked_label: torch.tensor
    mask_schema: torch.tensor
    plm_target_mapping: torch.tensor = None
    plm_perm_mask: torch.tensor = None


class MaskSequence(_MaskSequence, nn.Module):
    def __init__(self, hidden_size: int, pad_token: int = 0):
        super().__init__(hidden_size, pad_token)

    def forward(self, pos_emb, itemid_seq, training) -> MaskedSequence:
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
