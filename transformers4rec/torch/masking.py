from dataclasses import dataclass

import torch
from torch import nn

from ..utils.registry import Registry

masking_tasks = Registry("masking_tasks")


class MaskSequence(object):
    """
    Module to prepare masked data for LM tasks

    Parameters:
    ----------
        pad_token: index of padding.
        device: either 'cpu' or 'cuda' device.
        hidden_size: dimension of the interaction embeddings
    """

    def __init__(self, hidden_size: int, pad_token: int = 0, device: str = 'cuda'):
        super(MaskSequence, self).__init__()
        self.pad_token = pad_token
        self.device = device
        self.hidden_size = hidden_size


@dataclass
class MaskOutput:
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


@masking_tasks.register_with_multiple_names("clm", "causal")
class CausalLanguageModeling(MaskSequence, nn.Module):
    pass


@masking_tasks.register_with_multiple_names("mlm", "masked")
class MaskedLanguageModeling(MaskSequence, nn.Module):
    pass


@masking_tasks.register_with_multiple_names("mlm", "permutation")
class PermutationLanguageModeling(MaskSequence, nn.Module):
    pass


@masking_tasks.register_with_multiple_names("mlm", "replacement")
class ReplacementLanguageModeling(MaskSequence, nn.Module):
    pass
