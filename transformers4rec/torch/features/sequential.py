from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ProcessedSequence:
    """
    Class to store the Tensor resulting from the aggregation of a group of
    categorical and continuous variables defined in the same featuremap
    Parameters
    ----------
        name: name to give to the featuregroup
        values: the aggregated pytorch tensor
        metadata: list of columns names to log as metadata
    """

    name: str
    values: torch.Tensor
    metadata: List[str]
