import sys
import typing
from typing import Dict

import torch

if sys.version_info < (3, 7):
    ForwardRef = typing._ForwardRef  # pylint: disable=protected-access
else:
    ForwardRef = typing.ForwardRef

# TODO: Make this more generic and work with multi-hot features
TabularData = Dict[str, torch.Tensor]
TensorOrTabularData = typing.Union[torch.Tensor, TabularData]

TabularModule = ForwardRef("transformers4rec.torch.tabular.TabularModule")

ProcessedSequence = ForwardRef("transformers4rec.torch.features.ProcessedSequence")

FeatureAggregator = ForwardRef("transformers4rec.torch.aggregator.FeatureAggregation")

MaskSequence = ForwardRef("transformers4rec.torch.masking.MaskSequence")
MaskedSequence = ForwardRef("transformers4rec.torch.masking.MaskedSequence")

Block = ForwardRef("transformers4rec.torch.block.base.Block")
SequentialBlock = ForwardRef("transformers4rec.torch.block.base.SequentialBlock")
BuildableBlock = ForwardRef("transformers4rec.torch.block.base.BuildableBlock")
BlockWithHead = ForwardRef("transformers4rec.torch.block.with_head.BlockWithHead")

Head = ForwardRef("transformers4rec.torch.head.Head")
PredictionTask = ForwardRef("transformers4rec.torch.head.PredictionTask")

__all__ = [
    "TabularData",
    "TensorOrTabularData",
    "TabularModule",
    "ProcessedSequence",
    "FeatureAggregator",
    "MaskSequence",
    "MaskedSequence",
    "Block",
    "SequentialBlock",
    "BuildableBlock",
    "BlockWithHead",
    "Head",
    "PredictionTask",
]
