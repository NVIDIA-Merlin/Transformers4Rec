import sys
import typing
from typing import Dict

import tensorflow as tf

if sys.version_info < (3, 7):
    ForwardRef = typing._ForwardRef  # pylint: disable=protected-access
else:
    ForwardRef = typing.ForwardRef

# TODO: Make this more generic and work with multi-hot features
TabularData = Dict[str, tf.Tensor]

TabularModule = ForwardRef("transformers4rec.tf.tabular.TabularModule")

FeatureAggregator = ForwardRef("transformers4rec.tf.aggregator.FeatureAggregator")
MaskSequence = ForwardRef("transformers4rec.tf.masking.MaskSequence")

Block = ForwardRef("transformers4rec.tf.block.base.Block")
SequentialBlock = ForwardRef("transformers4rec.tf.block.base.SequentialBlock")
BuildableBlock = ForwardRef("transformers4rec.tf.block.base.BuildableBlock")
BlockWithHead = ForwardRef("transformers4rec.tf.block.with_head.BlockWithHead")

Head = ForwardRef("transformers4rec.tf.head.Head")
PredictionTask = ForwardRef("transformers4rec.tf.head.PredictionTask")

__all__ = [
    "TabularData",
    "TabularModule",
    "FeatureAggregator",
    "MaskSequence",
    "Block",
    "SequentialBlock",
    "BuildableBlock",
    "BlockWithHead",
    "Head",
    "PredictionTask",
]
