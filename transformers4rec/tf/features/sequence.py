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

from typing import Dict, Optional, Union

import tensorflow as tf

from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.utils.doc_utils import docstring_parameter

from ..block.base import Block, SequentialBlock
from ..block.mlp import MLPBlock
from ..masking import MaskSequence, masking_registry
from ..tabular.base import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    AsTabular,
    TabularAggregationType,
    TabularBlock,
    TabularTransformationType,
)
from ..utils import tf_utils
from . import embedding
from .tabular import TABULAR_FEATURES_PARAMS_DOCSTRING, TabularFeatures


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    embedding_features_parameters=embedding.EMBEDDING_FEATURES_PARAMS_DOCSTRING,
)
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class SequenceEmbeddingFeatures(embedding.EmbeddingFeatures):
    """Input block for embedding-lookups for categorical features. This module produces 3-D tensors,
    this is useful for sequential models like transformers.

    Parameters
    ----------
    {embedding_features_parameters}
    padding_idx: int
        The symbol to use for padding.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        feature_config: Dict[str, embedding.FeatureConfig],
        item_id: Optional[str] = None,
        mask_zero: bool = True,
        padding_idx: int = 0,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        embedding_tables={},
        **kwargs
    ):
        super().__init__(
            feature_config,
            item_id=item_id,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            embedding_tables=embedding_tables,
            **kwargs,
        )
        self.padding_idx = padding_idx
        self.mask_zero = mask_zero

    def lookup_feature(self, name, val, **kwargs):
        return super(SequenceEmbeddingFeatures, self).lookup_feature(
            name, val, output_sequence=True
        )

    def compute_call_output_shape(self, input_shapes):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)
        sequence_length = input_shapes[list(self.feature_config.keys())[0]][1]

        output_shapes = {}
        for name, val in input_shapes.items():
            output_shapes[name] = tf.TensorShape(
                [batch_size, sequence_length, self.feature_config[name].table.dim]
            )

        return output_shapes

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        outputs = {}
        for key, val in inputs.items():
            outputs[key] = tf.not_equal(val, self.padding_idx)

        return outputs

    def get_config(self):
        config = super().get_config()
        config["mask_zero"] = self.mask_zero
        config["padding_idx"] = self.padding_idx

        return config


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    tabular_features_parameters=TABULAR_FEATURES_PARAMS_DOCSTRING,
)
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class TabularSequenceFeatures(TabularFeatures):
    """Input module that combines different types of features to a sequence: continuous,
    categorical & text.

    Parameters
    ----------
    {tabular_features_parameters}
    projection_module: BlockOrModule, optional
        Module that's used to project the output of this module, typically done by an MLPBlock.
    masking: MaskSequence, optional
         Masking to apply to the inputs.
    {tabular_module_parameters}

    """

    EMBEDDING_MODULE_CLASS = SequenceEmbeddingFeatures

    def __init__(
        self,
        continuous_layer: Optional[TabularBlock] = None,
        categorical_layer: Optional[TabularBlock] = None,
        text_embedding_layer: Optional[TabularBlock] = None,
        projection_block: Optional[Block] = None,
        masking: Optional[MaskSequence] = None,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        name=None,
        **kwargs
    ):
        super().__init__(
            continuous_layer=continuous_layer,
            categorical_layer=categorical_layer,
            text_embedding_layer=text_embedding_layer,
            pre=pre,
            post=post,
            aggregation=aggregation,
            name=name,
            **kwargs,
        )
        self.projection_block = projection_block
        self.set_masking(masking)

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        continuous_tags=(Tag.CONTINUOUS,),
        categorical_tags=(Tag.CATEGORICAL,),
        aggregation=None,
        max_sequence_length=None,
        continuous_projection=None,
        projection=None,
        d_output=None,
        masking=None,
        **kwargs
    ) -> "TabularSequenceFeatures":
        """Instantiates ``TabularFeatures`` from a ``DatasetSchema``

        Parameters
        ----------
        schema : DatasetSchema
            Dataset schema
        continuous_tags : Optional[Union[DefaultTags, list, str]], optional
            Tags to filter the continuous features, by default Tag.CONTINUOUS
        categorical_tags : Optional[Union[DefaultTags, list, str]], optional
            Tags to filter the categorical features, by default Tag.CATEGORICAL
        aggregation : Optional[str], optional
            Feature aggregation option, by default None
        automatic_build : bool, optional
            Automatically infers input size from features, by default True
        max_sequence_length : Optional[int], optional
            Maximum sequence length for list features by default None
        continuous_projection : Optional[Union[List[int], int]], optional
            If set, concatenate all numerical features and project them by a number of MLP layers
            The argument accepts a list with the dimensions of the MLP layers, by default None
        projection: Optional[torch.nn.Module, BuildableBlock], optional
            If set, project the aggregated embeddings vectors into hidden dimension vector space,
            by default None
        d_output: Optional[int], optional
            If set, init a MLPBlock as projection module to project embeddings vectors,
            by default None
        masking: Optional[Union[str, MaskSequence]], optional
            If set, Apply masking to the input embeddings and compute masked labels, It requires
            a categorical_module including an item_id column, by default None

        Returns
        -------
        TabularFeatures:
            Returns ``TabularFeatures`` from a dataset schema"""
        output = super().from_schema(
            schema=schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            aggregation=aggregation,
            max_sequence_length=max_sequence_length,
            continuous_projection=continuous_projection,
            **kwargs,
        )
        if d_output and projection:
            raise ValueError("You cannot specify both d_output and projection at the same time")
        if (projection or masking or d_output) and not aggregation:
            # TODO: print warning here for clarity
            output.set_aggregation("concat")
        # hidden_size = output.output_size()

        if d_output and not projection:
            projection = MLPBlock([d_output])

        if projection:
            output.projection_block = projection

        if isinstance(masking, str):
            masking = masking_registry.parse(masking)(**kwargs)
        if masking and not getattr(output, "item_id", None):
            raise ValueError("For masking a categorical_module is required including an item_id.")
        output.set_masking(masking)

        return output

    def project_continuous_features(self, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions]

        continuous = self.continuous_layer
        continuous.set_aggregation("concat")

        continuous = SequentialBlock(
            [continuous, MLPBlock(dimensions), AsTabular("continuous_projection")]
        )

        self.to_merge["continuous_layer"] = continuous

        return self

    def call(self, inputs, training=True):
        outputs = super(TabularSequenceFeatures, self).call(inputs)

        if self.masking or self.projection_block:
            outputs = self.aggregation(outputs)

        if self.projection_block:
            outputs = self.projection_block(outputs)

        if self.masking:
            outputs = self.masking(
                outputs, item_ids=self.to_merge["categorical_layer"].item_seq, training=training
            )

        return outputs

    def compute_call_output_shape(self, input_shape):
        output_shapes = {}

        for layer in self.merge_values:
            output_shapes.update(layer.compute_output_shape(input_shape))

        return output_shapes

    def compute_output_shape(self, input_shapes):
        output_shapes = super().compute_output_shape(input_shapes)

        if self.projection_block:
            output_shapes = self.projection_block.compute_output_shape(output_shapes)

        return output_shapes

    @property
    def masking(self):
        return self._masking

    def set_masking(self, value):
        self._masking = value

    @property
    def item_id(self) -> Optional[str]:
        if "categorical_layer" in self.to_merge_dict:
            return getattr(self.to_merge_dict["categorical_layer"], "item_id", None)

        return None

    @property
    def item_embedding_table(self):
        if "categorical_layer" in self.to_merge_dict:
            return getattr(self.to_merge_dict["categorical_layer"], "item_embedding_table", None)

        return None

    def get_config(self):
        config = super().get_config()
        config = tf_utils.maybe_serialize_keras_objects(
            self, config, ["projection_block", "masking"]
        )

        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["projection_block", "masking"])

        return super().from_config(config)


TabularFeaturesType = Union[TabularSequenceFeatures, TabularFeatures]
