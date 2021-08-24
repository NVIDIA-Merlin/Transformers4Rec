from typing import Dict, Optional

import tensorflow as tf
from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig

from ...types import DatasetSchema, Tag
from ..block.base import SequentialBlock
from ..block.mlp import MLPBlock
from ..masking import masking_registry
from ..tabular import AsTabular
from .embedding import EmbeddingFeatures, TableConfig
from .tabular import TabularFeatures


class SequentialEmbeddingFeatures(EmbeddingFeatures):
    def __init__(
        self, feature_config: Dict[str, FeatureConfig], item_id=None, mask_zero=True, **kwargs
    ):
        super().__init__(feature_config, item_id, **kwargs)
        self.mask_zero = mask_zero

    def lookup_feature(self, name, val):
        table: TableConfig = self.embeddings[name].table
        table_var = self.embedding_tables[table.name]
        if isinstance(val, tf.SparseTensor):
            return tf.nn.safe_embedding_lookup_sparse(
                table_var, tf.cast(val, tf.int32), None, combiner=table.combiner
            )

        return tf.gather(table_var, tf.cast(val, tf.int32))

    def compute_output_shape(self, input_shapes):
        return super().compute_output_shape(input_shapes)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        filtered_inputs = self.filter_features(inputs)

        outputs = {}
        for key, val in filtered_inputs.items():
            outputs[key] = tf.not_equal(val, 0)

        return outputs


class TabularSequenceFeatures(TabularFeatures):
    EMBEDDING_MODULE_CLASS = SequentialEmbeddingFeatures

    def __init__(
        self,
        continuous_layer=None,
        categorical_layer=None,
        text_embedding_layer=None,
        projection_module=None,
        masking=None,
        aggregation=None,
        **kwargs
    ):
        super().__init__(
            continuous_layer, categorical_layer, text_embedding_layer, aggregation, **kwargs
        )
        self.projection_module = projection_module
        if masking:
            self.masking = masking

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        continuous_tags=Tag.CONTINUOUS,
        categorical_tags=Tag.CATEGORICAL,
        aggregation=None,
        max_sequence_length=None,
        continuous_projection=None,
        continuous_soft_embeddings_shape=None,
        projection=None,
        d_output=None,
        masking=None,
        **kwargs
    ) -> "TabularSequenceFeatures":
        """Instantiates ``TabularFeatures`` from a ```DatasetSchema`
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
            If set, concatenate all numerical features and projet them by a number of MLP layers.
            The argument accepts a list with the dimensions of the MLP layers, by default None
        continuous_soft_embeddings_shape : Optional[Union[Tuple[int, int], List[int, int]]]
            If set, uses soft one-hot encoding technique to represent continuous features.
            The argument accepts a tuple with 2 elements: [embeddings cardinality, embeddings dim],
            by default None
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
        TabularFeatures
            Returns ``TabularFeatures`` from a dataset schema
        """
        output = super().from_schema(
            schema=schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            aggregation=aggregation,
            max_sequence_length=max_sequence_length,
            continuous_projection=continuous_projection,
            # continuous_soft_embeddings_shape=,
            **kwargs
        )
        if d_output and projection:
            raise ValueError("You cannot specify both d_output and projection at the same time")
        if (projection or masking or d_output) and not aggregation:
            # TODO: print warning here for clarity
            output.aggregation = "sequential_concat"
        # hidden_size = output.output_size()

        if d_output and not projection:
            projection = MLPBlock([d_output])

        if isinstance(masking, str):
            masking = masking_registry.parse(masking)(**kwargs)
        if masking and not getattr(output, "item_id", None):
            raise ValueError("For masking a categorical_module is required including an item_id.")
        output.masking = masking

        return output

    def project_continuous_features(self, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions]

        continuous = self.continuous_layer
        continuous.set_aggregation("sequential_concat")

        continuous = SequentialBlock(
            [continuous, MLPBlock(dimensions), AsTabular("continuous_projection")]
        )

        self.to_merge["continuous_layer"] = continuous

        return self

    @property
    def masking(self):
        return self._masking

    @masking.setter
    def masking(self, value):
        self._masking = value

    @property
    def item_id(self) -> Optional[str]:
        if "categorical_layer" in self.to_merge:
            return getattr(self.to_merge["categorical_layer"], "item_id", None)

        return None

    @property
    def item_embedding_table(self):
        if "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_layer"], "item_embedding_table", None)

        return None
