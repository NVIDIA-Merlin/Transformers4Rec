from typing import Dict

import tensorflow as tf
from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig

from .embedding import EmbeddingFeatures, TableConfig


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
