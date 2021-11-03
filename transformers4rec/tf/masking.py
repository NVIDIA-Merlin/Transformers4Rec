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
from dataclasses import dataclass
from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.ops import array_ops

from merlin_standard_lib import Registry
from merlin_standard_lib.utils.doc_utils import docstring_parameter

masking_registry = Registry("tf.masking")


MASK_SEQUENCE_PARAMETERS_DOCSTRING = """
    hidden_size: int
        The hidden dimension of input tensors, needed to initialize trainable vector of masked
        positions.
    padding_idx: int, default = 0
        Index of padding item used for getting batch of sequences with the same length
    eval_on_last_item_seq_only: bool, default = True
        Predict only last item during evaluation
"""


@dataclass
class MaskingInfo:
    schema: tf.Tensor
    targets: tf.Tensor


@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class MaskSequence(tf.keras.layers.Layer):
    """Base class to prepare masked items inputs/labels for language modeling tasks.

    Transformer architectures can be trained in different ways. Depending of the training method,
    there is a specific masking schema. The masking schema sets the items to be predicted (labels)
    and mask (hide) their positions in the sequence so that they are not used by the Transformer
    layers for prediction.

    We currently provide 4 different masking schemes out of the box:
        - Causal LM (clm)
        - Masked LM (mlm)
        - Permutation LM (plm)
        - Replacement Token Detection (rtd)

    This class can be extended to add different a masking scheme.

    Parameters
    ----------
    {mask_sequence_parameters}
    """

    # TODO: Link to masking-class in the doc-string.

    def __init__(self, padding_idx: int = 0, eval_on_last_item_seq_only: bool = True, **kwargs):
        super(MaskSequence, self).__init__(**kwargs)
        self.padding_idx = padding_idx
        self.eval_on_last_item_seq_only = eval_on_last_item_seq_only
        self.mask_schema = None
        self.masked_targets = None

    def get_config(self):
        config = super(MaskSequence, self).get_config()
        config.update(
            {
                "padding_idx": self.padding_idx,
                "eval_on_last_item_seq_only": self.eval_on_last_item_seq_only,
            }
        )

        return config

    def _compute_masked_targets(self, item_ids: tf.Tensor, training=False) -> MaskingInfo:
        """
        Method to prepare masked labels based on the sequence of item ids.
        It returns The true labels of masked positions and the related boolean mask.

        Parameters
        ----------
        item_ids: tf.Tensor
            The sequence of input item ids used for deriving labels of
            next item prediction task.
        training: bool
            Flag to indicate whether we are in `Training` mode or not.
            During training, the labels can be any items within the sequence
            based on the selected masking task.
            During evaluation, we are predicting all next items or last item only
            in the sequence based on the param `eval_on_last_item_seq_only`.
        """
        raise NotImplementedError

    def compute_masked_targets(self, item_ids: tf.Tensor, training=False) -> MaskingInfo:
        """
        Method to prepare masked labels based on the sequence of item ids.
        It returns The true labels of masked positions and the related boolean mask.
        And the attributes of the class `mask_schema` and `masked_targets`
        are updated to be re-used in other modules.

        Parameters
        ----------
        item_ids: tf.Tensor
            The sequence of input item ids used for deriving labels of
            next item prediction task.
        training: bool
            Flag to indicate whether we are in `Training` mode or not.
            During training, the labels can be any items within the sequence
            based on the selected masking task.
            During evaluation, we are predicting the last item in the sequence.
        Returns
        -------
        Tuple[MaskingSchema, MaskedTargets]
        """
        assert item_ids.shape.rank == 2, "`item_ids` must have 2 dimensions."
        masking_info = self._compute_masked_targets(item_ids, training=training)
        self.mask_schema, self.masked_targets = masking_info.schema, masking_info.targets

        return masking_info

    def apply_mask_to_inputs(self, inputs: tf.Tensor, schema: tf.Tensor) -> tf.Tensor:
        """
        Control the masked positions in the inputs by replacing the true interaction
        by a learnable masked embedding.

        Parameters
        ----------
        inputs: tf.Tensor
            The 3-D tensor of interaction embeddings resulting from the ops:
            TabularFeatures + aggregation + projection(optional)
        schema: MaskingSchema
            The boolean mask indicating masked positions.
        """
        inputs = tf.where(
            tf.cast(tf.expand_dims(schema, -1), tf.bool),
            inputs,
            tf.cast(self.masked_item_embedding, dtype=inputs.dtype),
        )
        return inputs

    def predict_all(self, item_ids: tf.Tensor) -> MaskingInfo:
        """
        Prepare labels for all next item predictions instead of
        last-item predictions in a user's sequence.

        Parameters
        ----------
        item_ids: tf.Tensor
            The sequence of input item ids used for deriving labels of
            next item prediction task.
        Returns
        -------
        Tuple[MaskingSchema, MaskedTargets]
        """
        # TODO : Add option to predict N-last items
        # shift sequence of item-ids
        labels = item_ids[:, 1:]
        # As after shifting the sequence length will be subtracted by one, adding a masked item in
        # the sequence to return to the initial sequence.
        # This is important for ReformerModel(), for example
        labels = tf.concat(
            [
                labels,
                tf.zeros((tf.shape(labels)[0], 1), dtype=labels.dtype),
            ],
            axis=-1,
        )
        # apply mask on input where target is on padding index
        mask_labels = labels != self.padding_idx

        return MaskingInfo(mask_labels, labels)

    def call(self, inputs: tf.Tensor, item_ids: tf.Tensor, training=False) -> tf.Tensor:
        _ = self.compute_masked_targets(item_ids=item_ids, training=training)
        return self.apply_mask_to_inputs(inputs, self.mask_schema)

    def transformer_required_arguments(self) -> Dict[str, Any]:
        return {}

    def transformer_optional_arguments(self) -> Dict[str, Any]:
        return {}

    @property
    def transformer_arguments(self) -> Dict[str, Any]:
        """
        Prepare additional arguments to pass to the Transformer forward methods.
        """
        return {**self.transformer_required_arguments(), **self.transformer_optional_arguments()}

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        # Create a trainable embedding to replace masked interactions
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        self.masked_item_embedding = tf.Variable(
            initializer(shape=[self.hidden_size], dtype=tf.float32)
        )

        return super().build(input_shape)


@masking_registry.register_with_multiple_names("clm", "causal")
@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class CausalLanguageModeling(MaskSequence):
    """
    In Causal Language Modeling (clm) you predict the next item based on past positions of the
    sequence. Future positions are masked.

    Parameters
    ----------
    {mask_sequence_parameters}
    train_on_last_item_seq_only: predict only last item during training
    """

    def __init__(
        self,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        train_on_last_item_seq_only: bool = False,
        **kwargs
    ):
        super(CausalLanguageModeling, self).__init__(
            padding_idx=padding_idx, eval_on_last_item_seq_only=eval_on_last_item_seq_only, **kwargs
        )
        self.train_on_last_item_seq_only = train_on_last_item_seq_only
        self.label_seq_trg_eval = tf.Variable(
            tf.zeros(shape=[1, 1], dtype=tf.int32),
            dtype=tf.int32,
            trainable=False,
            shape=tf.TensorShape([None, None]),
        )

    def get_config(self):
        config = super(CausalLanguageModeling, self).get_config()
        config.update(
            {
                "train_on_last_item_seq_only": self.train_on_last_item_seq_only,
            }
        )
        return config

    def _compute_masked_targets(self, item_ids: tf.Tensor, training=False) -> MaskingInfo:
        item_ids = tf.cast(item_ids, dtype=tf.int32)
        masking_info: MaskingInfo = self.predict_all(item_ids)
        mask_labels, labels = masking_info.schema, masking_info.targets

        if (self.eval_on_last_item_seq_only and not training) or (
            self.train_on_last_item_seq_only and training
        ):
            last_item_sessions = tf.reduce_sum(tf.cast(mask_labels, labels.dtype), axis=1) - 1

            rows_ids = tf.range(tf.shape(labels)[0], dtype=labels.dtype)
            self.label_seq_trg_eval.assign(tf.zeros(tf.shape(labels), dtype=tf.int32))

            indices = tf.concat(
                [tf.expand_dims(rows_ids, 1), tf.expand_dims(last_item_sessions, 1)], axis=1
            )
            self.label_seq_trg_eval.scatter_nd_update(
                indices=indices, updates=tf.gather_nd(labels, indices)
            )
            # Updating labels and mask
            mask_labels = self.label_seq_trg_eval != self.padding_idx
        else:
            self.label_seq_trg_eval.assign(labels)

        return MaskingInfo(mask_labels, self.label_seq_trg_eval)

    def apply_mask_to_inputs(self, inputs: tf.Tensor, mask_schema: tf.Tensor) -> tf.Tensor:
        # shift sequence of interaction embeddings
        pos_emb_inp = inputs[:, :-1]
        # Adding a masked item in the sequence to return to the initial sequence.
        pos_emb_inp = tf.concat(
            [
                pos_emb_inp,
                tf.zeros(
                    (tf.shape(pos_emb_inp)[0], 1, pos_emb_inp.shape[2]), dtype=pos_emb_inp.dtype
                ),
            ],
            axis=1,
        )
        # Replacing the inputs corresponding to masked label with a trainable embedding
        pos_emb_inp = tf.where(
            tf.cast(tf.expand_dims(mask_schema, -1), tf.bool),
            pos_emb_inp,
            tf.cast(self.masked_item_embedding, dtype=inputs.dtype),
        )

        return pos_emb_inp


@masking_registry.register_with_multiple_names("mlm", "masked")
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class MaskedLanguageModeling(MaskSequence):
    """
    In Masked Language Modeling (mlm) you randomly select some positions of the sequence to be
    predicted, which are masked.
    During training, the Transformer layer is allowed to use positions on the right (future info).
    During inference, all past items are visible for the Transformer layer, which tries to predict
    the next item.

    Parameters
    ----------
    {mask_sequence_parameters}
    mlm_probability: Optional[float], default = 0.15
        Probability of an item to be selected (masked) as a label of the given sequence.
        p.s. We enforce that at least one item is masked for each sequence, so that the network can
        learn something with it.
    """

    def __init__(
        self,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        mlm_probability: float = 0.15,
        **kwargs
    ):
        super(MaskedLanguageModeling, self).__init__(
            padding_idx=padding_idx, eval_on_last_item_seq_only=eval_on_last_item_seq_only, **kwargs
        )
        self.mlm_probability = mlm_probability
        self.labels = tf.Variable(
            tf.zeros(shape=[1, 1], dtype=tf.int32),
            dtype=tf.int32,
            trainable=False,
            shape=tf.TensorShape([None, None]),
        )

    def get_config(self):
        config = super(MaskedLanguageModeling, self).get_config()
        config.update(
            {
                "mlm_probability": self.mlm_probability,
            }
        )
        return config

    def _compute_masked_targets(self, item_ids: tf.Tensor, training: bool = False) -> MaskingInfo:
        """
        Prepare sequence with mask schema for masked language modeling prediction
        the function is based on HuggingFace's transformers/data/data_collator.py

        Parameters
        ----------
        item_ids: tf.Tensor
            Sequence of input itemid (target) column

        Returns
        -------
        labels: tf.Tensor
            Sequence of masked item ids.
        mask_labels: tf.Tensor
            Masking schema for masked targets positions.
        """
        # cast item_ids to int32
        item_ids = tf.cast(item_ids, dtype=tf.int32)
        self.labels.assign(tf.fill(tf.shape(item_ids), self.padding_idx))

        non_padded_mask = tf.cast(item_ids != self.padding_idx, self.labels.dtype)
        rows_ids = tf.range(tf.shape(item_ids)[0], dtype=tf.int64)
        # During training, masks labels to be predicted according to a probability, ensuring that
        #   each session has at least one label to predict
        if training:
            # Selects a percentage of items to be masked (selected as labels)
            probability_matrix = tf.cast(
                backend.random_bernoulli(array_ops.shape(item_ids), p=self.mlm_probability),
                self.labels.dtype,
            )

            mask_labels = probability_matrix * non_padded_mask
            self.labels.assign(
                tf.where(
                    tf.cast(mask_labels, tf.bool),
                    item_ids,
                    tf.cast(tf.fill(tf.shape(item_ids), self.padding_idx), dtype=item_ids.dtype),
                )
            )

            # Set at least one item in the sequence to mask, so that the network
            # can learn something with this session
            one_random_index_by_session = tf.random.categorical(
                tf.math.log(tf.cast(non_padded_mask, tf.float32)), num_samples=1
            )
            indices = tf.concat([tf.expand_dims(rows_ids, 1), one_random_index_by_session], axis=1)
            self.labels.scatter_nd_update(indices=indices, updates=tf.gather_nd(item_ids, indices))
            mask_labels = tf.cast(self.labels != self.padding_idx, self.labels.dtype)

            # If a sequence has only masked labels, unmask one of the labels
            sequences_with_only_labels = tf.reduce_sum(mask_labels, axis=1) == tf.reduce_sum(
                non_padded_mask, axis=1
            )
            sampled_labels_to_unmask = tf.random.categorical(
                tf.math.log(tf.cast(mask_labels, tf.float32)), num_samples=1
            )

            labels_to_unmask = tf.boolean_mask(sampled_labels_to_unmask, sequences_with_only_labels)
            rows_to_unmask = tf.boolean_mask(rows_ids, sequences_with_only_labels)
            indices = tf.concat([tf.expand_dims(rows_to_unmask, 1), labels_to_unmask], axis=1)
            num_updates = tf.shape(indices)[0]
            self.labels.scatter_nd_update(
                indices, tf.cast(tf.fill((num_updates,), self.padding_idx), self.labels.dtype)
            )
            mask_labels = self.labels != self.padding_idx

        else:
            if self.eval_on_last_item_seq_only:
                last_item_sessions = tf.reduce_sum(non_padded_mask, axis=1) - 1

                indices = tf.concat(
                    [
                        tf.expand_dims(rows_ids, 1),
                        tf.cast(tf.expand_dims(last_item_sessions, 1), tf.int64),
                    ],
                    axis=1,
                )
                self.labels.scatter_nd_update(
                    indices=indices, updates=tf.gather_nd(item_ids, indices)
                )
                mask_labels = self.labels != self.padding_idx
            else:
                masking_info = self.predict_all(item_ids)
                mask_labels, labels = masking_info.schema, masking_info.targets
                self.labels.assign(labels)

        return MaskingInfo(mask_labels, self.labels)


# @masking_registry.register_with_multiple_names("plm", "permutation")
# class PermutationLanguageModeling(MaskSequence):
#     pass
#
#
# @masking_registry.register_with_multiple_names("rtd", "replacement")
# class ReplacementLanguageModeling(MaskSequence):
#     pass
