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
from typing import Any, Dict, Tuple

import tensorflow as tf

from ..utils.misc_utils import docstring_parameter
from ..utils.registry import Registry

masking_registry = Registry("tf.masking")

MaskingSchema = tf.Tensor
MaskedTargets = tf.Tensor


MASK_SEQUENCE_PARAMETERS_DOCSTRING = """
    hidden_size: int
        The hidden dimension of input tensors, needed to initialize trainable vector of masked
        positions.
    padding_idx: int, default = 0
        Index of padding item used for getting batch of sequences with the same length
    eval_on_last_item_seq_only: bool, default = True
        Predict only last item during evaluation
"""


@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
class MaskSequence(tf.keras.layers.Layer):
    """Base class to prepare masked items inputs/labels for language modeling tasks.
    Parameters

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

    ----------
    {mask_sequence_parameters}
    """

    # TODO: Link to masking-class in the doc-string.

    def __init__(
        self,
        hidden_size: int,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
    ):
        super(MaskSequence, self).__init__()
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.eval_on_last_item_seq_only = eval_on_last_item_seq_only
        self.mask_schema = None
        self.masked_targets = None

        # Create a trainable embedding to replace masked interactions
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        self.masked_item_embedding = tf.Variable(
            initializer(shape=[self.hidden_size], dtype=tf.float32)
        )

    def _compute_masked_targets(
        self, item_ids: tf.Tensor, training=False
    ) -> Tuple[MaskingSchema, MaskedTargets]:
        """
        Method to prepare masked labels based on the sequence of item ids.
        It returns The true labels of masked positions and the related boolean mask.
        Parameters
        ----------
        item_ids: torch.Tensor
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

    def compute_masked_targets(
        self, item_ids: tf.Tensor, training=False
    ) -> Tuple[MaskingSchema, MaskedTargets]:
        """
        Method to prepare masked labels based on the sequence of item ids.
        It returns The true labels of masked positions and the related boolean mask.
        And the attributes of the class `mask_schema` and `masked_targets`
        are updated to be re-used in other modules.
        Parameters
        ----------
        item_ids: torch.Tensor
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
        assert item_ids.ndim == 2, "`item_ids` must have 2 dimensions."
        self.mask_schema, self.masked_targets = self._compute_masked_targets(
            item_ids, training=training
        )

        return self.mask_schema, self.masked_targets

    def apply_mask_to_inputs(self, inputs: tf.Tensor, schema: MaskingSchema) -> tf.Tensor:
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
            schema.unsqueeze(-1).bool(),
            tf.cast(self.masked_item_embedding, dtype=inputs.dtype),
            inputs,
        )
        return inputs

    def predict_all(self, item_ids: tf.tensor) -> Tuple[MaskingSchema, MaskedTargets]:
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
        labels = tf.cat(
            [
                labels,
                tf.zeros((labels.shape[0], 1), dtype=labels.dtype),
            ],
            axis=-1,
        )
        # apply mask on input where target is on padding index
        mask_labels = labels != self.padding_idx
        return mask_labels, labels

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
        return super().build(input_shape)


@masking_registry.register_with_multiple_names("clm", "causal")
@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
class CausalLanguageModeling(MaskSequence):
    """
    In Causal Language Modeling (clm) you predict the next item based on past positions of the
    sequence. Future positions are masked.
    Parameters:
    ----------
    {mask_sequence_parameters}
    train_on_last_item_seq_only: predict only last item during training
    """

    def __init__(
        self,
        hidden_size: int,
        device: str = "cpu",
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        train_on_last_item_seq_only: bool = False,
    ):
        super(CausalLanguageModeling, self).__init__(
            hidden_size=hidden_size,
            device=device,
            padding_idx=padding_idx,
            eval_on_last_item_seq_only=eval_on_last_item_seq_only,
        )
        self.train_on_last_item_seq_only = train_on_last_item_seq_only

    def _compute_masked_targets(
        self, item_ids: tf.Tensor, training=False
    ) -> Tuple[MaskingSchema, MaskedTargets]:
        mask_labels, labels = self.predict_all(item_ids)

        if (self.eval_on_last_item_seq_only and not training) or (
            self.train_on_last_item_seq_only and training
        ):
            rows_ids = tf.range(labels.size(0), dtype=tf.int64)
            last_item_sessions = mask_labels.sum(axis=1) - 1
            label_seq_trg_eval = tf.zeros(labels.shape, dtype=tf.int64)
            label_seq_trg_eval[rows_ids, last_item_sessions] = labels[rows_ids, last_item_sessions]
            # Updating labels and mask
            labels = label_seq_trg_eval
            mask_labels = label_seq_trg_eval != self.padding_idx
        return mask_labels, labels

    def apply_mask_to_inputs(self, inputs: tf.Tensor, mask_schema: MaskingSchema) -> tf.Tensor:
        # shift sequence of interaction embeddings
        pos_emb_inp = inputs[:, :-1]
        # Adding a masked item in the sequence to return to the initial sequence.
        pos_emb_inp = tf.cat(
            [
                pos_emb_inp,
                tf.zeros(
                    (pos_emb_inp.shape[0], 1, pos_emb_inp.shape[2]),
                    dtype=pos_emb_inp.dtype,
                ).to(self.device),
            ],
            axis=1,
        )
        # Replacing the inputs corresponding to masked label with a trainable embedding
        pos_emb_inp = tf.where(
            mask_schema.unsqueeze(-1).bool(),
            pos_emb_inp,
            self.masked_item_embedding.to(pos_emb_inp.dtype),
        )
        return pos_emb_inp


@masking_registry.register_with_multiple_names("mlm", "masked")
class MaskedLanguageModeling(MaskSequence):
    pass
