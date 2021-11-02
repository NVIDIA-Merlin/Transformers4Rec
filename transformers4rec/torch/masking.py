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
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from merlin_standard_lib import Registry
from merlin_standard_lib.utils.doc_utils import docstring_parameter

from .utils.torch_utils import OutputSizeMixin

masking_registry = Registry("torch.masking")


@dataclass
class MaskingInfo:
    schema: torch.Tensor
    targets: torch.Tensor


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
class MaskSequence(OutputSizeMixin, torch.nn.Module):
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
    hidden_size:
        The hidden dimension of input tensors, needed to initialize trainable vector of
        masked positions.
    pad_token: int, default = 0
        Index of the padding token used for getting batch of sequences with the same length
    """

    # TODO: Link to masking-class in the doc-string.

    def __init__(
        self,
        hidden_size: int,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        **kwargs
    ):
        super(MaskSequence, self).__init__()
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.eval_on_last_item_seq_only = eval_on_last_item_seq_only
        self.mask_schema: Optional[torch.Tensor] = None
        self.masked_targets: Optional[torch.Tensor] = None

        # Create a trainable embedding to replace masked interactions
        self.masked_item_embedding = nn.Parameter(torch.Tensor(self.hidden_size))
        torch.nn.init.normal_(
            self.masked_item_embedding,
            mean=0,
            std=0.001,
        )

    def _compute_masked_targets(self, item_ids: torch.Tensor, training=False) -> MaskingInfo:
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

    def compute_masked_targets(self, item_ids: torch.Tensor, training=False) -> MaskingInfo:
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
        masking_info = self._compute_masked_targets(item_ids, training=training)
        self.mask_schema, self.masked_targets = masking_info.schema, masking_info.targets

        return masking_info

    def apply_mask_to_inputs(self, inputs: torch.Tensor, schema: torch.Tensor) -> torch.Tensor:
        """
        Control the masked positions in the inputs by replacing the true interaction
        by a learnable masked embedding.

        Parameters
        ----------
        inputs: torch.Tensor
            The 3-D tensor of interaction embeddings resulting from the ops:
            TabularFeatures + aggregation + projection(optional)
        schema: MaskingSchema
            The boolean mask indicating masked positions.
        """
        inputs = torch.where(
            schema.unsqueeze(-1).bool(),
            self.masked_item_embedding.to(inputs.dtype),
            inputs,
        )
        return inputs

    def predict_all(self, item_ids: torch.Tensor) -> MaskingInfo:
        """
        Prepare labels for all next item predictions instead of
        last-item predictions in a user's sequence.

        Parameters
        ----------
        item_ids: torch.Tensor
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
        labels = torch.cat(  # type: ignore
            [
                labels,
                torch.zeros((labels.shape[0], 1), dtype=labels.dtype).to(item_ids.device),
            ],
            axis=-1,
        )
        # apply mask on input where target is on padding index
        mask_labels = labels != self.padding_idx

        return MaskingInfo(mask_labels, labels)

    def forward(self, inputs: torch.Tensor, item_ids: torch.Tensor, training=False) -> torch.Tensor:
        _ = self.compute_masked_targets(item_ids=item_ids, training=training)
        if self.mask_schema is None:
            raise ValueError("`mask_schema must be set.`")
        schema: torch.Tensor = self.mask_schema
        return self.apply_mask_to_inputs(inputs, schema)

    def forward_output_size(self, input_size):
        return input_size

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


@masking_registry.register_with_multiple_names("clm", "causal")
@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
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
        hidden_size: int,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        train_on_last_item_seq_only: bool = False,
        **kwargs
    ):
        super(CausalLanguageModeling, self).__init__(
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            eval_on_last_item_seq_only=eval_on_last_item_seq_only,
            kwargs=kwargs,
        )
        self.train_on_last_item_seq_only = train_on_last_item_seq_only

    def _compute_masked_targets(self, item_ids: torch.Tensor, training=False) -> MaskingInfo:
        masking_info = self.predict_all(item_ids)
        mask_labels, labels = masking_info.schema, masking_info.targets

        if (self.eval_on_last_item_seq_only and not training) or (
            self.train_on_last_item_seq_only and training
        ):
            rows_ids = torch.arange(
                labels.size(0), dtype=torch.long, device=item_ids.device  # type: ignore
            )
            last_item_sessions = mask_labels.sum(dim=1) - 1
            label_seq_trg_eval = torch.zeros(labels.shape, dtype=torch.long, device=item_ids.device)
            label_seq_trg_eval[rows_ids, last_item_sessions] = labels[rows_ids, last_item_sessions]
            # Updating labels and mask
            labels = label_seq_trg_eval
            mask_labels = label_seq_trg_eval != self.padding_idx

        return MaskingInfo(mask_labels, labels)

    def apply_mask_to_inputs(self, inputs: torch.Tensor, mask_schema: torch.Tensor) -> torch.Tensor:
        # shift sequence of interaction embeddings
        pos_emb_inp = inputs[:, :-1]
        # Adding a masked item in the sequence to return to the initial sequence.
        pos_emb_inp = torch.cat(  # type: ignore
            [
                pos_emb_inp,
                torch.zeros(
                    (pos_emb_inp.shape[0], 1, pos_emb_inp.shape[2]),
                    dtype=pos_emb_inp.dtype,
                ).to(inputs.device),
            ],
            axis=1,
        )
        # Replacing the inputs corresponding to masked label with a trainable embedding
        pos_emb_inp = torch.where(
            mask_schema.unsqueeze(-1).bool(),
            pos_emb_inp,
            self.masked_item_embedding.to(pos_emb_inp.dtype),
        )
        return pos_emb_inp


@masking_registry.register_with_multiple_names("mlm", "masked")
@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
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
        hidden_size: int,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        mlm_probability: float = 0.15,
        **kwargs
    ):
        super(MaskedLanguageModeling, self).__init__(
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            eval_on_last_item_seq_only=eval_on_last_item_seq_only,
            kwargs=kwargs,
        )
        self.mlm_probability = mlm_probability

    def _compute_masked_targets(self, item_ids: torch.Tensor, training=False) -> MaskingInfo:
        """
        Prepare sequence with mask schema for masked language modeling prediction
        the function is based on HuggingFace's transformers/data/data_collator.py

        Parameters
        ----------
        item_ids: torch.Tensor
            Sequence of input itemid (target) column

        Returns
        -------
        labels: torch.Tensor
            Sequence of masked item ids.
        mask_labels: torch.Tensor
            Masking schema for masked targets positions.
        """

        labels = torch.full(
            item_ids.shape, self.padding_idx, dtype=item_ids.dtype, device=item_ids.device
        )
        non_padded_mask = item_ids != self.padding_idx

        rows_ids = torch.arange(item_ids.size(0), dtype=torch.long, device=item_ids.device)
        # During training, masks labels to be predicted according to a probability, ensuring that
        #   each session has at least one label to predict
        if training:
            # Selects a percentage of items to be masked (selected as labels)
            probability_matrix = torch.full(
                item_ids.shape, self.mlm_probability, device=item_ids.device
            )
            mask_labels = torch.bernoulli(probability_matrix).bool() & non_padded_mask
            labels = torch.where(
                mask_labels,
                item_ids,
                torch.full_like(item_ids, self.padding_idx),
            )

            # Set at least one item in the sequence to mask, so that the network
            # can learn something with this session
            one_random_index_by_session = torch.multinomial(
                non_padded_mask.float(), num_samples=1
            ).squeeze()
            labels[rows_ids, one_random_index_by_session] = item_ids[
                rows_ids, one_random_index_by_session
            ]
            mask_labels = labels != self.padding_idx

            # If a sequence has only masked labels, unmasks one of the labels
            sequences_with_only_labels = mask_labels.sum(dim=1) == non_padded_mask.sum(dim=1)
            sampled_labels_to_unmask = torch.multinomial(
                mask_labels.float(), num_samples=1
            ).squeeze()

            labels_to_unmask = torch.masked_select(
                sampled_labels_to_unmask, sequences_with_only_labels
            )
            rows_to_unmask = torch.masked_select(rows_ids, sequences_with_only_labels)

            labels[rows_to_unmask, labels_to_unmask] = self.padding_idx
            mask_labels = labels != self.padding_idx

        else:
            if self.eval_on_last_item_seq_only:
                last_item_sessions = non_padded_mask.sum(dim=1) - 1
                labels[rows_ids, last_item_sessions] = item_ids[rows_ids, last_item_sessions]
                mask_labels = labels != self.padding_idx
            else:
                masking_info = self.predict_all(item_ids)
                mask_labels, labels = masking_info.schema, masking_info.targets

        return MaskingInfo(mask_labels, labels)


@masking_registry.register_with_multiple_names("plm", "permutation")
@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
class PermutationLanguageModeling(MaskSequence):
    """
    In Permutation Language Modeling (plm) you use a permutation factorization at the level of the
    self-attention layer to define the accessible bidirectional context.

    Parameters
    ----------
    {mask_sequence_parameters}
    max_span_length: int
        maximum length of a span of masked items
    plm_probability: float
        The ratio of surrounding items to unmask to define the context of the span-based
        prediction segment of items
    permute_all: bool
        Compute partial span-based prediction (=False) or not.
    """

    def __init__(
        self,
        hidden_size: int,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        plm_probability: float = 1 / 6,
        max_span_length: int = 5,
        permute_all: bool = False,
        **kwargs
    ):
        super(PermutationLanguageModeling, self).__init__(
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            eval_on_last_item_seq_only=eval_on_last_item_seq_only,
            kwargs=kwargs,
        )

        self.plm_probability = plm_probability
        self.max_span_length = max_span_length
        self.permute_all = permute_all

        # additional masked scheme needed for XLNet-PLM task :
        self.target_mapping: Optional[torch.Tensor] = None
        self.perm_mask: Optional[torch.Tensor] = None

    def _compute_masked_targets(
        self,
        item_ids: torch.Tensor,
        training=False,
    ):
        pass

    def _compute_masked_targets_extended(
        self,
        item_ids: torch.Tensor,
        training=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the attention masks needed for permutation language modeling
        The function is based on HuggingFace's transformers/data/data_collator.py

        Parameters
        ----------
        item_ids: torch.Tensor
            Sequence of input itemid (target) column.

        Returns
        -------
        labels: torch.Tensor
            Sequence of masked item ids.
        mask_labels: torch.Tensor
            Masking schema for masked targets positions.
        perm_mask: torch.Tensor of shape (bs, seq_len, seq_len)
            The random factorization order attention mask for each target
        target_mapping: torch.Tensor of shape (bs, seq_len, seq_len) :
            Binary mask to specify the items to predict.
        """

        labels = torch.full(
            item_ids.shape, self.padding_idx, dtype=item_ids.dtype, device=item_ids.device
        )
        non_padded_mask = item_ids != self.padding_idx

        rows_ids = torch.arange(item_ids.size(0), dtype=torch.long, device=item_ids.device)
        mask_labels = torch.full(labels.shape, 0, dtype=torch.bool, device=item_ids.device)
        # During training:
        # Masks a span of consecutive items to be predicted according to plm_probability,
        # While ensuring that each session has at least one  item to predict
        if training:
            target_mapping = torch.zeros(
                (labels.size(0), labels.size(1), labels.size(1)),
                dtype=torch.float32,
                device=item_ids.device,
            )
            perm_mask = torch.zeros(
                (labels.size(0), labels.size(1), labels.size(1)),
                dtype=torch.float32,
                device=item_ids.device,
            )
            if self.permute_all:
                # Permute all non padded items
                mask_labels = non_padded_mask
            else:
                # For each session select a span of consecutive item ids to be masked
                for i in range(labels.size(0)):
                    # Start from the beginning of the sequence by setting `cur_len = 0`
                    # (number of tokens processed so far).
                    cur_len = 0
                    max_len = non_padded_mask.sum(1)[i]  # mask only non-padded items
                    while cur_len < max_len:
                        # Sample a `span_length` from the interval `[1, max_span_length]`
                        # (length of span of tokens to be masked)
                        span_length = torch.randint(1, self.max_span_length + 1, (1,)).item()
                        # Reserve a context
                        # to surround span to be masked
                        context_length = int(span_length / self.plm_probability)
                        # Sample a starting point `start_index`
                        # from the interval `[cur_len, cur_len + context_length - span_length]`
                        start_index = (
                            cur_len
                            + torch.randint(  # type: ignore
                                context_length - span_length + 1, (1,)
                            ).item()
                        )
                        if start_index < max_len:
                            # Mask the span of non-padded items
                            #   `start_index:start_index + span_length`
                            mask_labels[i, start_index : start_index + span_length] = 1
                        # Set `cur_len = cur_len + context_length`
                        cur_len += context_length
                    # if no item was masked:
                    if mask_labels[i].sum() == 0:
                        # Set at least one item in the sequence to mask, so that the network can
                        # learn something with this session
                        one_random_index_by_session = torch.multinomial(
                            non_padded_mask[i].float(), num_samples=1
                        ).squeeze()
                        mask_labels[i, one_random_index_by_session] = item_ids[
                            i, one_random_index_by_session
                        ]
                    # Since we're replacing non-masked tokens with padding_idxs in the labels tensor
                    # instead of skipping them altogether,
                    # the i-th predict corresponds to the i-th token.
                    # N.B: the loss function will be computed only on non paded items
                    target_mapping[i] = torch.eye(labels.size(1))

            labels = torch.where(mask_labels, item_ids, torch.full_like(item_ids, self.padding_idx))

            # If a sequence has only masked labels, unmasks one of the labels
            sequences_with_only_labels = mask_labels.sum(dim=1) == non_padded_mask.sum(dim=1)
            sampled_labels_to_unmask = torch.multinomial(
                mask_labels.float(), num_samples=1
            ).squeeze()

            labels_to_unmask = torch.masked_select(
                sampled_labels_to_unmask, sequences_with_only_labels
            )
            rows_to_unmask = torch.masked_select(rows_ids, sequences_with_only_labels)

            labels[rows_to_unmask, labels_to_unmask] = self.padding_idx
            mask_labels = labels != self.padding_idx

            for i in range(labels.size(0)):
                # Generate permutation indices i.e.
                #  sample a random factorisation order for the sequence.
                #  This will determine which tokens a given token can attend to
                # (encoded in `perm_mask`).
                # Create a linear factorisation order
                perm_index = torch.arange(labels.size(1), dtype=torch.long, device=item_ids.device)
                # randomly permute indices of each session
                perm_index = perm_index[torch.randperm(labels.size(1))]
                # Set the permutation indices of non-masked (non-functional) tokens to the
                # smallest index (-1) so that:
                # (1) They can be seen by all other positions
                # (2) They cannot see masked positions, so there won't be information leak
                perm_index.masked_fill_(~mask_labels[i], -1)
                # The logic for whether the i-th token can attend on the j-th token
                # based on the factorisation order:
                # 0 (can attend):
                # If perm_index[i] > perm_index[j] or j is neither masked nor a padded item
                # 1 (cannot attend):
                # If perm_index[i] <= perm_index[j] and j is either masked or a padded item
                perm_mask[i] = (
                    perm_index.reshape((labels.size(1), 1))
                    <= perm_index.reshape((1, labels.size(1)))
                ) & mask_labels[i]
        # During evaluation always mask the last item of the session
        else:
            if self.eval_on_last_item_seq_only:
                last_item_sessions = non_padded_mask.sum(dim=1) - 1
                labels[rows_ids, last_item_sessions] = item_ids[rows_ids, last_item_sessions]
                mask_labels = labels != self.padding_idx
                perm_mask = torch.zeros(
                    (labels.size(0), labels.size(1), labels.size(1)),
                    dtype=torch.float32,
                    device=item_ids.device,
                )
                # Previous tokens don't see last non-padded token
                perm_mask[rows_ids, :, last_item_sessions] = 1
                # add causal mask to avoid attending to future when evaluating
                causal_mask = torch.ones([labels.size(1), labels.size(1)], device=item_ids.device)
                mask_up = torch.triu(causal_mask, diagonal=1)
                temp_perm = (
                    mask_up.expand((labels.size(0), labels.size(1), labels.size(1))) + perm_mask
                )
                perm_mask = (temp_perm > 0).long()
                # the i-th predict corresponds to the i-th token.
                target_mapping = torch.diag(
                    torch.ones(labels.size(1), dtype=torch.float32, device=item_ids.device)
                ).expand((labels.size(0), labels.size(1), labels.size(1)))

            else:
                # predict all next items
                masking_info = self.predict_all(item_ids)
                mask_labels, labels = masking_info.schema, masking_info.targets
                # targets:  the i-th predict corresponds to the i-th item in the sequence.
                target_mapping = torch.nn.functional.one_hot(
                    torch.arange(0, labels.size(1), dtype=torch.long), num_classes=labels.size(1)
                )
                target_mapping = target_mapping.expand(
                    (labels.size(0), labels.size(1), labels.size(1))
                )
                # perm_mask: causal mask
                # Perm mask:
                perm_mask = torch.zeros(
                    (labels.size(0), labels.size(1), labels.size(1)),
                    dtype=torch.float32,
                    device=item_ids.device,
                )
                # add causal mask to avoid attending to future when evaluating
                causal_mask = torch.ones([labels.size(1), labels.size(1)], device=item_ids.device)
                mask_up = torch.triu(causal_mask, diagonal=1)
                temp_perm = (
                    mask_up.expand((labels.size(0), labels.size(1), labels.size(1))) + perm_mask
                )
                perm_mask = (temp_perm > 0).long()

        return mask_labels, labels, target_mapping, perm_mask

    def compute_masked_targets(self, item_ids: torch.Tensor, training=False) -> MaskingInfo:
        (
            self.mask_schema,
            self.masked_targets,
            self.target_mapping,
            self.perm_mask,
        ) = self._compute_masked_targets_extended(item_ids, training=training)

        return MaskingInfo(self.mask_schema, self.masked_targets)

    def transformer_required_arguments(self) -> Dict[str, Any]:
        return dict(target_mapping=self.target_mapping, perm_mask=self.perm_mask)


@masking_registry.register_with_multiple_names("rtd", "replacement")
@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
class ReplacementLanguageModeling(MaskedLanguageModeling):
    """
    Replacement Language Modeling (rtd) you use MLM to randomly select some items, but replace
    them by random tokens.
    Then, a discriminator model (that can share the weights with the generator or not), is asked
    to classify whether the item at each position belongs or not to the original sequence.
    The generator-discriminator architecture was jointly trained using Masked LM and RTD tasks.

    Parameters
    ----------
    {mask_sequence_parameters}
    sample_from_batch: bool
        Whether to sample replacement item ids from the same batch or not
    """

    def __init__(
        self,
        hidden_size: int,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        sample_from_batch: bool = False,
        **kwargs
    ):
        super(ReplacementLanguageModeling, self).__init__(
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            eval_on_last_item_seq_only=eval_on_last_item_seq_only,
            kwargs=kwargs,
        )

        self.sample_from_batch = sample_from_batch

    def get_fake_tokens(self, itemid_seq, target_flat, logits):
        """
        Second task of RTD is binary classification to train the discriminator.
        The task consists of generating fake data by replacing [MASK] positions with random items,
        ELECTRA discriminator learns to detect fake replacements.

        Parameters
        ----------
        itemid_seq: torch.Tensor of shape (bs, max_seq_len)
            input sequence of item ids
        target_flat: torch.Tensor of shape (bs*max_seq_len)
            flattened masked label sequences
        logits: torch.Tensor of shape (#pos_item, vocab_size or #pos_item),
            mlm probabilities of positive items computed by the generator model.
            The logits are over the whole corpus if sample_from_batch = False,
            over the positive items (masked) of the current batch otherwise

        Returns
        -------
        corrupted_inputs: torch.Tensor of shape (bs, max_seq_len)
            input sequence of item ids with fake replacement
        discriminator_labels: torch.Tensor of shape (bs, max_seq_len)
            binary labels to distinguish between original and replaced items
        batch_updates: torch.Tensor of shape (#pos_item)
            the indices of replacement item within the current batch if sample_from_batch is enabled
        """
        # TODO: Generate fake interactions embeddings using metadatainfo in addition to item ids.

        # Replace only items that were masked during MLM
        non_pad_mask = target_flat != self.padding_idx
        pos_labels = torch.masked_select(target_flat, non_pad_mask)
        # Sample random item ids
        if self.sample_from_batch:
            # get batch indices for replacement items
            batch_updates = self.sample_from_softmax(logits).flatten()
            # get item ids based on batch indices
            updates = pos_labels[batch_updates]
        else:
            # get replacement item ids directly from logits over the whole corpus
            updates = self.sample_from_softmax(logits).flatten()
            batch_updates = []

        # Replace masked labels by replacement item ids
        # detach() is needed to not propagate the discriminator loss through generator
        corrupted_labels = (
            target_flat.clone().detach().scatter(-1, non_pad_mask.nonzero().flatten(), updates)
        )
        # Build discriminator label : distinguish original token from replaced one
        discriminator_labels = (corrupted_labels != target_flat).view(-1, itemid_seq.size(1))
        # Build corrupted inputs : replacing [MASK] by sampled item
        corrupted_inputs = (
            itemid_seq.clone()
            .detach()
            .reshape(-1)
            .scatter(-1, non_pad_mask.nonzero().flatten(), updates)
        )

        return (
            corrupted_inputs.view(-1, itemid_seq.size(1)),
            discriminator_labels,
            batch_updates,
        )

    def sample_from_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sampling method for replacement token modeling (ELECTRA)

        Parameters
        ----------
        logits: torch.Tensor(pos_item, vocab_size)
            scores of probability of masked positions returned  by the generator model

        Returns
        -------
        samples: torch.Tensor(#pos_item)
            ids of replacements items.
        """
        # add noise to logits to prevent from the case where the generator learn to exactly
        # retrieve the true item that was masked
        uniform_noise = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-9) + 1e-9)
        s = logits + gumbel_noise

        return torch.argmax(torch.nn.functional.softmax(s, dim=-1), -1)
