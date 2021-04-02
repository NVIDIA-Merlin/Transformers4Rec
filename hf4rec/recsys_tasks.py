"""
A meta class to support various language modeling pre-training tasks.
"""
import torch


class RecSysTask:
    """
    pad_token: Id of padding item
    device: the device used for RecSysMetaModel
    training: bool whether we generate data for training (True) or evaluation process
    """

    def __init__(self, pad_token, device, training):
        self.pad_token = pad_token
        self.device = device
        self.training = training

    def mask_tokens(self, itemid_seq, mlm_probability):
        """
            prepare sequence with mask for masked language modeling prediction
            the function is based on HuggingFace's transformers/data/data_collator.py 

            INPUT:
            itemid_seq: sequence of input itemid (label) column
            mlm_probability: probability of an item to be selected (masked) to be a label for this sequence. P.s. We enforce that at least one item is masked for each sequence, so that the network can learn something with it.

            OUTPUT:
            labels: item id sequence as label
            masked_labels: bool mask with is true only for masked labels (targets)
            """

        # labels = itemid_seq.clone()
        labels = torch.full(
            itemid_seq.shape, self.pad_token, dtype=itemid_seq.dtype, device=self.device
        )
        non_padded_mask = itemid_seq != self.pad_token

        rows_ids = torch.arange(
            itemid_seq.size(0), dtype=torch.long, device=self.device
        )
        # During training, masks labels to be predicted according to a probability, ensuring that each session has at least one label to predict
        if self.training:
            # Selects a percentage of items to be masked (selected as labels)
            probability_matrix = torch.full(
                itemid_seq.shape, mlm_probability, device=self.device
            )
            masked_labels = torch.bernoulli(probability_matrix).bool() & non_padded_mask
            labels = torch.where(
                masked_labels, itemid_seq, torch.full_like(itemid_seq, self.pad_token),
            )

            # Set at least one item in the sequence to mask, so that the network can learn something with this session
            one_random_index_by_session = torch.multinomial(
                non_padded_mask.float(), num_samples=1
            ).squeeze()
            labels[rows_ids, one_random_index_by_session] = itemid_seq[
                rows_ids, one_random_index_by_session
            ]
            masked_labels = labels != self.pad_token

            # If a sequence has only masked labels, unmasks one of the labels
            sequences_with_only_labels = masked_labels.sum(
                axis=1
            ) == non_padded_mask.sum(axis=1)
            sampled_labels_to_unmask = torch.multinomial(
                masked_labels.float(), num_samples=1
            ).squeeze()

            labels_to_unmask = torch.masked_select(
                sampled_labels_to_unmask, sequences_with_only_labels
            )
            rows_to_unmask = torch.masked_select(rows_ids, sequences_with_only_labels)

            labels[rows_to_unmask, labels_to_unmask] = self.pad_token
            masked_labels = labels != self.pad_token

            # Logging the real percentage of masked items (labels)
            # perc_masked_labels = masked_labels.sum() / non_padded_mask.sum().float()
            # logger.info(f"  % Masked items as labels: {perc_masked_labels}")

        # During evaluation always masks the last item of the session
        else:
            last_item_sessions = non_padded_mask.sum(axis=1) - 1
            labels[rows_ids, last_item_sessions] = itemid_seq[
                rows_ids, last_item_sessions
            ]
            masked_labels = labels != self.pad_token

        return labels, masked_labels

    def plm_mask_tokens(self, itemid_seq, max_span_length, plm_probability):

        """
        Prepare the attention masks needed for partial-prediction permutation language modeling
        The function is based on HuggingFace's transformers/data/data_collator.py

        INPUT:
        itemid_seq: sequence of input itemid (label) column
        plm_probability: The ratio of surrounding items to unmask to define the context of the span-based prediction segment of items
        max_span_length:  The maximum length of the span of items to predict

        OUTPUT:
        labels: item id sequence as labels
        perm_mask: shape (bs, seq_len, seq_len) : Define  the random factorization order attention mask for each target
        target_mapping: (bs, seq_len, seq_len)  : Binary mask to specify the items to predict
        """

        labels = torch.full(
            itemid_seq.shape, self.pad_token, dtype=itemid_seq.dtype, device=self.device
        )
        non_padded_mask = itemid_seq != self.pad_token

        rows_ids = torch.arange(
            itemid_seq.size(0), dtype=torch.long, device=self.device
        )
        masked_labels = torch.full(
            labels.shape, 0, dtype=torch.bool, device=self.device
        )
        # During training, masks a span of consecutive items to be predicted according to plm_probability,
        # While  ensuring that each session has at least one  item to predict
        if self.training:
            target_mapping = torch.zeros(
                (labels.size(0), labels.size(1), labels.size(1)),
                dtype=torch.float32,
                device=self.device,
            )
            perm_mask = torch.zeros(
                (labels.size(0), labels.size(1), labels.size(1)),
                dtype=torch.float32,
                device=self.device,
            )
            # For each session select a span of consecutive item ids to be masked
            for i in range(labels.size(0)):
                # Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
                cur_len = 0
                max_len = non_padded_mask.sum(1)[i]  # mask only non-padded items
                while cur_len < max_len:
                    # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
                    span_length = torch.randint(1, max_span_length + 1, (1,)).item()
                    # Reserve a context of length `context_length = span_length / plm_probability` to surround span to be masked
                    context_length = int(span_length / plm_probability)
                    # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]`
                    start_index = (
                        cur_len
                        + torch.randint(context_length - span_length + 1, (1,)).item()
                    )
                    if start_index < max_len:
                        # Mask the span of non-padded items `start_index:start_index + span_length`
                        masked_labels[i, start_index : start_index + span_length] = 1
                    # Set `cur_len = cur_len + context_length`
                    cur_len += context_length
                # if no item was masked:
                if masked_labels[i].sum() == 0:
                    # Set at least one item in the sequence to mask, so that the network can learn something with this session
                    one_random_index_by_session = torch.multinomial(
                        non_padded_mask[i].float(), num_samples=1
                    ).squeeze()
                    masked_labels[i, one_random_index_by_session] = itemid_seq[
                        i, one_random_index_by_session
                    ]
                # Since we're replacing non-masked tokens with pad_tokens in the labels tensor instead of skipping them altogether,
                # the i-th predict corresponds to the i-th token.
                # N.B: the loss function will be computed only on non paded items
                target_mapping[i] = torch.eye(labels.size(1))

            labels = torch.where(
                masked_labels, itemid_seq, torch.full_like(itemid_seq, self.pad_token)
            )

            for i in range(labels.size(0)):
                # Generate permutation indices i.e.
                #  sample a random factorisation order for the sequence.
                #  This will determine which tokens a given token can attend to
                # (encoded in `perm_mask`).
                # Create a linear factorisation order
                perm_index = torch.arange(
                    labels.size(1), dtype=torch.long, device=self.device
                )
                # randomly permute indices of each session
                perm_index = perm_index[torch.randperm(labels.size(1))]
                # Set the permutation indices of non-masked (non-functional) tokens to the
                # smallest index (-1) so that:
                # (1) They can be seen by all other positions
                # (2) They cannot see masked positions, so there won't be information leak
                perm_index.masked_fill_(~masked_labels[i], -1)
                # The logic for whether the i-th token can attend on the j-th token
                # based on the factorisation order:
                # 0 (can attend):
                # If perm_index[i] > perm_index[j] or j is neither masked nor a padded item
                # 1 (cannot attend):
                # If perm_index[i] <= perm_index[j] and j is either masked or a padded item
                perm_mask[i] = (
                    perm_index.reshape((labels.size(1), 1))
                    <= perm_index.reshape((1, labels.size(1)))
                ) & masked_labels[i]
        # During evaluation always mask the last item of the session
        else:
            last_item_sessions = non_padded_mask.sum(axis=1) - 1
            labels[rows_ids, last_item_sessions] = itemid_seq[
                rows_ids, last_item_sessions
            ]
            masked_labels = labels != self.pad_token
            perm_mask = torch.zeros(
                (labels.size(0), labels.size(1), labels.size(1)),
                dtype=torch.float32,
                device=self.device,
            )
            # Previous tokens don't see last non-padded token
            perm_mask[rows_ids, :, last_item_sessions] = 1
            # add causal mask to avoid attending to future when evaluating
            causal_mask = torch.ones(
                [labels.size(1), labels.size(1)], device=self.device
            )
            mask_up = torch.triu(causal_mask, diagonal=1)
            temp_perm = (
                mask_up.expand((labels.size(0), labels.size(1), labels.size(1)))
                + perm_mask
            )
            perm_mask = (temp_perm > 0).long()
            # the i-th predict corresponds to the i-th token.
            target_mapping = torch.diag(
                torch.ones(labels.size(1), dtype=torch.float32, device=self.device)
            ).expand((labels.size(0), labels.size(1), labels.size(1)))

        return labels, masked_labels, target_mapping, perm_mask

    def sample_from_softmax(self, logits):
        """
        Sampling method for replacement token modeling (ELECTRA): 
        Args:
            logits: of shape (pos_item, vocab_size), mlm probabilities computed by the generator model 
        """
        # add noise to logits to prevent from the case where the generator learn to exactly retrieve the true
        # item that was masked
        uniform_noise = torch.rand(
            logits.shape, dtype=torch.float32, device=self.device
        )
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-9) + 1e-9)
        s = logits + gumbel_noise
        return torch.argmax(torch.nn.functional.softmax(s, dim=-1), -1)

    def get_fake_data(self, emb_inp, target_flat, logits, embedding_table):
        """
        Generate fake data by replacing [MASK] items to train ELECTRA discriminator 
        Args:
            emb_inp: (bs, seq_len, embedding_dim) The embeddings of the input items 
            target_flat: (#pos_item,) The ids of positive items 
            logits: of shape (#pos_item, vocab_size), mlm probabilities of positive items computed by the generator model 
            embedding_table: Generator and discriminator shares the same item embedding table 
        """
        # Sample random item ids
        updates = self.sample_from_softmax(logits).flatten()
        # Replace only items that were masked during MLM
        non_pad_mask = target_flat != self.pad_token
        # Replace [MASK] by random item ids samples from mlm_logits
        corrupted_labels = target_flat.scatter(
            -1, non_pad_mask.nonzero().flatten(), updates
        )
        # Build discriminator label : distinguish orginal token from replaced one
        discriminator_labels = (corrupted_labels != target_flat).view(
            -1, emb_inp.size(1)
        )
        # Build corrupted item embedding input
        emb_updates = embedding_table(updates)
        corrupted_emb_inp = emb_inp.view(-1, emb_inp.size(2))
        corrupted_emb_inp[non_pad_mask.nonzero().flatten(), :] = emb_updates
        return (
            corrupted_emb_inp.view(emb_inp.shape),
            discriminator_labels,
        )

