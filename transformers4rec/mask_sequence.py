"""
A meta class to support various language modeling pre-training tasks.
"""
import torch
from torch import nn 


class MaskSequence(nn.Module):
    """
    Module to prepare masked data for LM tasks 
    
    Parameters: 
    ----------
        task: the LM task : ['clm', 'mlm', 'plm', 'rtd']
        pad_token: index of padding.
        device: either 'cpu' or 'cuda' device.
        training: bool whether we generate data for training (True) or evaluation (False).
    """

    def __init__(self,
                 task: str = 'mlm',
                 hidden_size: int= 128,
                 pad_token: int = 0,
                 device: str = 'cuda'):
        super(MaskSequence, self).__init__()
        
        self.task = task        
        self.pad_token = pad_token
        self.device = device
        self.hidden_size = hidden_size
        
        self.max_span_length = 20
        self.mlm_probability = 0.15
        self.plm_probability = 0.15
        self.plm_permute_all = False
        self.sample_from_batch = False
        
        # Creating a trainable embedding for masking inputs for Masked LM
        self.masked_item_embedding = nn.Parameter(torch.Tensor(self.hidden_size)).to(self.device)

        nn.init.normal_( 
            self.masked_item_embedding, mean=0, std=0.001,
        )

        
    def forward(self, pos_emb, itemid_seq, training): 
        plm_target_mapping, plm_perm_mask = None, None
        
        if self.task == 'mlm':
            labels, mask_labels = self.mlm_mask_tokens(itemid_seq, training, self.mlm_probability)
            pos_emb_inp = torch.where(
                        mask_labels.unsqueeze(-1).bool(),
                        self.masked_item_embedding.to(pos_emb.dtype),
                        pos_emb,
                    )
            
        elif self.task == 'plm': 
            labels, mask_labels, plm_target_mapping, plm_perm_mask = self.plm_mask_tokens(itemid_seq,
                                                                           training,
                                                                           self.max_span_length,
                                                                           self.plm_probability,
                                                                           self.plm_permute_all,
                                                                          )
            pos_emb_inp = torch.where(
                    mask_labels.unsqueeze(-1).bool(),
                    self.masked_item_embedding.to(pos_emb.dtype),
                    pos_emb,
                )
            
        elif self.task == 'clm': 
            labels, mask_labels = self.clm_mask_tokens(itemid_seq, training)
            pos_emb_inp = pos_emb[:, :-1]
            # As after shifting the sequence length will be subtracted by one, adding a masked item in
            # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
            pos_emb_inp = torch.cat(
                [
                    pos_emb_inp,
                    torch.zeros(
                        (pos_emb_inp.shape[0], 1, pos_emb_inp.shape[2]),
                        dtype=pos_emb_inp.dtype,
                    ).to(self.device),
                ],
                axis=1,
            )

            # Replacing the inputs corresponding to masked label with a trainable embedding
            pos_emb_inp = torch.where(
                mask_trg_pad.unsqueeze(-1).bool(),
                pos_emb_inp,
                self.masked_item_embedding.to(pos_emb_inp.dtype),
            )
                
        
        elif self.task == 'rtd':
            # Generate mask labels for the ELECTRA generator task 
            labels, mask_labels = self.mlm_mask_tokens(itemid_seq, self.mlm_probability)
            pos_emb_inp = torch.where(
            mask_labels.unsqueeze(-1).bool(),
            self.masked_item_embedding.to(pos_emb.dtype),
            pos_emb,
        )
            
        return pos_emb_inp, labels, mask_labels, plm_target_mapping, plm_perm_mask
            
    
    def get_fake_tokens(self, itemid_seq, target_flat, logits):
        """
        Generate fake data by replacing [MASK] positions by random items to train ELECTRA discriminator
        
        INPUT:
        -----
            itemid_seq: (bs, max_seq_len), input sequence of item ids
            target_flat: (bs*max_seq_len), flattened masked label sequences
            logits: (#pos_item, vocab_size or #pos_item),
                    mlm probabilities of positive items computed by the generator model.
                    The logits are over the whole corpus if sample_from_batch = False,
                    over the positive items (masked) of the current batch otherwise
        OUTPUT:
        ------
            corrupted_inputs: (bs, max_seq_len) input sequence of item ids with fake reprelacement
            discriminator_labels: (bs, max_seq_len) binary labels to distinguish between original and replaced items
            batch_updates: (#pos_item) the indices of replacement item within the current batch if sample_from_batch is enabled
        """
        # Replace only items that were masked during MLM
        non_pad_mask = target_flat != self.pad_token
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
            target_flat.clone()
            .detach()
            .scatter(-1, non_pad_mask.nonzero().flatten(), updates)
        )
        # Build discriminator label : distinguish orginal token from replaced one
        discriminator_labels = (corrupted_labels != target_flat).view(
            -1, input_ids.size(1)
        )
        # Build corrupted inputs : replacing [MASK] by sampled item
        corrupted_inputs = (
            input_ids.clone()
            .detach()
            .reshape(-1)
            .scatter(-1, non_pad_mask.nonzero().flatten(), updates)
        )
        return (
            corrupted_inputs.view(-1, input_ids.size(1)),
            discriminator_labels,
            batch_updates,
        )

    
    
    def clm_mask_tokens(self, itemid_seq, training, train_on_last_item_seq_only=False, eval_on_last_item_seq_only=True): 
        """
        Causal Language Modeling - Prepare labels for Next token predictions 
        """

        labels = itemid_seq[:, 1:]
        # As after shifting the sequence length will be subtracted by one, adding a masked item in
        # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
        labels = torch.cat(
            [
                labels,
                torch.zeros(
                    (labels.shape[0], 1), dtype=labels.dtype
                ).to(self.device),
            ],
            axis=-1,
        )
        mask_labels = labels != self.pad_token
        
        if (eval_on_last_item_seq_only and not training) or (
                train_on_last_item_seq_only and training
            ): 
            
            rows_ids = torch.arange(
                    labels.size(0), dtype=torch.long, device=self.device
                )
            last_item_sessions = mask_labels.sum(axis=1) - 1
            
            label_seq_trg_eval = torch.zeros(
                    labels.shape, dtype=torch.long, device=self.device
                )
            label_seq_trg_eval[rows_ids, last_item_sessions] = labels[
                    rows_ids, last_item_sessions
                ]
            # Updating labels and mask
            labels = label_seq_trg_eval
            mask_labels = label_seq_trg != self.pad_token
            
        return labels, mask_labels
        
        
    def mlm_mask_tokens(self, itemid_seq, training, mlm_probability):
        """
        prepare sequence with mask for masked language modeling prediction
        the function is based on HuggingFace's transformers/data/data_collator.py

        INPUTS:
        -----
        itemid_seq: sequence of input itemid (label) column
        mlm_probability: probability of an item to be selected (masked) to be a label for this sequence. P.s. We enforce that at least one item is masked for each sequence, so that the network can learn something with it.

        OUTPUTS:
        ------
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
        if training:
            # Selects a percentage of items to be masked (selected as labels)
            probability_matrix = torch.full(
                itemid_seq.shape, mlm_probability, device=self.device
            )
            masked_labels = torch.bernoulli(probability_matrix).bool() & non_padded_mask
            labels = torch.where(
                masked_labels,
                itemid_seq,
                torch.full_like(itemid_seq, self.pad_token),
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

    def plm_mask_tokens(
        self, itemid_seq, training, max_span_length, plm_probability, plm_permute_all
    ):
        """
        Prepare the attention masks needed for partial-prediction permutation language modeling
        The function is based on HuggingFace's transformers/data/data_collator.py

        INPUT:
        itemid_seq: sequence of input itemid (label) column
        plm_probability: The ratio of surrounding items to unmask to define the context of the span-based prediction segment of items
        max_span_length:  The maximum length of the span of items to predict
        plm_permute_all: compute permutation for all non paded itemids

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
        if training:
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
            if plm_permute_all:
                # Permute all non padded items
                masked_labels = non_padded_mask
            else:
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
                            + torch.randint(
                                context_length - span_length + 1, (1,)
                            ).item()
                        )
                        if start_index < max_len:
                            # Mask the span of non-padded items `start_index:start_index + span_length`
                            masked_labels[
                                i, start_index : start_index + span_length
                            ] = 1
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
        INPUT:
            logits: (pos_item, vocab_size), mlm probabilities computed by the generator model
        OUTPUT:
            samples: (#pos_item), ids of replacements items
        """
        # add noise to logits to prevent from the case where the generator learn to exactly retrieve the true
        # item that was masked
        uniform_noise = torch.rand(logits.shape, dtype=logits.dtype, device=self.device)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-9) + 1e-9)
        s = logits + gumbel_noise
        return torch.argmax(torch.nn.functional.softmax(s, dim=-1), -1)
