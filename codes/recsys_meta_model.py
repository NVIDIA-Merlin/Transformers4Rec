"""
A meta class supports various (Huggingface) transformer models for RecSys tasks.

"""

import logging
from collections import OrderedDict

import math
import torch
from torch import nn
from torch.nn import functional as F

from transformers import (GPT2Model, PreTrainedModel)

from loss_functions import TOP1, TOP1_max, BPR, BPR_max, BPR_max_reg

import numpy as np

logger = logging.getLogger(__name__)

torch.manual_seed(0)


class AttnMerge(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W1 = nn.ModuleList([nn.Linear(input_dim, input_dim)] * output_dim)
        self.output_dim = output_dim 
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inp):
        out = []
        for i in range(self.output_dim):
            attn_weight = self.softmax(self.W1[i](inp))
            out.append(torch.mul(attn_weight,inp).sum(-1))
        return torch.stack(out, dim=-1)


def get_embedding_size_from_cardinality(cardinality, multiplier=2):
    # A rule-of-thumb from Google.
    embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))
    return embedding_size


class RecSysMetaModel(PreTrainedModel):
    """
    vocab_sizes : sizes of vocab for each discrete inputs
        e.g., [product_id_vocabs, category_vocabs, etc.]
    """
    def __init__(self, model, config, model_args, data_args, feature_map):
        super(RecSysMetaModel, self).__init__(config)
        
        self.model = model
        #Temporary, replacing the Attention block of GPT-2 for one that enforces mask future informatino
        #if type(self.model) is GPT2Model:
        #    for block in self.model.h:
        #        block.attn = AttentionFixed(128, 20, config, True)

        """
        if self.model.__class__ in [nn.GRU, nn.LSTM, nn.RNN]:
            self.is_rnn = True
        else:
            self.is_rnn = False
        """

        self.feature_map = feature_map
        """
        feature_map : 
        {
            cname: {
                'dtype': categorial / long / float
                'cardinality' : num_items
                'is_label' : True, False
            }
        }
        """

        self.pad_token = data_args.pad_token
        self.mask_token = data_args.mask_token
        self.embedding_tables = nn.ModuleDict()

        self.session_aware = data_args.session_aware
        self.session_aware_features_prefix = data_args.session_aware_features_prefix
        
        self.col_prefix_neg = data_args.feature_prefix_neg_sample

        self.use_ohe_item_ids_inputs = model_args.use_ohe_item_ids_inputs

        self.loss_scale_factor = model_args.loss_scale_factor

        self.mf_constrained_embeddings = model_args.mf_constrained_embeddings
        
        #For the SIGIR paper experiments
        self.mf_constrained_embeddings_disable_bias = model_args.mf_constrained_embeddings_disable_bias
        self.item_embedding_with_dmodel_dim = model_args.item_embedding_with_dmodel_dim

        self.constrained_embeddings = model_args.constrained_embeddings
        self.negative_sampling = model_args.negative_sampling

        if self.mf_constrained_embeddings and self.constrained_embeddings:
            raise ValueError('You cannot enable both --mf_constrained_embeddings and --constrained_embeddings.')

        concat_input_dim = 0
        target_dim = -1

        self.label_feature = None

        # set embedding tables
        for cname, cinfo in self.feature_map.items():
            # if self.col_prefix_neg not in cname:

            if self.col_prefix_neg not in cname:
                # Ignoring past features to define embedding tables
                if (self.session_aware and cname.startswith(self.session_aware_features_prefix)):
                    continue

                if cinfo['dtype'] == 'categorical':
                    if self.use_ohe_item_ids_inputs:
                        feature_size = cinfo['cardinality']
                    else:
                        if 'is_label' in cinfo and cinfo['is_label']:
                            self.label_embedding_table_name = cinfo['emb_table']

                        if ('is_label' in cinfo and cinfo['is_label']) and \
                           (model_args.constrained_embeddings or model_args.mf_constrained_embeddings or \
                            model_args.item_embedding_with_dmodel_dim):
                            embedding_size = model_args.d_model
                        else:                      
                            embedding_size = get_embedding_size_from_cardinality(cinfo['cardinality'])                        
                        

                        feature_size = embedding_size
                        self.embedding_tables[cinfo['emb_table']] = nn.Embedding(
                            cinfo['cardinality'], 
                            embedding_size, 
                            padding_idx=self.pad_token
                        ).to(self.device)

                        # Added to initialize embeddings to small weights
                        self.embedding_tables[cinfo['emb_table']].weight.data.normal_(0., 1./math.sqrt(embedding_size))
                    
                    logger.info('Categ Feature: {} - Cardinality: {} - Feature Size: {}'.format(cname, cinfo['cardinality'], feature_size))

                    concat_input_dim += feature_size
                elif cinfo['dtype'] in ['long', 'float']:
                    logger.info('Numerical Feature: {} - Feature Size: 1'.format(cname))

                    concat_input_dim += 1
                elif cinfo['is_control']:
                    #Control features are not used as input for the model
                    continue
                else:
                    raise NotImplementedError

                if 'is_label' in cinfo and cinfo['is_label']:
                    target_dim = cinfo['cardinality']

        if target_dim == -1:
            raise RuntimeError('label column is not declared in feature map.')

        self.neg_sampling_store_size = model_args.neg_sampling_store_size
        self.neg_sampling_extra_samples_per_batch = model_args.neg_sampling_extra_samples_per_batch
        self.neg_sampling_alpha = model_args.neg_sampling_alpha

        self.inp_merge = model_args.inp_merge
        if self.inp_merge == 'mlp':
            self.mlp_merge = nn.Linear(concat_input_dim, model_args.d_model).to(self.device)
        elif self.inp_merge == 'attn':
            self.attn_merge = AttnMerge(concat_input_dim, model_args.d_model).to(self.device)
        else:
            raise NotImplementedError

        self.layernorm1 = nn.LayerNorm(normalized_shape = concat_input_dim)
        self.layernorm2 = nn.LayerNorm(normalized_shape = model_args.d_model)

        self.eval_on_last_item_seq_only = model_args.eval_on_last_item_seq_only
        self.train_on_last_item_seq_only = model_args.train_on_last_item_seq_only

        self.total_seq_length = data_args.total_seq_length
        self.n_layer = model_args.n_layer

        self.mlm = model_args.mlm
        self.mlm_probability = model_args.mlm_probability

        # Creating a trainable embedding for masking inputs for Masked LM
        self.masked_item_embedding = nn.Parameter(torch.Tensor(model_args.d_model)).to(self.device)

        # nn.init.normal_(self.masked_item_embedding, mean = 0, std = 0.4)
        # nn.init.normal_(self.masked_item_embedding, mean = 0, std = 1./math.sqrt(model_args.d_model))
        nn.init.normal_(self.masked_item_embedding, mean = 0, std = 0.01)

        self.target_dim = target_dim
        self.similarity_type = model_args.similarity_type
        self.margin_loss = model_args.margin_loss

        self.output_layer = nn.Linear(model_args.d_model, target_dim).to(self.device)
        self.loss_type = model_args.loss_type
        self.log_softmax = nn.LogSoftmax(dim = -1)

        self.output_layer_bias = nn.Parameter(torch.Tensor(target_dim)).to(self.device)
        nn.init.zeros_(self.output_layer_bias)

        self.loss_nll = nn.NLLLoss(ignore_index = self.pad_token)

        if self.loss_type == 'cross_entropy_neg':
            self.loss_fn = nn.NLLLoss()
        elif self.loss_type == 'cross_entropy_neg_1d':
            self.loss_fn = nll_1d
        elif self.loss_type.startswith('margin_hinge'):
            # https://pytorch.org/docs/master/generated/torch.nn.CosineEmbeddingLoss.html
            self.loss_fn = nn.CosineEmbeddingLoss(margin = model_args.margin_loss, reduction = 'sum')

        elif self.loss_type == 'top1':
            self.loss_fn = TOP1()

        elif self.loss_type == 'top1_max':
            self.loss_fn = TOP1_max()

        elif self.loss_type == 'bpr':
            self.loss_fn = BPR()

        #elif self.loss_type == 'bpr_max':
        #    self.loss_fn = BPR_max()

        elif self.loss_type == 'bpr_max_reg':
            self.loss_fn = BPR_max_reg(lambda_ = model_args.bpr_max_reg_lambda)

        elif self.loss_type != 'cross_entropy':
            raise NotImplementedError

        if model_args.model_type == 'reformer':
            tf_out_size = model_args.d_model * 2
        else:
            tf_out_size = model_args.d_model

        self.transformer_output_project = nn.Linear(tf_out_size, model_args.d_model).to(self.device)

        if self.similarity_type in ['concat_mlp', 'multi_mlp']:
            m_factor = 2 if self.similarity_type == 'concat_mlp' else 1
            self.sim_mlp = nn.Sequential(
                OrderedDict([
                    ('linear0', nn.Linear(model_args.d_model * m_factor , model_args.d_model).to(self.device)),
                    ('relu0', nn.LeakyReLU()),
                    ('linear1', nn.Linear(model_args.d_model, model_args.d_model // 2).to(self.device)),
                    ('relu1', nn.LeakyReLU()),
                    ('linear2', nn.Linear(model_args.d_model // 2, model_args.d_model // 4).to(self.device)),
                    ('relu2', nn.LeakyReLU()),
                    ('linear3', nn.Linear(model_args.d_model // 4, 1).to(self.device)),
                    ('sigmoid', nn.Sigmoid()),
                ]       
            ))
        
        self.all_rescale_factor = model_args.all_rescale_factor
        self.neg_rescale_factor = model_args.neg_rescale_factor

        if model_args.tf_out_activation == 'tanh':
            self.tf_out_act = torch.tanh
        elif model_args.tf_out_activation == 'relu':
            self.tf_out_act = torch.relu

        self.disable_positional_embeddings = model_args.disable_positional_embeddings        

    def forward(self, *args, **kwargs):
        inputs = kwargs


        #print('DEVICE={} - input device: {} - INPUTS: {}'.format(self.device, inputs['sess_pid_seq'].device, inputs['sess_pid_seq'][:2,:5].cpu().numpy()))

        # Step1. Unpack inputs, get embedding, and concatenate them
        label_seq = None
        
        max_feature_seq_len = None

        pos_inp, label_seq, max_feature_seq_len, metadata_for_pred_logging = self.feature_process(inputs)
        #if self.loss_type != 'cross_entropy':
        #    neg_inp, _, _, _ = self.feature_process(inputs, max_feature_seq_len, is_neg=True)

        assert label_seq is not None, 'label sequence is not declared in feature_map'

        # To mark past sequence labels
        if self.session_aware:
            masked_past_session = torch.zeros_like(label_seq, dtype=torch.long, device=self.device)

        if self.mlm:
            """
            Masked Language Model
            """
            label_seq_trg, label_mlm_mask = self.mask_tokens(
                label_seq, self.mlm_probability)
            
            #To mark past sequence labels
            if self.session_aware:
                label_seq_trg = torch.cat([masked_past_session, label_seq_trg], axis=1)
                label_mlm_mask = torch.cat([masked_past_session.bool(), label_mlm_mask], axis=1)
            
        else:
            """
            Predict Next token
            """

            label_seq_inp = label_seq[:, :-1]
            label_seq_trg = label_seq[:, 1:]

            # As after shifting the sequence length will be subtracted by one, adding a masked item in 	
            # the sequence to return to the initial sequence. This is important for ReformerModel(), for example	
            label_seq_inp = torch.cat([label_seq_inp, 	
                        torch.zeros((label_seq_inp.shape[0], 1), dtype=label_seq_inp.dtype).to(self.device)], axis=-1)	
            label_seq_trg = torch.cat([label_seq_trg, 	
                        torch.zeros((label_seq_trg.shape[0], 1), dtype=label_seq_trg.dtype).to(self.device)], axis=-1)
            
            # apply mask on input where target is on padding token
            mask_trg_pad = (label_seq_trg != self.pad_token)

            label_seq_inp = label_seq_inp * mask_trg_pad

            #When evaluating, computes metrics only for the last item of the session
            if (self.eval_on_last_item_seq_only and not self.training) or \
               (self.train_on_last_item_seq_only and self.training):
                rows_ids = torch.arange(label_seq_inp.size(0), dtype=torch.long, device=self.device)
                last_item_sessions = mask_trg_pad.sum(axis=1) - 1
                label_seq_trg_eval = torch.zeros(label_seq_trg.shape, dtype=torch.long, device=self.device)
                label_seq_trg_eval[rows_ids, last_item_sessions] = label_seq_trg[rows_ids, last_item_sessions]
                #Updating labels and mask
                label_seq_trg = label_seq_trg_eval
                mask_trg_pad = (label_seq_trg != self.pad_token)

            #To mark past sequence labels
            if self.session_aware:
                label_seq_trg_original = label_seq_trg.clone()
                label_seq_trg = torch.cat([masked_past_session, label_seq_trg], axis=1)
                mask_trg_pad = torch.cat([masked_past_session, mask_trg_pad], axis=1)

        # Creating an additional feature with the position in the sequence
        metadata_for_pred_logging['seq_pos'] = torch.arange(1, label_seq.shape[1] + 1, device = self.device).repeat(
            label_seq.shape[0], 1)
        metadata_for_pred_logging['seq_len'] = (label_seq != self.pad_token).int().sum(axis = 1).unsqueeze(-1).repeat(1,
                                                                                                                      label_seq.shape[
                                                                                                                          1])
        # Keeping only metadata features for the next-clicks (targets)
        if not (self.mlm and self.training):
            for feat_name in metadata_for_pred_logging:
                metadata_for_pred_logging[feat_name] = metadata_for_pred_logging[feat_name][:, 1:]

                # As after shifting the sequence length will be subtracted by one, adding a masked item in 	
                # the sequence to return to the initial sequence. This is important for ReformerModel(), for example	
                metadata_for_pred_logging[feat_name] = torch.cat([metadata_for_pred_logging[feat_name], 	
                        torch.zeros((metadata_for_pred_logging[feat_name].shape[0], 1), dtype=metadata_for_pred_logging[feat_name].dtype).to(self.device)], axis=-1)

        # Step 2. Merge features

        if self.inp_merge == 'mlp':
            # pos_emb = self.tf_out_act(self.mlp_merge(pos_inp))

            pos_inp = self.layernorm1(pos_inp)
            pos_emb = self.tf_out_act(self.layernorm2(self.mlp_merge(pos_inp)))

            # if self.loss_type != 'cross_entropy':
            #    neg_emb = torch.tanh(self.mlp_merge(neg_inp))
        elif self.inp_merge == 'attn':
            pos_emb = self.attn_merge(pos_inp)
            # if self.loss_type != 'cross_entropy':
            #    neg_emb = self.attn_merge(neg_inp)

        if self.mlm:
            # Masking inputs (with trainable [mask] embedding]) at masked label positions      
            pos_emb_inp = torch.where(label_mlm_mask.unsqueeze(-1).bool(),
                                      self.masked_item_embedding.to(pos_emb.dtype),
                                      pos_emb)
            # if self.loss_type != 'cross_entropy':
            #    pos_emb_trg = pos_emb #.clone()
            #    neg_emb_inp = neg_emb   
        else:
            # slice over time-steps for input and target and ensuring masking is applied
            # pos_emb_inp = pos_emb[:, :-1] * mask_trg_pad.unsqueeze(-1)

            # Truncating the input sequences length to -1
            pos_emb_inp = pos_emb[:, :-1]

            # As after shifting the sequence length will be subtracted by one, adding a masked item in 	
            # the sequence to return to the initial sequence. This is important for ReformerModel(), for example	
            pos_emb_inp = torch.cat([pos_emb_inp, 	
                        torch.zeros((pos_emb_inp.shape[0], 1, pos_emb_inp.shape[2]), 	
                                     dtype=pos_emb_inp.dtype).to(self.device)], axis=1)

            # Replacing the inputs corresponding to masked label with a trainable embedding
            pos_emb_inp = torch.where(mask_trg_pad.unsqueeze(-1).bool(),
                                      pos_emb_inp,
                                      self.masked_item_embedding.to(pos_emb_inp.dtype))

            '''
            if type(self.model) is GPT2Model:
                #Temporary hack to test if the last shifted item leaks and improves last item prediction accuracy
                pos_emb_inp = pos_emb[:, :-1]
            else:
                pos_emb_inp = pos_emb[:, :-1] * mask_trg_pad.unsqueeze(-1)
            '''
            # if self.loss_type != 'cross_entropy':
                # # Shifting labels by one	
                # pos_emb_trg = pos_emb[:, 1:]	
                # neg_emb_inp = neg_emb[:, :-1]	
                # # As after shifting the sequence length will be subtracted by one, adding a masked item in 	
                # # the sequence to return to the initial sequence. This is important for ReformerModel(), for example	
                # pos_emb_trg = torch.cat([pos_emb_trg, 	
                #             torch.zeros((pos_emb_trg.shape[0], 1, pos_emb_trg.shape[2]), 	
                #                         dtype=pos_emb_trg.dtype).to(self.device)], axis=1)	
                # neg_emb_inp = torch.cat([neg_emb_inp, 	
                #             torch.zeros((neg_emb_inp.shape[0], 1, neg_emb_inp.shape[2]), 	
                #                         dtype=neg_emb_inp.dtype).to(self.device)], axis=1)	
                # pos_emb_trg = pos_emb_trg * mask_trg_pad.unsqueeze(-1)	
                # neg_emb_inp = neg_emb_inp * mask_trg_pad.unsqueeze(-1).unsqueeze(-1)

        # Step3. Run forward pass on model architecture

        if not isinstance(self.model, PreTrainedModel):  # Checks if its a transformer
            # compute output through RNNs
            results = self.model(
                input = pos_emb_inp
            )

            if type(results) is tuple or type(results) is list:
                pos_emb_pred = results[0]
            else:
                pos_emb_pred = results

            model_outputs = (None,)

        else:
            """
            Transformer Models
            """
            '''
            if self.disable_positional_embeddings:
                position_ids = torch.zeros(max_seq_len-1, requires_grad=False, dtype=torch.long, device=self.device)
            else:
                position_ids = None
            '''

            # label_seq_inp_ohe = torch.nn.functional.one_hot(label_seq_inp, self.target_dim)

            if type(self.model) is GPT2Model:
                seq_len = pos_emb_inp.shape[1]
                # head_mask has shape n_layer x batch x n_heads x N x N
                head_mask = torch.tril(torch.ones((seq_len, seq_len), dtype = torch.uint8, device = self.device)) \
                    .view(1, 1, 1, seq_len, seq_len) \
                    .repeat(self.n_layer, 1, 1, 1, 1)

                model_outputs = self.model(
                    # Temporary, to see if the problem of hard attention is related to the item embedding generation
                    # input_ids=label_seq_inp,

                    inputs_embeds = pos_emb_inp,
                    head_mask = head_mask,
                    # position_ids=position_ids
                )
            else:
                model_outputs = self.model(
                    inputs_embeds = pos_emb_inp,
                    # position_ids=position_ids
                )

            pos_emb_pred = model_outputs[0]
            model_outputs = tuple(model_outputs[1:])

        pos_emb_pred = self.tf_out_act(self.transformer_output_project(pos_emb_pred))

        trg_flat = label_seq_trg.flatten()
        non_pad_mask = (trg_flat != self.pad_token)
        num_elem = non_pad_mask.sum()

        labels_all = torch.masked_select(trg_flat, non_pad_mask)

        # Step4. Compute logit and label for neg+pos samples

        # remove zero padding elements 

        pos_emb_pred = self.remove_pad_3d(pos_emb_pred, non_pad_mask)
        # if self.loss_type != 'cross_entropy':
        #    pos_emb_trg = self.remove_pad_3d(pos_emb_trg, non_pad_mask)
        #    neg_emb_inp = self.remove_pad_4d(neg_emb_inp, non_pad_mask)

        if not self.mlm:
            ##Temporary. Replace by the commented code after experiments with past information
            if self.session_aware:
                non_pad_original_mask = (label_seq_trg_original.flatten() != self.pad_token)
                for feat_name in metadata_for_pred_logging:                
                    metadata_for_pred_logging[feat_name] = torch.masked_select(metadata_for_pred_logging[feat_name].flatten(), 
                                                                            non_pad_original_mask)
            else:
                #Keeping removing zero-padded items metadata features for the next-clicks (targets), so that they are aligned
                for feat_name in metadata_for_pred_logging:                
                    metadata_for_pred_logging[feat_name] = torch.masked_select(metadata_for_pred_logging[feat_name].flatten(), 
                                                                            non_pad_mask)

        if self.mf_constrained_embeddings:
            if self.mf_constrained_embeddings_disable_bias:
                logits_all = F.linear(pos_emb_pred, weight = self.embedding_tables[self.label_embedding_table_name].weight)
            else:
                logits_all = F.linear(pos_emb_pred, weight = self.embedding_tables[self.label_embedding_table_name].weight,
                                    bias = self.output_layer_bias)
        else:
            logits_all = self.output_layer(pos_emb_pred)

        if not self.negative_sampling:
            predictions_all = self.log_softmax(logits_all)
            loss_ce = self.loss_nll(predictions_all, labels_all)
            loss = loss_ce
            # accuracy
            _, max_idx = torch.max(logits_all, dim = 1)
            train_acc = (max_idx == labels_all).mean(dtype = torch.float32)

        loss_neg = None
        predictions_neg = None
        labels_neg = None
        train_acc_neg = None

        if self.negative_sampling:
            # Compute pairwise loss using negative samples
            # The negative samples are the targets present in the other sessions of same the mini-batch
            # ==> (items with the same session are not considered as negatives)
            bs = label_seq_trg.shape[0]
            # build negative mask for each session (bs, #negatives):
            if self.mlm:
                negatives = torch.masked_select(label_seq_trg, label_mlm_mask)
                negative_mask = self.compute_neg_mask(label_mlm_mask)
            else:
                negatives = torch.masked_select(label_seq_trg, mask_trg_pad)
                negative_mask = self.compute_neg_mask(mask_trg_pad)
            # If adding extra negative samples: neg_sampling_extra_samples_per_batch > 0
            if self.neg_sampling_extra_samples_per_batch:
                if self.neg_sampling_store_size != 0:
                    if self.neg_sampling_store_pointer == self.neg_sampling_store_rows:
                        # if all examples in the cache were used: re-sample a new cache
                        self.neg_samples = self.generate_neg_samples(length = self.neg_sampling_store_rows)
                        self.neg_sampling_store_pointer = 0
                    # Get a vector of neg_sampling_extra_samples_per_batch for the current batch
                    sample = self.neg_samples[self.neg_sampling_store_pointer].to(self.device)
                    self.neg_sampling_store_pointer += 1
                else:
                    # Sample for each batch without using a cache
                    sample = self.generate_neg_samples(length = 1).to(self.device)

                # Concat the additional samples to mini-batch negatives
                negatives = torch.cat([negatives, sample], dim = 0)
                # add ones to negative_mask for additional negatives
                negative_mask = torch.cat([negative_mask, torch.ones((bs, len(sample)), device = self.device).bool()],
                                          dim = 1)
            positives = ~negative_mask
            # flat negative mask : of shape  N_pos_targets x N_negatives
            negative_mask_all = torch.repeat_interleave(negative_mask, positives.sum(1), dim = 0)
            # Get logit scores
            logit_sample = logits_all[:, negatives]
            # Compute loss:
            loss = self.loss_fn(logit_sample, negative_mask_all)

            '''
            pos_emb_pred_expanded = pos_emb_pred.unsqueeze(
                1).expand_as(neg_emb_inp)
            pred_emb_flat = torch.cat((pos_emb_pred.unsqueeze(
                1), pos_emb_pred_expanded), dim=1).flatten(end_dim=1)
            trg_emb_flat = torch.cat((pos_emb_trg.unsqueeze(
                1), neg_emb_inp), dim=1).flatten(end_dim=1)

            n_neg_items = neg_emb_inp.size(1)
            n_pos_ex = pos_emb_trg.size(0)
            n_neg_ex = neg_emb_inp.size(0) * n_neg_items
            labels_neg = torch.LongTensor(
                [n_neg_items] * n_pos_ex).to(self.device)

            # compute similarity
            if self.similarity_type == 'concat_mlp':
                pos_cos_score = self.sim_mlp(
                    torch.cat((pos_emb_pred, pos_emb_trg), dim=1))
                neg_cos_score = self.sim_mlp(
                    torch.cat((pos_emb_pred_expanded, neg_emb_inp), dim=2)).squeeze(2)
            elif self.similarity_type == 'cosine':
                pos_cos_score = torch.cosine_similarity(
                    pos_emb_pred, pos_emb_trg).unsqueeze(1)
                neg_cos_score = torch.cosine_similarity(
                    pos_emb_pred_expanded, neg_emb_inp, dim=2)
            elif self.similarity_type == 'multi_mlp':
                pos_cos_score = self.sim_mlp(pos_emb_pred * pos_emb_trg)
                neg_cos_score = self.sim_mlp(
                    pos_emb_pred_expanded * neg_emb_inp).squeeze(2)

            # compute predictionss (logits)
            cos_sim_concat = torch.cat((neg_cos_score, pos_cos_score), dim=1)
            items_prob_log = F.log_softmax(cos_sim_concat, dim=1)
            predictions_neg = torch.exp(items_prob_log)

            # Step5. Compute loss and accuracy
            loss_neg = torch.tensor(
                0.0, requires_grad=False, device=self.device)

            if self.loss_type in ['cross_entropy_neg', 'cross_entropy_neg_1d']:

                loss_neg = self.loss_fn(items_prob_log, labels_neg)

            elif self.loss_type.startswith('margin_hinge'):

                # _label = torch.LongTensor([1] * n_pos_ex + [-1] * n_neg_ex).to(pred_emb_flat.device)

                # loss = self.loss_fn(pred_emb_flat, trg_emb_flat, _label) / num_elem
                pos_dist, neg_dist = pos_cos_score, neg_cos_score

                if self.loss_type == 'margin_hinge_a':
                    # case A
                    loss_neg = (pos_dist.sum(
                    ) + torch.relu(self.margin_loss - neg_dist).sum()) / (n_pos_ex + n_neg_ex)
                elif self.loss_type == 'margin_hinge_b':
                    # case B (case of the paper)
                    n_neg_samples = neg_emb_inp.size(1)
                    loss_neg = (pos_dist.sum() * n_neg_samples + torch.relu(
                        self.margin_loss - neg_dist).sum()) / (n_pos_ex + n_neg_ex)

            else:
                raise NotImplementedError

            #Multi-task learning (XE over all items + XE over negative samples)
            loss_neg *= self.neg_rescale_factor
            loss_ce *= self.all_rescale_factor
            loss = loss_neg + loss_ce
            

            # accuracy
            _, max_idx = torch.max(cos_sim_concat, dim=1)
            train_acc_neg = (max_idx == n_neg_items).sum(
                dtype=torch.float32) / num_elem
            '''

        #Scaling the loss
        loss = loss * self.loss_scale_factor

        outputs = {'loss': loss,
                   'labels': labels_all,
                   'predictions': logits_all,                   
                   'pred_metadata': metadata_for_pred_logging,
                   'model_outputs': model_outputs # Keep mems, hidden states, attentions if there are in it
                   #'metadata_pred': metadata_for_pred_logging
                   }
        #outputs = (train_acc, train_acc_neg, loss, loss_neg, loss_ce, predictions_neg, labels_neg, predictions_all,
        #           labels_all, metadata_for_pred_logging) + model_outputs  # Keep mems, hidden states, attentions if there are in it

        return outputs  # return (train_acc), (loss), (predictions), (labels), (mems), (hidden states), (attentions)

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

        #labels = itemid_seq.clone()
        labels = torch.full(itemid_seq.shape, self.pad_token, dtype=itemid_seq.dtype, device=self.device)
        non_padded_mask = (itemid_seq != self.pad_token)

        rows_ids = torch.arange(itemid_seq.size(0), dtype=torch.long, device=self.device)

        # During training, masks labels to be predicted according to a probability, ensuring that each session has at least one label to predict
        if self.training:
            # Selects a percentage of items to be masked (selected as labels)
            probability_matrix = torch.full(itemid_seq.shape, mlm_probability, device=self.device)            
            masked_labels = torch.bernoulli(probability_matrix).bool() & non_padded_mask
            labels = torch.where(masked_labels, itemid_seq, 
                                torch.tensor(self.pad_token, device=self.device))

            # Set at least one item in the sequence to mask, so that the network can learn something with this session
            one_random_index_by_session = torch.multinomial(
                non_padded_mask.float(), num_samples=1).squeeze()         
            labels[rows_ids, one_random_index_by_session] = itemid_seq[rows_ids, one_random_index_by_session]
            masked_labels = (labels != self.pad_token)

            # If a sequence has only masked labels, unmasks one of the labels
            sequences_with_only_labels = masked_labels.sum(axis=1) == non_padded_mask.sum(axis=1)
            sampled_labels_to_unmask = torch.multinomial(masked_labels.float(), num_samples=1).squeeze() 

            labels_to_unmask = torch.masked_select(sampled_labels_to_unmask, sequences_with_only_labels)
            rows_to_unmask = torch.masked_select(rows_ids, sequences_with_only_labels)
            
            labels[rows_to_unmask, labels_to_unmask] = self.pad_token
            masked_labels = (labels != self.pad_token)

            # Logging the real percentage of masked items (labels)
            #perc_masked_labels = masked_labels.sum() / non_padded_mask.sum().float()
            #logger.info(f"  % Masked items as labels: {perc_masked_labels}")

        
        # During evaluation always masks the last item of the session
        else:
            last_item_sessions = non_padded_mask.sum(axis=1) - 1
            labels[rows_ids, last_item_sessions] = itemid_seq[rows_ids, last_item_sessions]
            masked_labels = (labels != self.pad_token)


        #TODO: Based on HF Data Collator (https://github.com/huggingface/transformers/blob/bef0175168002e40588da53c45a0760744731e76/src/transformers/data/data_collator.py#L164). Experiment later how much replacing masked input by a random item or keeping the masked input token unchanged helps, like
        
        ## 80% of the time, we replace masked input tokens with tokenizer.mask_token
        #indices_replaced = torch.bernoulli(torch.full(
        #    labels.shape, 0.8, device=self.device)).bool() & masked_indices
        ##itemid_seq[indices_replaced] = self.mask_token 
        
        ## 10% of the time, we replace masked input tokens with random word
        #indices_random = torch.bernoulli(torch.full(
        #    labels.shape, 0.5, device=self.device)).bool() & masked_indices & ~indices_replaced
        ##random_words = torch.randint(
        ##    self.target_dim, labels.shape, dtype=torch.long, device=self.device)
        ##itemid_seq[indices_random] = random_words[indices_random]

        #input_seq[indices_random] = torch.zeros(input_seq.size(2), device=self.device)

        #itemid_seq[masked_labels] = self.mask_token

        return labels, masked_labels

    def _unflatten_neg_seq(self, neg_seq, seqlen):
        """
        neg_seq: n_batch x (num_neg_samples x max_seq_len); flattened. 2D.
        """
        assert neg_seq.dim() == 2

        n_batch, flatten_len = neg_seq.size()
        
        assert flatten_len % seqlen == 0

        n_neg_seqs_per_pos_seq = flatten_len // seqlen
        return neg_seq.reshape((n_batch, seqlen, n_neg_seqs_per_pos_seq))

    def feature_process(self, inputs, max_seq_len=None, is_neg=False):
        
        if is_neg:
            assert max_seq_len is not None, "for negative samples, max_seq_len should be provided"
        
        label_seq, output = None, []
        metadata_for_pred_logging = {}

        transformed_features = OrderedDict()
        for cname, cinfo in self.feature_map.items():

            # represent (not is_neg and self.col_prefix_neg not in cname) or (is_neg and self.col_prefix_neg in cname):

            if not (bool(is_neg) ^ bool(self.col_prefix_neg in cname)): 

                cdata = inputs[cname]
                if is_neg:
                    cdata = self._unflatten_neg_seq(cdata, max_seq_len)

                #Temporary change to see if with sequences multiples of 8 we can get better acceleration with --fp16, by using TensorCores
                #cdata = cdata[:,:10]

                if 'is_label' in cinfo and cinfo['is_label']:
                    label_seq = cdata

                if cinfo['dtype'] == 'categorical':
                    if 'is_label' in cinfo and cinfo['is_label']:
                        if self.use_ohe_item_ids_inputs:          
                            cdata = torch.nn.functional.one_hot(cdata.long(), num_classes=self.target_dim).float()
                        elif self.constrained_embeddings: # use output layer for the embedding matrix
                            cdata = self.output_layer.weight[cdata.long(), :]
                        else:
                            cdata = self.embedding_tables[cinfo['emb_table']](cdata.long())                    
                    else:
                        cdata = self.embedding_tables[cinfo['emb_table']](cdata.long())                    

                    if max_seq_len is None:
                        max_seq_len = cdata.size(1)

                elif cinfo['dtype'] == 'long':
                    cdata = cdata.unsqueeze(-1).long()
                elif cinfo['dtype'] == 'float':
                    cdata = cdata.unsqueeze(-1).float()
                elif cinfo['is_control']:
                    #Control features are not used as input for the model
                    continue
                else:
                    raise NotImplementedError

                if not is_neg:
                    # Keeping item metadata features that will
                    if 'log_with_preds_as_metadata' in cinfo and cinfo['log_with_preds_as_metadata'] == True:
                        metadata_for_pred_logging[cname] = inputs[cname].detach()

                        #Temporary change to see if with sequences multiples of 8 we can get better acceleration with --fp16, by using TensorCores
                        #metadata_for_pred_logging[cname] = inputs[cname][:,:17].detach()

                transformed_features[cname] = cdata
                #output.append(cdata)

        if self.session_aware:
            # Concatenates past sessions before the session with the current session, for each feature
            # assuming that features from past sessions have a common prefix
            features_to_delete = []            
            for fname in transformed_features.keys():
                if not fname.startswith(self.session_aware_features_prefix):
                    past_fname = self.session_aware_features_prefix + fname
                    if past_fname in transformed_features:
                        transformed_features[fname] = \
                            torch.cat([transformed_features[past_fname],
                                       transformed_features[fname]], axis = 1)
                    features_to_delete.append(past_fname)
            for past_fname in features_to_delete:
                del transformed_features[past_fname]

        if len(transformed_features) > 1:
            output = torch.cat(list(transformed_features.values()), dim = -1)
        else:
            output = list(transformed_features.values())[0]

        return output, label_seq, max_seq_len, metadata_for_pred_logging

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor))
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))
        return out_tensor

    def remove_pad_4d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x n_negex x emb_dim)
        inp_tensor_fl = inp_tensor.reshape(-1, inp_tensor.size(2), inp_tensor.size(3))
        inp_tensor_fl = torch.masked_select(inp_tensor_fl, non_pad_mask.unsqueeze(1).unsqueeze(2).expand_as(inp_tensor_fl))
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(2), inp_tensor.size(3))
        return out_tensor


    def set_items_freq_for_sampling(self, items_sorted_freq_series):
        self.items_ids_sorted_by_freq = torch.tensor(items_sorted_freq_series.index.values, device=self.device)
        self.items_freq_sorted = torch.tensor(items_sorted_freq_series.values, device=self.device)

        # If should adding extra negative samples to the batch ones         
        if (self.neg_sampling_store_size != 0) and self.neg_sampling_extra_samples_per_batch > 0:
            # Generate a cumulative distribution of frequency (sorted in ascending order), so that more popular items can be sampled more often
            self.items_freq_sorted_norm = self.items_freq_sorted ** self.neg_sampling_alpha
            self.items_freq_sorted_norm = self.items_freq_sorted_norm.cumsum(dim = 0) / self.items_freq_sorted_norm.sum(dim=0)
            self.items_freq_sorted_norm[-1] = 1

            # Defines a cache that pre-stores N="neg_sampling_store_size" negative samples
            self.neg_sampling_store_rows = self.neg_sampling_store_size // self.neg_sampling_extra_samples_per_batch
            if self.neg_sampling_store_rows <= 1:
                self.neg_sampling_store_rows = 0
                print('No negative samples store was used.')
            else:
                self.neg_samples = self.generate_neg_samples(length = self.neg_sampling_store_rows)
                self.neg_sampling_store_pointer = 0
                print('Created sample store with {} batches of samples (type=CPU)'.format(self.neg_sampling_store_rows))
        else:
            print('No example store was used')

    def generate_neg_samples(self, length):
        """
        Args:
            length: the number of vectors of shape self.neg_sampling_extra_samples_per_batch to store in cache memory
        return:
            sample: Tensor of negative samples of shape length x self.neg_sampling_extra_samples_per_batch
        """

        if self.neg_sampling_alpha:
            samples_idx = torch.searchsorted(self.items_freq_sorted_norm, torch.rand(self.neg_sampling_extra_samples_per_batch * length, device=self.device))
            #Retrieves the correct item ids from the sampled indices over the cumulative prob distribution
            sampled_item_ids = self.items_ids_sorted_by_freq[samples_idx]
        else:
            n_items = self.items_freq_sorted_norm.shape[0]
            sampled_item_ids = torch.randint(0, n_items, size = (self.neg_sampling_extra_samples_per_batch * length,), device=self.device)
        if length > 1:
            sampled_item_ids = sampled_item_ids.reshape((length, self.neg_sampling_extra_samples_per_batch))
        return sampled_item_ids

    def compute_neg_mask(self, positive_mask):
        """
        Args:
            positive_mask: Tensor of shape bs x seq_len: mask  input where target is on padding token
        Return:
            negative_mask: Tensor of shape #pos_targets x negatives to specify the negative items
                            for each positive target
        """
        # TODO: Refactor the code to not use  For loop
        bs, _ = positive_mask.shape
        N_neg = positive_mask.flatten().sum()
        pos_target_per_session = positive_mask.sum(1)
        pos_target_per_session = torch.cat([torch.tensor([0], device=self.device), pos_target_per_session])
        cumul_pos_target = pos_target_per_session.cumsum(dim = 0)
        # define mask over all mini-batch negatives
        mask = torch.zeros(bs, N_neg, device=self.device)
        for i in range(bs):
            mask[i, cumul_pos_target[i]:cumul_pos_target[i + 1]] = 1
        return ~mask.bool()


def nll_1d(items_prob, _label=None):
    # https://github.com/gabrielspmoreira/chameleon_recsys/blob/da7f73a2b31d6867d444eded084044304b437413/nar_module/nar/nar_model.py#L639
    items_prob = torch.exp(items_prob)
    positive_prob = items_prob[:, 0]
    xe_loss = torch.log(positive_prob)
    cosine_sim_loss = - torch.mean(xe_loss)
    return cosine_sim_loss
