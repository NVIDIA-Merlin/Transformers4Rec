"""
A meta class supports various (Huggingface) transformer models for RecSys tasks.

"""

import logging
from collections import OrderedDict

import math
import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_utils import PreTrainedModel

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
        self.embedding_tables = nn.ModuleDict()
        
        self.col_prefix_neg = data_args.feature_prefix_neg_sample
        concat_input_dim = 0
        target_dim = -1
        
        # set embedding tables
        for cname, cinfo in self.feature_map.items():
            if self.col_prefix_neg not in cname:
                if cinfo['dtype'] == 'categorical':
                    embedding_size = get_embedding_size_from_cardinality(cinfo['cardinality'])
                    self.embedding_tables[cinfo['emb_table']] = nn.Embedding(
                        cinfo['cardinality'], 
                        embedding_size, 
                        padding_idx=self.pad_token
                    )
                    concat_input_dim += embedding_size
                elif cinfo['dtype'] in ['long', 'float']:
                    concat_input_dim += 1   
                else:
                    raise NotImplementedError
                if 'is_label' in cinfo and cinfo['is_label']:
                    target_dim = cinfo['cardinality']
        
        if target_dim == -1:
            raise RuntimeError('label column is not declared in feature map.')
        
        self.inp_merge = model_args.inp_merge
        if self.inp_merge == 'mlp':
            self.mlp_merge = nn.Linear(concat_input_dim, model_args.d_model)
        elif self.inp_merge == 'attn':
            self.attn_merge = AttnMerge(concat_input_dim, model_args.d_model)
        else:
            raise NotImplementedError

        
        self.similarity_type = model_args.similarity_type
        self.margin_loss = model_args.margin_loss
        self.output_layer = nn.Linear(model_args.d_model, target_dim)
        self.loss_type = model_args.loss_type
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        self.loss_nll = nn.NLLLoss(ignore_index=self.pad_token)
        
        if self.loss_type == 'cross_entropy_neg':
            self.loss_fn = nn.NLLLoss()
        elif self.loss_type == 'cross_entropy_neg_1d':
            self.loss_fn = nll_1d
        elif self.loss_type.startswith('margin_hinge'):
            # https://pytorch.org/docs/master/generated/torch.nn.CosineEmbeddingLoss.html
            self.loss_fn = nn.CosineEmbeddingLoss(margin=model_args.margin_loss, reduction='sum')
        elif self.loss_type != 'cross_entropy':
            raise NotImplementedError

        if model_args.model_type == 'reformer':
            tf_out_size = model_args.d_model * 2
        else:
            tf_out_size = model_args.d_model

        self.transformer_output_project = nn.Linear(tf_out_size, model_args.d_model)

        if self.similarity_type in ['concat_mlp', 'multi_mlp']:
            m_factor = 2 if self.similarity_type == 'concat_mlp' else 1
            self.sim_mlp = nn.Sequential(
                OrderedDict([
                    ('linear0', nn.Linear(model_args.d_model * m_factor , model_args.d_model)),
                    ('relu0', nn.LeakyReLU()),
                    ('linear1', nn.Linear(model_args.d_model, model_args.d_model // 2)),
                    ('relu1', nn.LeakyReLU()),
                    ('linear2', nn.Linear(model_args.d_model // 2, model_args.d_model // 4)),
                    ('relu2', nn.LeakyReLU()),
                    ('linear3', nn.Linear(model_args.d_model // 4, 1)),
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


    def forward(self, inputs):
        
        # Step1. Unpack inputs, get embedding, and concatenate them
        label_seq = None
        
        max_seq_len = None

        pos_inp, label_seq, max_seq_len, metadata_for_pred_logging = self.feature_process(inputs)
        if self.loss_type != 'cross_entropy':
            neg_inp, _, _, _ = self.feature_process(inputs, max_seq_len, is_neg=True)

        if label_seq is not None:
            label_seq_inp = label_seq[:, :-1] 
            label_seq_trg = label_seq[:, 1:] 
        else:
            raise RuntimeError('label sequence is not declared in feature_map')

        # apply mask on input where target is on padding token
        mask_trg_pad = (label_seq_trg != self.pad_token)
        
        label_seq_inp = label_seq_inp * mask_trg_pad


        # Creating an additional feature with the position in the sequence
        metadata_for_pred_logging['seq_pos'] = torch.arange(1, label_seq.shape[1]+1, device=self.device).repeat(label_seq.shape[0], 1)
        for feat_name in metadata_for_pred_logging:
            #Keeping only metadata features for the next-clicks (targets)
            metadata_for_pred_logging[feat_name] = metadata_for_pred_logging[feat_name][:, 1:]


        # Step 2. Merge features
        
        if self.inp_merge == 'mlp':
            pos_emb = self.tf_out_act(self.mlp_merge(pos_inp))            
            if self.loss_type != 'cross_entropy':
                neg_emb = torch.tanh(self.mlp_merge(neg_inp))
        elif self.inp_merge == 'attn':
            pos_emb = self.attn_merge(pos_inp)
            if self.loss_type != 'cross_entropy':
                neg_emb = self.attn_merge(neg_inp)

        # slice over time-steps for input and target and ensuring masking is applied
        pos_emb_inp = pos_emb[:, :-1] * mask_trg_pad.unsqueeze(-1)
        if self.loss_type != 'cross_entropy':
            pos_emb_trg = pos_emb[:, 1:] * mask_trg_pad.unsqueeze(-1)
            neg_emb_inp = neg_emb[:, :-1] * mask_trg_pad.unsqueeze(-1).unsqueeze(-1)

        # Step3. Run forward pass on model architecture

        if not isinstance(self.model, PreTrainedModel): #Checks if its a transformer
            # compute output through RNNs
            results = self.model(
                input=pos_emb_inp
            )

            if type(results) is tuple or type(results) is list:
                pos_emb_pred = results[0]
            else:
                pos_emb_pred = results

            model_outputs = (None, )
            
        else:
            """
            Transformer Models
            """
            if self.disable_positional_embeddings:
                position_ids = torch.zeros(max_seq_len-1, requires_grad=False, dtype=torch.long, device=self.device)
            else:
                position_ids = None

            model_outputs = self.model(
                inputs_embeds=pos_emb_inp,
                position_ids=position_ids
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
        if self.loss_type != 'cross_entropy':
            pos_emb_trg = self.remove_pad_3d(pos_emb_trg, non_pad_mask)
            neg_emb_inp = self.remove_pad_4d(neg_emb_inp, non_pad_mask)

        #Keeping removing zero-padded items metadata features for the next-clicks (targets), so that they are aligned
        for feat_name in metadata_for_pred_logging:
            metadata_for_pred_logging[feat_name] = torch.masked_select(metadata_for_pred_logging[feat_name].flatten(), non_pad_mask)

        logits_all = self.output_layer(pos_emb_pred)
        predictions_all = self.log_softmax(logits_all)
        loss_ce = self.loss_nll(predictions_all, labels_all) 
        loss = loss_ce

        # accuracy
        _, max_idx = torch.max(logits_all, dim=1)
        train_acc = (max_idx == labels_all).mean(dtype=torch.float32)


        loss_neg = None
        predictions_neg = None
        labels_neg = None
        train_acc_neg = None
        # concatenate 
        if self.loss_type != 'cross_entropy':
            pos_emb_pred_expanded = pos_emb_pred.unsqueeze(1).expand_as(neg_emb_inp)
            pred_emb_flat = torch.cat((pos_emb_pred.unsqueeze(1), pos_emb_pred_expanded), dim=1).flatten(end_dim=1)
            trg_emb_flat = torch.cat((pos_emb_trg.unsqueeze(1), neg_emb_inp), dim=1).flatten(end_dim=1)

            n_neg_items = neg_emb_inp.size(1)
            n_pos_ex = pos_emb_trg.size(0)
            n_neg_ex = neg_emb_inp.size(0) * n_neg_items
            labels_neg = torch.LongTensor([n_neg_items] * n_pos_ex).to(self.device)

            # compute similarity
            if self.similarity_type == 'concat_mlp':
                pos_cos_score = self.sim_mlp(torch.cat((pos_emb_pred, pos_emb_trg), dim=1))
                neg_cos_score = self.sim_mlp(torch.cat((pos_emb_pred_expanded, neg_emb_inp), dim=2)).squeeze(2)
            elif self.similarity_type == 'cosine':
                pos_cos_score = torch.cosine_similarity(pos_emb_pred, pos_emb_trg).unsqueeze(1)
                neg_cos_score = torch.cosine_similarity(pos_emb_pred_expanded, neg_emb_inp, dim=2)
            elif self.similarity_type == 'multi_mlp':
                pos_cos_score = self.sim_mlp(pos_emb_pred * pos_emb_trg)
                neg_cos_score = self.sim_mlp(pos_emb_pred_expanded * neg_emb_inp).squeeze(2)

            # compute predictionss (logits)
            cos_sim_concat = torch.cat((neg_cos_score, pos_cos_score), dim=1)
            items_prob_log = F.log_softmax(cos_sim_concat, dim=1)
            predictions_neg = torch.exp(items_prob_log)

            # Step5. Compute loss and accuracy
            loss_neg = torch.tensor(0.0, requires_grad=False, device=self.device)

            if self.loss_type in ['cross_entropy_neg', 'cross_entropy_neg_1d']:

                loss_neg = self.loss_fn(items_prob_log, labels_neg)

            elif self.loss_type.startswith('margin_hinge'):

                # _label = torch.LongTensor([1] * n_pos_ex + [-1] * n_neg_ex).to(pred_emb_flat.device)

                # loss = self.loss_fn(pred_emb_flat, trg_emb_flat, _label) / num_elem 
                pos_dist, neg_dist = pos_cos_score, neg_cos_score
                
                if self.loss_type == 'margin_hinge_a':
                    # case A
                    loss_neg = (pos_dist.sum() + torch.relu(self.margin_loss - neg_dist).sum()) / (n_pos_ex + n_neg_ex)
                elif self.loss_type == 'margin_hinge_b':
                    # case B (case of the paper)
                    n_neg_samples = neg_emb_inp.size(1)
                    loss_neg = (pos_dist.sum() * n_neg_samples + torch.relu(self.margin_loss - neg_dist).sum()) / (n_pos_ex + n_neg_ex)

            else:
                raise NotImplementedError

            #Multi-task learning (XE over all items + XE over negative samples)
            loss_neg *= self.neg_rescale_factor            
            loss_ce *= self.all_rescale_factor
            loss = loss_neg + loss_ce

            # accuracy
            _, max_idx = torch.max(cos_sim_concat, dim=1)
            train_acc_neg = (max_idx == n_neg_items).sum(dtype=torch.float32) / num_elem

        outputs = (train_acc, train_acc_neg, loss, loss_neg, loss_ce, predictions_neg, labels_neg, predictions_all, labels_all, metadata_for_pred_logging) + model_outputs  # Keep mems, hidden states, attentions if there are in it

        return outputs  # return (train_acc), (loss), (predictions), (labels), (mems), (hidden states), (attentions)

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
        for cname, cinfo in self.feature_map.items():

            # represent (not is_neg and self.col_prefix_neg not in cname) or (is_neg and self.col_prefix_neg in cname):

            if not (bool(is_neg) ^ bool(self.col_prefix_neg in cname)): 

                cdata = inputs[cname]
                if is_neg:
                    cdata = self._unflatten_neg_seq(cdata, max_seq_len)

                if 'is_label' in cinfo and cinfo['is_label']:
                    label_seq = cdata

                if cinfo['dtype'] == 'categorical':
                    cdata = self.embedding_tables[cinfo['emb_table']](cdata.long())
                    if max_seq_len is None:
                        max_seq_len = cdata.size(1)

                elif cinfo['dtype'] == 'long':
                    cdata = cdata.unsqueeze(-1).long()
                elif cinfo['dtype'] == 'float':
                    cdata = cdata.unsqueeze(-1).float()
                else:
                    raise NotImplementedError

                if not is_neg:
                    # Keeping item metadata features that will
                    if 'log_with_preds_as_metadata' in cinfo and cinfo['log_with_preds_as_metadata'] == True:
                        metadata_for_pred_logging[cname] = inputs[cname]

                output.append(cdata)

        output = torch.cat(output, dim=-1)

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


def nll_1d(items_prob, _label=None):
    # https://github.com/gabrielspmoreira/chameleon_recsys/blob/da7f73a2b31d6867d444eded084044304b437413/nar_module/nar/nar_model.py#L639
    items_prob = torch.exp(items_prob)
    positive_prob = items_prob[:, 0]
    xe_loss = torch.log(positive_prob)
    cosine_sim_loss = - torch.mean(xe_loss)
    return cosine_sim_loss