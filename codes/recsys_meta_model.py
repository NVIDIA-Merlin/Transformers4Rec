"""
A meta class supports various (Huggingface) transformer models for RecSys tasks.

"""

import logging
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

torch.manual_seed(0)

class RecSysMetaModel(PreTrainedModel):
    """
    vocab_sizes : sizes of vocab for each discrete inputs
        e.g., [product_id_vocabs, category_vocabs, etc.]
    """
    def __init__(self, model, config, model_args, data_args, feature_map):
        super(RecSysMetaModel, self).__init__(config)
        
        self.model = model 

        if self.model.__class__ in [nn.GRU, nn.LSTM, nn.RNN]:
            self.is_rnn = True
        else:
            self.is_rnn = False

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
        
        concat_input_dim = 0
        target_dim = -1
        
        # set embedding tables
        for cname, cinfo in self.feature_map.items():
            if '_neg_' not in cname:
                if cinfo['dtype'] == 'categorical':
                    self.embedding_tables[cinfo['emb_table']] = nn.Embedding(
                        cinfo['cardinality'], 
                        model_args.d_model, 
                        padding_idx=self.pad_token
                    )
                    concat_input_dim += model_args.d_model
                elif cinfo['dtype'] in ['long', 'float']:
                    concat_input_dim += 1   
                else:
                    raise NotImplementedError
                if 'is_label' in cinfo and cinfo['is_label']:
                    target_dim = cinfo['cardinality']
        
        if target_dim == -1:
            raise RuntimeError('label column is not declared in feature map.')

        self.mlp_merge = nn.Linear(concat_input_dim, model_args.d_model)
        
        self.similarity_type = model_args.similarity_type
        self.margin_loss = model_args.margin_loss
        self.output_layer = nn.Linear(model_args.d_model, target_dim)
        self.loss_type = model_args.loss_type
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        if self.loss_type == 'cross_entropy':
            self.loss_fn = nn.NLLLoss(ignore_index=self.pad_token)
        elif self.loss_type == 'cross_entropy_neg':
            self.loss_fn = nn.NLLLoss()
        elif self.loss_type == 'cross_entropy_neg_1d':
            self.loss_fn = nll_1d
        elif self.loss_type.startswith('margin_hinge'):
            # https://pytorch.org/docs/master/generated/torch.nn.CosineEmbeddingLoss.html
            self.loss_fn = nn.CosineEmbeddingLoss(margin=model_args.margin_loss, reduction='sum')
        else:
            raise NotImplementedError

        if self.similarity_type == 'concat_mlp':
            self.sim_mlp = nn.Sequential(
                OrderedDict([
                    ('linear0', nn.Linear(model_args.d_model * 2, model_args.d_model)),
                    ('relu0', nn.LeakyReLU()),
                    ('linear1', nn.Linear(model_args.d_model, model_args.d_model // 2)),
                    ('relu1', nn.LeakyReLU()),
                    ('linear2', nn.Linear(model_args.d_model // 2, model_args.d_model // 4)),
                    ('relu2', nn.LeakyReLU()),
                    ('linear3', nn.Linear(model_args.d_model // 4, 1)),
                    ('sigmoid', nn.Sigmoid()),
                ]       
            ))
        
        self.cnts = []

    def _unflatten_neg_seq(self, neg_seq, seqlen):
        """
        neg_seq: n_batch x (num_neg_samples x max_seq_len); flattened. 2D.
        """
        assert neg_seq.dim() == 2

        n_batch, flatten_len = neg_seq.size()
        
        assert flatten_len % seqlen == 0

        n_neg_seqs_per_pos_seq = flatten_len // seqlen
        return neg_seq.reshape((n_batch, seqlen, n_neg_seqs_per_pos_seq))

    def forward(self, inputs):
        
        # Step1. Unpack inputs, get embedding, and concatenate them
        pos_inp, neg_inp, label_seq = [], [], None
        
        max_seq_len = None

        # NOTE: we have separate code for positive and negative samples
        #       as we need to know sequence length first from postiive samples
        #       and then use the sequence length to process negative samples

        # TODO: refactorize using a function. 
        for cname, cinfo in self.feature_map.items():
            # Positive Samples
            if '_neg_' not in cname:
                cdata = inputs[cname]
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

                pos_inp.append(cdata)

        for cname, cinfo in self.feature_map.items():
            # Negative Samples
            if '_neg_' in cname:
                cdata = inputs[cname]
                cdata = self._unflatten_neg_seq(cdata, max_seq_len)
                if cinfo['dtype'] == 'categorical':
                    neg_pids = cdata
                    cdata = self.embedding_tables[cinfo['emb_table']](cdata)
                elif cinfo['dtype'] == 'long':
                    cdata = cdata.unsqueeze(-1).long()
                elif cinfo['dtype'] == 'float':
                    cdata = cdata.unsqueeze(-1).float()
                else:
                    raise NotImplementedError
                neg_inp.append(cdata)

        pos_inp = torch.cat(pos_inp, dim=-1)
        neg_inp = torch.cat(neg_inp, dim=-1)

        if label_seq is not None:
            label_seq_inp = label_seq[:, :-1] 
            label_seq_trg = label_seq[:, 1:] 
        else:
            raise RuntimeError('label sequence is not declared in feature_map')

        # apply mask on input where target is on padding token
        mask_trg_pad = (label_seq_trg != self.pad_token)
        label_seq_inp = label_seq_inp * mask_trg_pad

        # Step 2. Merge features
        
        pos_emb = torch.tanh(self.mlp_merge(pos_inp))
        neg_emb = torch.tanh(self.mlp_merge(neg_inp))

        # slice over time-steps for input and target 
        pos_emb_inp = pos_emb[:, :-1]
        pos_emb_trg = pos_emb[:, 1:]
        neg_emb_inp = neg_emb[:, :-1]

        # Step3. Run forward pass on model architecture

        if self.is_rnn:
            # compute output through RNNs
            pos_emb_pred, _ = self.model(
                input=pos_emb_inp
            )
            model_outputs = (None, )
            
        else:
            """
            Transformer Models
            """
            model_outputs = self.model(
                inputs_embeds=pos_emb_inp,
            )
            pos_emb_pred = model_outputs[0]
            model_outputs = tuple(model_outputs[1:])

        trg_flat = label_seq_trg.flatten()
        non_pad_mask = (trg_flat != self.pad_token)        
        num_elem = non_pad_mask.sum()

        trg_flat_nonpad = torch.masked_select(trg_flat, non_pad_mask)

        # Step4. Compute logit and label for neg+pos samples

        # remove zero padding elements 
        pos_emb_pred = pos_emb_pred.flatten(end_dim=1)
        pos_emb_pred_fl = torch.masked_select(pos_emb_pred, non_pad_mask.unsqueeze(1).expand_as(pos_emb_pred))
        pos_emb_pred = pos_emb_pred_fl.view(-1, pos_emb_pred.size(1))

        pos_emb_trg = pos_emb_trg.flatten(end_dim=1)
        pos_emb_trg_fl = torch.masked_select(pos_emb_trg, non_pad_mask.unsqueeze(1).expand_as(pos_emb_trg))
        pos_emb_trg = pos_emb_trg_fl.view(-1, pos_emb_trg.size(1))

        # neg_emb_inp:  (n_batch x seqlen x n_negex x emb_dim)
        neg_emb_inp_fl = neg_emb_inp.reshape(-1, neg_emb_inp.size(2), neg_emb_inp.size(3))
        neg_emb_inp_fl = torch.masked_select(neg_emb_inp_fl, non_pad_mask.unsqueeze(1).unsqueeze(2).expand_as(neg_emb_inp_fl))
        neg_emb_inp = neg_emb_inp_fl.view(-1, neg_emb_inp.size(2), neg_emb_inp.size(3))

        # concatenate 
        pos_emb_pred_expanded = pos_emb_pred.unsqueeze(1).expand_as(neg_emb_inp)
        pred_emb_flat = torch.cat((pos_emb_pred.unsqueeze(1), pos_emb_pred_expanded), dim=1).flatten(end_dim=1)
        trg_emb_flat = torch.cat((pos_emb_trg.unsqueeze(1), neg_emb_inp), dim=1).flatten(end_dim=1)

        n_neg_items = neg_emb_inp.size(1)
        n_pos_ex = pos_emb_trg.size(0)
        n_neg_ex = neg_emb_inp.size(0) * n_neg_items
        labels = torch.LongTensor([n_neg_items] * n_pos_ex).to(self.device)

        # compute similarity
        if self.similarity_type == 'concat_mlp':
            pos_cos_score = self.sim_mlp(torch.cat((pos_emb_pred, pos_emb_trg), dim=1))
            neg_cos_score = self.sim_mlp(torch.cat((pos_emb_pred_expanded, neg_emb_inp), dim=2)).squeeze(2)
        elif self.similarity_type == 'cosine':
            pos_cos_score = torch.cosine_similarity(pos_emb_pred, pos_emb_trg).unsqueeze(1)
            neg_cos_score = torch.cosine_similarity(pos_emb_pred_expanded, neg_emb_inp, dim=2)

        # compute predictionss (logits)
        cos_sim_concat = torch.cat((neg_cos_score, pos_cos_score), dim=1)
        items_prob_log = F.log_softmax(cos_sim_concat, dim=1)
        predictions = torch.exp(items_prob_log)

        # Step5. Compute loss and accuracy

        if self.loss_type in ['cross_entropy_neg', 'cross_entropy_neg_1d']:

            loss = self.loss_fn(items_prob_log, labels)

        elif self.loss_type.startswith('margin_hinge'):

            # _label = torch.LongTensor([1] * n_pos_ex + [-1] * n_neg_ex).to(pred_emb_flat.device)

            # loss = self.loss_fn(pred_emb_flat, trg_emb_flat, _label) / num_elem 
            pos_dist, neg_dist = pos_cos_score, neg_cos_score
            
            if self.loss_type == 'margin_hinge_a':
                # case A
                loss = (pos_dist.sum() + torch.relu(self.margin_loss - neg_dist).sum()) / (n_pos_ex + n_neg_ex)
            elif self.loss_type == 'margin_hinge_b':
                # case B (case of the paper)
                n_neg_samples = neg_emb_inp.size(1)
                loss = (pos_dist.sum() * n_neg_samples + torch.relu(self.margin_loss - neg_dist).sum()) / (n_pos_ex + n_neg_ex)

        elif self.loss_type == 'cross_entropy':

            # compute logits (predicted probability of item ids)
            logits_all = self.output_layer(pos_emb_pred)
            pred_flat = self.log_softmax(logits_all)

            loss = self.loss_fn(pred_flat, trg_flat_nonpad)
            
        else:
            raise NotImplementedError

        # accuracy
        _, max_idx = torch.max(cos_sim_concat, dim=1)
        train_acc = (max_idx == n_neg_items).sum(dtype=torch.float32) / num_elem
        print(f"train_acc: {train_acc}")

        self.cnts.extend(max_idx.cpu().tolist())

        outputs = (train_acc, loss, predictions, labels) + model_outputs  # Keep mems, hidden states, attentions if there are in it

        return outputs  # return (train_acc), (loss), (predictions), (labels), (mems), (hidden states), (attentions)


def nll_1d(items_prob, _label=None):
    # https://github.com/gabrielspmoreira/chameleon_recsys/blob/da7f73a2b31d6867d444eded084044304b437413/nar_module/nar/nar_model.py#L639
    items_prob = torch.exp(items_prob)
    positive_prob = items_prob[:, 0]
    xe_loss = torch.log(positive_prob)
    cosine_sim_loss = - torch.mean(xe_loss)
    return cosine_sim_loss