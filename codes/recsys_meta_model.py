"""
A meta class supports various (Huggingface) transformer models for RecSys tasks.

"""

import logging

import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)


class RecSysMetaModel(PreTrainedModel):
    """
    vocab_sizes : sizes of vocab for each discrete inputs
        e.g., [product_id_vocabs, category_vocabs, etc.]
    """
    def __init__(self, model, config, model_args, data_args):
        super(RecSysMetaModel, self).__init__(config)
        
        self.model = model 

        if self.model.__class__ in [nn.GRU, nn.LSTM, nn.RNN]:
            self.is_rnn = True
        else:
            self.is_rnn = False

        self.pad_token = data_args.pad_token

        # set embedding tables
        self.embedding_product = nn.Embedding(data_args.num_product, model_args.d_model, padding_idx=self.pad_token)
        self.embedding_category = nn.Embedding(data_args.num_category, model_args.d_model, padding_idx=self.pad_token)

        self.merge = model_args.merge_inputs
        
        if self.merge == 'concat_mlp':
            n_embeddings = data_args.num_categorical_features
            self.mlp_merge = nn.Linear(model_args.d_model * n_embeddings, model_args.d_model)
        
        self.similarity_type = model_args.similarity_type
        self.margin_loss = model_args.margin_loss
        self.output_layer = nn.Linear(model_args.d_model, data_args.num_product)
        self.loss_type = model_args.loss_type
        self.neg_log_likelihood = nn.NLLLoss(ignore_index=self.pad_token)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # NOTE: https://pytorch.org/docs/master/generated/torch.nn.CosineEmbeddingLoss.html
        self.cosine_emb_loss = nn.CosineEmbeddingLoss(margin=model_args.margin_loss, reduction='sum')

    def _unflatten_neg_seq(self, neg_seq, seqlen):
        """
        neg_seq: n_batch x (num_neg_samples x max_seq_len); flattened. 2D.
        """
        assert neg_seq.dim() == 2

        n_batch, flatten_len = neg_seq.size()
        
        assert flatten_len % seqlen == 0

        n_neg_seqs_per_pos_seq = flatten_len // seqlen
        return neg_seq.reshape((n_batch, n_neg_seqs_per_pos_seq, seqlen))

    def forward(
        self,
        product_seq,
        category_seq,
        neg_product_seq=None,
        neg_category_seq=None,
    ):
        """
        For cross entropy loss, we split input and target BEFORE embedding layer.
        For margin hinge loss, we split input and target AFTER embedding layer.
        """
        
        # Step1. Obtain Embedding
        product_seq_trg = product_seq[:, 1:] 
        if self.loss_type == 'cross_entropy':    
            product_seq = product_seq[:, :-1]
            category_seq = category_seq[:, :-1]

        pos_prd_emb = self.embedding_product(product_seq)
        pos_cat_emb = self.embedding_category(category_seq)
        
        if self.loss_type == 'margin_hinge':

            # unflatten negative sample
            max_seq_len = product_seq.size(1)
            neg_product_seq = self._unflatten_neg_seq(neg_product_seq, max_seq_len)
            neg_category_seq = self._unflatten_neg_seq(neg_category_seq, max_seq_len)
            
            # obtain embeddings
            neg_prd_emb = self.embedding_product(neg_product_seq)
            neg_cat_emb = self.embedding_category(neg_category_seq)

        # Step 2. Merge features

        if self.merge == 'elem_add':
            pos_emb = pos_prd_emb + pos_cat_emb
            
            if self.loss_type == 'margin_hinge':
                neg_emb = neg_prd_emb + neg_cat_emb

        elif self.merge == 'concat_mlp':
            pos_emb = torch.tanh(self.mlp_merge(torch.cat((pos_prd_emb, pos_cat_emb), dim=-1)))

            if self.loss_type == 'margin_hinge':
                neg_emb = torch.tanh(self.mlp_merge(torch.cat((neg_prd_emb, neg_cat_emb), dim=-1)))

        else:
            raise NotImplementedError

        if self.loss_type == 'margin_hinge':
            # set input and target from emb
            pos_emb_inp = pos_emb[:, :-1]
            pos_emb_trg = pos_emb[:, 1:]

            neg_emb_inp = neg_emb[:, :, :-1]
            neg_emb_trg = neg_emb[:, :, 1:]

        elif self.loss_type == 'cross_entropy':
            pos_emb_inp = pos_emb

        # Step3. Run forward pass on model architecture

        if self.is_rnn:
            # compute output through RNNs
            model_outputs = self.model(
                input=pos_emb_inp
            )
            
        else:
            # compute output through transformer
            model_outputs = self.model(
                inputs_embeds=pos_emb_inp,
            )
            
        pos_emb_pred = model_outputs[0]

        # compute logits (predicted probability of item ids)
        logits = self.output_layer(pos_emb_pred)
        outputs = (logits,) + model_outputs[1:]  # Keep mems, hidden states, attentions if there are in it
        
        pred_flat = self.log_softmax(logits).flatten(end_dim=1)
        trg_flat = product_seq_trg.flatten()
        num_elem = (product_seq_trg.flatten() != self.pad_token).sum()

        # Step4. Compute loss and accuracy

        if self.loss_type == 'margin_hinge':

            pos_emb_pred_expanded = pos_emb_pred.unsqueeze(1).expand_as(neg_emb_trg)
            pred_emb_flat = torch.cat((pos_emb_pred.unsqueeze(1), pos_emb_pred_expanded), dim=1).flatten(end_dim=2)
            trg_emb_flat = torch.cat((pos_emb_trg.unsqueeze(1), neg_emb_trg), dim=1).flatten(end_dim=2)

            n_pos_ex = pos_emb_trg.size(0) * pos_emb_trg.size(1)
            n_neg_ex = neg_emb_trg.size(0) * neg_emb_trg.size(1) * neg_emb_trg.size(2)
            _label = torch.LongTensor([1] * n_pos_ex + [-1] * n_neg_ex).to(pred_emb_flat.device)

            loss_sum = self.cosine_emb_loss(pred_emb_flat, trg_emb_flat, _label)
            loss = loss_sum / num_elem

            # accuracy
            pos_emb_pred_expanded = pos_emb_pred.unsqueeze(1).expand_as(neg_emb_trg)
            pos_sim = F.cosine_similarity(pos_emb_trg, pos_emb_pred, dim=2)		
            pos_sim = pos_sim.unsqueeze(1)		
            neg_sim = F.cosine_similarity(neg_emb_trg, pos_emb_pred_expanded, dim=3)		
            combine_sim = torch.cat((pos_sim, neg_sim), dim=1)
            _, max_idx = torch.max(combine_sim, 1)
            train_acc = (max_idx == 0).sum(dtype=torch.float32) / num_elem

        elif self.loss_type == 'cross_entropy':
            loss = self.neg_log_likelihood(pred_flat, trg_flat)
    
            # accuracy
            _, max_idx = torch.max(pred_flat, 1)
            train_acc = (max_idx == trg_flat).sum(dtype=torch.float32) / num_elem

        else:
            raise NotImplementedError

        outputs = (loss,) + outputs

        
        outputs = (train_acc,) + outputs

        return outputs  # return (train_acc), (loss), logits, (mems), (hidden states), (attentions)
