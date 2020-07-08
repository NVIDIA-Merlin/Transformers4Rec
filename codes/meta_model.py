"""
A meta class supports various (Huggingface) transformer models for RecSys tasks.

"""

import logging

import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class RecSysMetaModel(nn.Module):
    """
    vocab_sizes : sizes of vocab for each discrete inputs
        e.g., [product_id_vocabs, category_vocabs, etc.]
    """
    def __init__(self, model, vocab_sizes, d_model, 
                 merge_inputs='add', similarity_type='cos', margin_loss=1.0, embed_pad_token=0):
        super(RecSysMetaModel, self).__init__()
        
        self.model = model 

        if self.model.__class__ in [nn.GRU, nn.LSTM, nn.RNN]:
            self.is_rnn = True
        else:
            self.is_rnn = False

        # set embedding tables
        self.embedding_product = nn.Embedding(vocab_sizes[0], d_model, padding_idx=embed_pad_token)
        self.embedding_category = nn.Embedding(vocab_sizes[1], d_model, padding_idx=embed_pad_token)

        self.merge = merge_inputs
        
        if self.merge == 'mlp':
            n_embeddings = len(vocab_sizes)
            self.mlp_merge = nn.Linear(d_model * n_embeddings, d_model)
        
        self.similarity_type = similarity_type
        self.margin_loss = margin_loss
        self.output_layer = nn.Linear(d_model, vocab_sizes[0])

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
        neg_product_seq,
        neg_category_seq,
    ):
        # unflatten negative sample
        max_seq_len = product_seq.size(1)
        neg_product_seq = self._unflatten_neg_seq(neg_product_seq, max_seq_len)
        neg_category_seq = self._unflatten_neg_seq(neg_category_seq, max_seq_len)

        # obtain embeddings
        # NOTE: padding elements are automatically removed through padded_idx option
        pos_prd_emb = self.embedding_product(product_seq)
        pos_cat_emb = self.embedding_category(category_seq)

        neg_prd_emb = self.embedding_product(neg_product_seq)
        neg_cat_emb = self.embedding_category(neg_category_seq)

        # merge different features
        if self.merge == 'add':
            pos_emb_seq = pos_prd_emb + pos_cat_emb
            neg_emb_seq = neg_prd_emb + neg_cat_emb

        elif self.merge == 'mlp':
            pos_emb_seq = F.tanh(self.mlp_merge(torch.cat(pos_prd_emb, pos_cat_emb)))
            neg_emb_seq = F.tanh(self.mlp_merge(torch.cat(neg_prd_emb, neg_cat_emb)))

        else:
            raise NotImplementedError

        # set input and target from emb_seq
        pos_emb_seq_inp = pos_emb_seq[:, :-1]
        pos_emb_seq_trg = pos_emb_seq[:, 1:]

        neg_emb_seq_inp = neg_emb_seq[:, :, :-1]
        neg_emb_seq_trg = neg_emb_seq[:, :, 1:]

        if self.is_rnn:
            # compute output through RNNs
            model_outputs = self.model(
                input=pos_emb_seq_inp
            )
            
        else:
            # compute output through transformer
            model_outputs = self.model(
                inputs_embeds=pos_emb_seq_inp,
            )
            
        pos_emb_seq_pred = model_outputs[0]

        # compute similarity 
        if self.similarity_type == 'cos':
            pos_emb_seq_pred_expanded = pos_emb_seq_pred.unsqueeze(1).expand_as(neg_emb_seq_trg)
            pos_sim_seq = F.cosine_similarity(pos_emb_seq_trg, pos_emb_seq_pred, dim=2)
            pos_sim_seq = pos_sim_seq.unsqueeze(1)
            neg_sim_seq = F.cosine_similarity(neg_emb_seq_trg, pos_emb_seq_pred_expanded, dim=3)

        elif self.similarity_type == 'softmax':
            # TODO: ref. https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/nar_module/nar/nar_model.py#L508
            raise NotImplementedError

        else:
            raise NotImplementedError

        # compute logits (predicted probability of item ids)
        logits = self.output_layer(pos_emb_seq_pred)
        outputs = (logits,) + model_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        # compute margin (hinge) loss.
        # NOTE: simply taking average here.
        loss = - (pos_sim_seq.sum(-1) + self.margin_loss - neg_sim_seq.sum(-1)).mean()
        
        outputs = (loss,) + outputs

        return outputs  # return (loss), logits, (mems), (hidden states), (attentions)
