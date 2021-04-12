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
import logging

import torch
import torch.nn as nn

# load transformer model and its configuration classes
from transformers import (
    AlbertConfig,
    AlbertModel,
    ElectraConfig,
    ElectraModel,
    GPT2Config,
    GPT2Model,
    LongformerConfig,
    LongformerModel,
    PretrainedConfig,
    ReformerConfig,
    ReformerModel,
    TransfoXLConfig,
    TransfoXLModel,
    XLNetConfig,
    XLNetModel,
)

logger = logging.getLogger(__name__)


def get_recsys_model(model_args, data_args, training_args, target_size=None):
    total_seq_length = data_args.total_seq_length

    # For Causal LM (not Masked LM), reduces the length by 1 because the
    # sequence is shifted and trimmed (last item in the sequence is not used as input because it is the last)
    # if not model_args.mlm:
    #    total_seq_length -= 1

    if model_args.model_type == "avgseq":
        model_cls = AvgSeq()
        config = PretrainedConfig()

    elif model_args.model_type == "reformer":
        model_cls = ReformerModel
        config = ReformerConfig(
            attention_head_size=model_args.d_model,
            attn_layers=["local", "lsh"] * (model_args.n_layer // 2)
            if model_args.n_layer > 2
            else ["local"],
            is_decoder=not model_args.mlm,  # is_decoder must be False for Masked LM and True for Causal LM
            feed_forward_size=model_args.d_model * 4,
            hidden_size=model_args.d_model,
            num_attention_heads=model_args.n_head,
            hidden_act=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            hidden_dropout_prob=model_args.dropout,
            lsh_attention_probs_dropout_prob=model_args.dropout,
            pad_token_id=data_args.pad_token,
            axial_pos_embds=True,  # If `True` use axial position embeddings.
            axial_pos_shape=[
                model_args.axial_pos_shape_first_dim,
                total_seq_length // model_args.axial_pos_shape_first_dim,
            ],  # The position dims of the axial position encodings. During training the product of the position dims has to equal the sequence length.
            axial_pos_embds_dim=[
                model_args.d_model // 2,
                model_args.d_model // 2,
            ],  # The embedding dims of the axial position encodings. The sum of the embedding dims has to equal the hidden size.
            lsh_num_chunks_before=model_args.num_chunks_before,
            lsh_num_chunks_after=model_args.num_chunks_after,
            num_buckets=None,  # Number of buckets, the key query vectors can be 'hashed into' using the locality sensitive hashing scheme. When training a model from scratch, it is recommended to leave config.num_buckets=None, so that depending on the sequence length a good value for num_buckets is calculated on the fly. This value will then automatically be saved in the config and should be reused for inference.
            num_hashes=model_args.lsh_num_hashes,
            local_attn_chunk_length=model_args.attn_chunk_length,
            lsh_attn_chunk_length=model_args.attn_chunk_length,
            chunk_size_feed_forward=model_args.chunk_size_feed_forward,
            output_attentions=training_args.log_attention_weights,
            max_position_embeddings=data_args.total_seq_length,
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
            # vocab_size = model_args.d_model  # to make it output hidden states size
        )

    elif model_args.model_type == "transfoxl":
        model_cls = TransfoXLModel
        config = TransfoXLConfig(
            d_model=model_args.d_model,
            d_embed=model_args.d_model,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            d_inner=model_args.d_model * 4,
            untie_r=True,
            attn_type=0,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            pad_token_id=data_args.pad_token,
            output_attentions=training_args.log_attention_weights,
            mem_len=1,  # We do not use mems, because we feed the full sequence to the Transformer models and not sliding segments (which is useful for the long sequences in NLP. As setting mem_len to 0 leads to NaN in loss, we set it to one, to minimize the computing overhead)
            div_val=1,  # Disables adaptative input (embeddings), because the embeddings are managed by RecSysMetaModel
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
        )

    elif model_args.model_type == "xlnet":
        model_cls = XLNetModel
        config = XLNetConfig(
            d_model=model_args.d_model,
            d_inner=model_args.d_model * 4,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            ff_activation=model_args.hidden_act,
            untie_r=True,
            bi_data=False,
            attn_type="bi",
            summary_type=model_args.summary_type,
            use_mems_train=True,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            pad_token_id=data_args.pad_token,
            output_attentions=training_args.log_attention_weights,
            mem_len=1,  # We do not use mems, because we feed the full sequence to the Transformer models and not sliding segments (which is useful for the long sequences in NLP. As setting mem_len to 0 leads to NaN in loss, we set it to one, to minimize the computing overhead)
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
        )

    elif model_args.model_type == "gpt2":
        model_cls = GPT2Model
        config = GPT2Config(
            n_embd=model_args.d_model,
            n_inner=model_args.d_model * 4,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            activation_function=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            resid_pdrop=model_args.dropout,
            embd_pdrop=model_args.dropout,
            attn_pdrop=model_args.dropout,
            n_positions=data_args.total_seq_length,
            n_ctx=data_args.total_seq_length,
            output_attentions=training_args.log_attention_weights,
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
        )

    elif model_args.model_type == "longformer":
        model_cls = LongformerModel
        config = LongformerConfig(
            hidden_size=model_args.d_model,
            num_hidden_layers=model_args.n_layer,
            num_attention_heads=model_args.n_head,
            hidden_act=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            max_position_embeddings=data_args.total_seq_length,
            pad_token_id=data_args.pad_token,
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
        )

    elif model_args.model_type == "electra":
        model_cls = ElectraModel
        config = ElectraConfig(
            hidden_size=model_args.d_model,
            embedding_size=model_args.d_model,
            num_hidden_layers=model_args.n_layer,
            num_attention_heads=model_args.n_head,
            intermediate_size=model_args.d_model * 4,
            hidden_act=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            hidden_dropout_prob=model_args.dropout,
            max_position_embeddings=data_args.total_seq_length,
            pad_token_id=data_args.pad_token,
            vocab_size=1,
        )

    elif model_args.model_type == "albert":
        # If --num_hidden_groups -1 (used on hypertuning), uses --n_layer value
        if model_args.num_hidden_groups == -1:
            model_args.num_hidden_groups = model_args.n_layer

        model_cls = AlbertModel
        config = AlbertConfig(
            hidden_size=model_args.d_model,
            num_attention_heads=model_args.n_head,
            num_hidden_layers=model_args.n_layer,
            num_hidden_groups=model_args.num_hidden_groups,
            inner_group_num=model_args.inner_group_num,
            intermediate_size=model_args.d_model * 4,
            hidden_act=model_args.hidden_act,
            hidden_dropout_prob=model_args.dropout,
            attention_probs_dropout_prob=model_args.dropout,
            max_position_embeddings=data_args.total_seq_length,
            embedding_size=model_args.d_model,  # should be same as dimension of the input to ALBERT
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
        )

    elif model_args.model_type == "gru":
        model_cls = nn.GRU(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig(hidden_size=model_args.d_model,)  # dummy config

    elif model_args.model_type == "lstm":
        model_cls = nn.LSTM(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig(hidden_size=model_args.d_model,)

    elif model_args.model_type == "rnn":
        model_cls = nn.RNN(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig(hidden_size=model_args.d_model,)

    else:
        raise NotImplementedError

    if model_args.model_type in ["gru", "lstm", "gru4rec", "avgseq"]:
        model = model_cls

    elif model_args.model_name_or_path:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        if model_args.model_type == "electra":
            # define two transformers blocks for discriminator and generator model
            if model_args.rtd_tied_generator:
                # Using same model for generator and discriminator
                model = (model_cls(config), ())
            else:
                # Using a smaller generator based on discriminator layers size
                seq_model_disc = model_cls(config)
                # re-define hidden_size parameters for small generator model
                config.hidden_size = int(
                    round(config.hidden_size * model_args.electra_generator_hidden_size)
                )
                config.embedding_size = config.hidden_size
                seq_model_gen = model_cls(config)
                model = (seq_model_gen, seq_model_disc)
        else:
            model = model_cls(config)

    return model, config


class AvgSeq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # seq: n_batch x n_seq x n_dim
        output = []
        for i in range(1, input.size(1) + 1):
            output.append(input[:, :i].mean(1))

        return (torch.stack(output).permute(1, 0, 2),)
