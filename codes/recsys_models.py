
import logging

import torch
import torch.nn as nn

# load transformer model and its configuration classes
from transformers.modeling_xlnet import XLNetModel
from transformers.configuration_xlnet import XLNetConfig
from transformers.modeling_transfo_xl import TransfoXLModel
from transformers.configuration_transfo_xl import TransfoXLConfig
from transformers.modeling_gpt2 import GPT2Model
from transformers.configuration_gpt2 import GPT2Config
from transformers.modeling_longformer import LongformerModel
from transformers.configuration_longformer import LongformerConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_reformer import ReformerModel
from transformers.configuration_reformer import ReformerConfig

from models.gru4rec import GRU4REC

logger = logging.getLogger(__name__)


def get_recsys_model(model_args, data_args, training_args, target_size=None):
    

    if model_args.model_type == 'avgseq':
        model_cls = AvgSeq()
        config = PretrainedConfig()

    elif model_args.model_type == 'reformer':
        model_cls = ReformerModel
        config = ReformerConfig(
            attention_head_size=model_args.d_model,
            attn_layers= ["local", "lsh"] * (model_args.n_layer // 2) \
                if model_args.n_layer > 2 else ["local"],
            is_decoder=True,
            feed_forward_size=model_args.d_model * 4,
            hidden_size=model_args.d_model,
            num_attention_heads=model_args.n_head,
            hidden_act=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            hidden_dropout_prob=model_args.dropout,
            pad_token_id=data_args.pad_token,
            axial_pos_shape=[19,1],
            axial_pos_embds_dim=[model_args.d_model // 2, model_args.d_model // 2],
            vocab_size=model_args.d_model # to make it output hidden states size
        )

    elif model_args.model_type == 'transfoxl':
        model_cls = TransfoXLModel
        config = TransfoXLConfig(
            d_model=model_args.d_model,
            d_embed=model_args.d_model,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            d_inner=model_args.d_model * 4,
            ff_activation=model_args.hidden_act,
            untie_r=True,
            attn_type=0,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            pad_token_id=data_args.pad_token,
        )

    elif model_args.model_type == 'xlnet':
        model_cls = XLNetModel
        config = XLNetConfig(
            d_model=model_args.d_model,
            d_embed=model_args.d_model,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            d_inner=model_args.d_model * 4,
            ff_activation=model_args.hidden_act,
            untie_r=True,
            attn_type="bi",
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            pad_token_id=data_args.pad_token,
        )


    elif model_args.model_type == 'gpt2':
        model_cls = GPT2Model
        config = GPT2Config(
            n_embd=model_args.d_model,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            activation_function=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            n_positions=data_args.max_seq_len,
            pad_token_id=data_args.pad_token,
            output_attentions=training_args.log_attention_weights
        )

    elif model_args.model_type == 'longformer':
        model_cls = LongformerModel
        config = LongformerConfig(
            hidden_size=model_args.d_model,
            num_hidden_layers=model_args.n_layer,
            num_attention_heads=model_args.n_head,
            hidden_act=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            max_position_embeddings=data_args.max_seq_len,
            vocab_size=target_size,
            pad_token_id=data_args.pad_token,
        )

    elif model_args.model_type == 'gru':
        model_cls = nn.GRU(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig()  # dummy config

    elif model_args.model_type == 'lstm':
        model_cls = nn.LSTM(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig()

    elif model_args.model_type == 'rnn':
        model_cls = nn.RNN(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig()

    elif model_args.model_type == 'gru4rec':
        model_cls = GRU4REC(
            input_size=model_args.d_model, 
            hidden_size=model_args.d_model, 
            output_size=model_args.d_model, 
            num_layers=model_args.n_layer, 
            final_act='tanh',
            dropout_hidden=model_args.dropout, 
            dropout_input=model_args.dropout, 
            batch_size=training_args.per_device_train_batch_size, 
            embedding_dim=-1, 
            use_cuda=False
        )

    else:
        raise NotImplementedError

    if model_args.model_type in ['gru', 'lstm', 'gru4rec', 'avgseq']:
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
        model = model_cls(config)
        
    return model, config


class AvgSeq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs_embeds):
        # seq: n_batch x n_seq x n_dim
        output = []
        for i in range(1, inputs_embeds.size(1) + 1):
            output.append(inputs_embeds[:, :i].mean(1))

        return (torch.stack(output).permute(1,0,2),)
