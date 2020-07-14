
import logging

import torch.nn as nn

# load transformer model and its configuration classes
from transformers.modeling_xlnet import XLNetModel
from transformers.configuration_xlnet import XLNetConfig
from transformers.modeling_gpt2 import GPT2Model
from transformers.configuration_gpt2 import GPT2Config
from transformers.modeling_longformer import LongformerModel
from transformers.configuration_longformer import LongformerConfig

logger = logging.getLogger(__name__)


def get_recsys_model(model_args, d_model=512, n_layer=12, n_head=8, 
                     dropout=0.1, layer_norm_eps=1e-12, max_seq_len=2048):

    if model_args.model_type == 'xlnet':
        model_cls = XLNetModel
        config = XLNetConfig(
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_inner=d_model * 4,
            ff_activation="gelu",
            untie_r=True,
            attn_type="bi",
            initializer_range=0.02,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )

    #NOTE: gpt2 and longformer are not fully tested supported yet.

    elif model_args.model_type == 'gpt2':
        model_cls = GPT2Model
        config = GPT2Config(
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            activation_function="gelu",
            initializer_range=0.02,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            n_positions=max_seq_len,
        )

    elif model_args.model_type == 'longformer':
        model_cls = LongformerModel
        config = LongformerConfig(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act="gelu",
            initializer_range=0.02,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            max_position_embedding=max_seq_len,
        )

    elif model_args.model_type == 'gru':
        model_cls = nn.GRU(
            input_size=d_model,
            num_layers=n_layer,
            hidden_size=d_model,
            dropout=dropout,
        )

    elif model_args.model_type == 'lstm':
        model_cls = nn.LSTM(
            input_size=d_model,
            num_layers=n_layer,
            hidden_size=d_model,
            dropout=dropout,
        )
    elif model_args.model_type == 'rnn':
        model_cls = nn.RNN(
            input_size=d_model,
            num_layers=n_layer,
            hidden_size=d_model,
            dropout=dropout,
        )

    else:
        raise NotImplementedError

    if model_args.model_type in ['gru', 'lstm']:
        model = model_cls

    elif model_args.model_name_or_path:
        transformer_model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_cls(config)
    
    return model