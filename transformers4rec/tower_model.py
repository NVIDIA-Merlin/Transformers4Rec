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
from typing import Optional, Callable, Any, Dict, List 

import torch
import torch.nn as nn

# load transformer model and its configuration classes
from transformers import (
    PreTrainedModel,
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


class TowerModel(nn.Module): 
    def __init__(self,
                 model_args: List[List] = [['xlnet', 128, 2, 4, 20]],
                 device: str = 'cuda',
                 
                ):
        super(TowerModel, self).__init__()
        self.model_args = model_args
        self.device = device
        
        self.models = [get_recsys_model(*model,  **extra_config[model[0]]).to(device) for model in self.model_args]
        
        
    def forward(self, inputs, **kwargs):
        return [self.get_output(self.models[i], x, **map_mask) for i, (x,  map_mask) in enumerate(inputs)]
            
    
    
    def get_output(self, model, pos_emb_inp, **kwargs): 
        if not isinstance(model, PreTrainedModel):  # Checks if its a transformer
            """
            RNN Models
            """

            results = model(input=pos_emb_inp)

            if type(results) is tuple or type(results) is list:
                pos_emb_pred = results[0]
            else:
                pos_emb_pred = results

            model_outputs = (None,)

        else:
            """
            Transformer Models
            """

            if type(model) is GPT2Model:
                seq_len = pos_emb_inp.shape[1]
                # head_mask has shape n_layer x batch x n_heads x N x N
                head_mask = (
                    torch.tril(
                        torch.ones(
                            (seq_len, seq_len), dtype=torch.uint8, device=self.device
                        )
                    )
                    .view(1, 1, 1, seq_len, seq_len)
                    .repeat(self.n_layer, 1, 1, 1, 1)
                )

                model_outputs = model(
                    inputs_embeds=pos_emb_inp, head_mask=head_mask,
                )
               
            elif 'task' in kwargs and kwargs['task']=='plm':
                assert (
                    type(model) is XLNetModel
                ), "Permutation language model is only supported for XLNET model "
                model_outputs = model(
                    inputs_embeds=pos_emb_inp,
                    target_mapping=kwargs['target_mapping'],
                    perm_mask=kwargs['perm_mask'],
                )

            else:
                model_outputs = model(inputs_embeds=pos_emb_inp)

            pos_emb_pred = model_outputs[0]
            model_outputs = tuple(model_outputs[1:])
        return (pos_emb_pred,model_outputs)
            
            
extra_config = {'reformer' : {'is_decoder': False, 'axial_pos_embds':True, 'axial_pos_shape_first_dim': 4,
                              'lsh_num_chunks_before': 3, 'lsh_num_chunks_after':3, 'num_buckets': None, 
                              'lsh_num_hashes': 5, 'attn_chunk_length': 20 , 'chunk_size_feed_forward': 1},
                
                'transformer_xl':  {'mem_len': 1},
                
                'xlnet': {'untie_r':True,  'bi_data':False,  'attn_type':"bi",  'use_mems_train':True,   'mem_len':1, 
                          'use_rtd': False, 'rtd_tied_generator': True, 'electra_generator_hidden_size': 0.4},
                
                'albert': {'num_hidden_groups':1,  'inner_group_num':1},
                
                'electra': {'rtd_tied_generator': True, 'electra_generator_hidden_size': 0.4 },
                'rnn': {},
                'gru': {},
                'lstm': {},
                "avgseq": {}
               }
        
def get_recsys_model(model_type, d_model, n_head, n_layer,  total_seq_length,
                    hidden_act='gelu', initializer_range=0.01,
                    layer_norm_eps=0.03, dropout=0.3,  pad_token=0,
                    log_attention_weights=True, target_size=None, 
                    model_name_or_path=None, cache_dir=None,
                     **kwargs):
    if model_type == "avgseq":
        model_cls = AvgSeq()
        config = PretrainedConfig()

    elif model_type == "reformer":
        model_cls = ReformerModel
        kwargs['axial_pos_shape']=[kwargs['axial_pos_shape_first_dim'],
                                   total_seq_length //kwargs['axial_pos_shape_first_dim']
                                  ],
        config = ReformerConfig(
            attention_head_size=d_model,
            attn_layers=["local", "lsh"] * (n_layer // 2)
            if n_layer > 2
            else ["local"],
            feed_forward_size=d_model * 4,
            hidden_size=d_model,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=dropout,
            lsh_attention_probs_dropout_prob=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            max_position_embeddings=total_seq_length,
            axial_pos_embds_dim=[
                d_model // 2,
                d_model // 2,
            ],
            vocab_size=1,  
            **kwargs)

    elif model_type == "transfoxl":
        model_cls = TransfoXLModel
        config = TransfoXLConfig(
            d_model=d_model,
            d_embed=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_inner=d_model * 4,
            untie_r=True,
            attn_type=0,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            div_val=1,  # Disables adaptative input (embeddings), because the embeddings are managed by RecSysMetaModel
            vocab_size=1, 
            **kwargs
        )

    elif model_type == "xlnet":
        model_cls = XLNetModel
        config = XLNetConfig(
            d_model=d_model,
            d_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            ff_activation=hidden_act,            
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs
        )

    elif model_type == "gpt2":
        model_cls = GPT2Model
        config = GPT2Config(
            n_embd=d_model,
            n_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            activation_function=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            n_positions=total_seq_length,
            n_ctx=total_seq_length,
            output_attentions=log_attention_weights,
            vocab_size=1,  
            **kwargs
        )

    elif model_type == "longformer":
        model_cls = LongformerModel
        config = LongformerConfig(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            max_position_embeddings=total_seq_length,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1, 
            **kwargs
        )

    elif model_type == "electra":
        model_cls = ElectraModel
        config = ElectraConfig(
            hidden_size=d_model,
            embedding_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            intermediate_size=d_model * 4,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=dropout,
            max_position_embeddings=total_seq_length,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs
        )

    elif model_type == "albert":
        # If --num_hidden_groups -1 (used on hypertuning), uses --n_layer value
        if kwargs['num_hidden_groups'] == -1:
            kwargs['num_hidden_groups'] = n_layer

        model_cls = AlbertModel
        config = AlbertConfig(
            hidden_size=d_model,
            num_attention_heads=n_head,
            num_hidden_layers=n_layer,
            hidden_act=hidden_act,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=total_seq_length,
            embedding_size=d_model,  # should be same as dimension of the input to ALBERT
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            output_attentions=log_attention_weights,
            vocab_size=1, 
            **kwargs
        )

    elif model_type == "gru":
        model_cls = nn.GRU(
            input_size=d_model,
            num_layers=n_layer,
            hidden_size=d_model,
            dropout=dropout,
        )
        config = PretrainedConfig(hidden_size=d_model)  # dummy config

    elif model_type == "lstm":
        model_cls = nn.LSTM(
            input_size=d_model,
            num_layers=n_layer,
            hidden_size=d_model,
            dropout=dropout,
        )
        config = PretrainedConfig(hidden_size=d_model,)

    elif model_type == "rnn":
        model_cls = nn.RNN(
            input_size=d_model,
            num_layers=n_layer,
            hidden_size=d_model,
            dropout=dropout,
        )
        config = PretrainedConfig(hidden_size=d_model)

    else:
        raise NotImplementedError

    if model_type in ["gru", "lstm", "gru4rec", "avgseq"]:
        model = model_cls

    elif model_name_or_path:
        model = model_cls.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
        )
    else:
        if model_type == 'electra' or (model_type == 'xlnet' and kwargs['use_rtd']): 
            # Define two transformers blocks for discriminator and generator model
            if kwargs['rtd_tied_generator']:
                # Using same model for generator and discriminator
                model = (model_cls(config), ())
            else:
                # Using a smaller generator based on discriminator layers size
                discriminator = model_cls(config)
                # update hidden_size parameters for small generator model
                config.hidden_size = int(
                    round(config.hidden_size * kwargs['electra_generator_hidden_size'])
                )
                config.embedding_size = config.hidden_size
                generator = model_cls(config)
                model = (generator, discriminator)
        else:
            model = model_cls(config)

    return model


class AvgSeq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # seq: n_batch x n_seq x n_dim
        output = []
        for i in range(1, input.size(1) + 1):
            output.append(input[:, :i].mean(1))

        return (torch.stack(output).permute(1, 0, 2),)
