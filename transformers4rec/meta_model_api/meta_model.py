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

import torch
from torch import nn 

from typing import Optional, Callable, Any, Dict, List 

from feature_process import get_feature_process
from mask_sequence import get_masking_task
from tower_model import TowerModel
from prediction_head import ItemPrediction


class MetaModel(nn.Module): 
    """
    Meta-model for a single Task 

    Parameters:
    -----------
        feature_group_config:
        masking_task:
        max_seq_length:
        model_type:
        n_head:
        n_layer:
        
    """
    def __init__(self, 
                 feature_group_config,
                 masking_task,
                 max_seq_length=20,
                 model_type='xlnet', 
                 n_head=4,
                 n_layer=2): 
        super(MetaModel, self).__init__()
        self.feature_group = get_feature_process(feature_group_config).feature_groups[0]
        self.hidden_dim =  self.feature_group.input_combined_dim 
        self.mask_task = get_masking_task(masking_task, self.hidden_dim)
        self.tower_model = TowerModel(max_seq_length=max_seq_length, model_type=model_type, hidden_size=self.hidden_dim, n_head=n_head, n_layer=n_layer)
        self.body = nn.Linear(self.hidden_dim, self.feature_group.item_embedding_dim).to(self.feature_group.feature_config.device)
        task = [x for x in self.feature_group.labels if x.type=='item_prediction'][0]
        self.prediction_head = ItemPrediction(loss=nn.NLLLoss(ignore_index=self.feature_group.feature_config.pad_token),
                                              task =task, body = self.body, feature_process=self.feature_group)
    
    def forward(self, inputs, training=False): 
        out = self.feature_group(inputs)
        input_sequence = out.values
        itemid_seq =  inputs[self.feature_group.itemid_name]
        mask_out = self.mask_task(input_sequence, itemid_seq, training = training)
        tower_out = self.tower_model(mask_out)
        

        trg_flat = mask_out.masked_label.flatten()
        non_pad_mask = trg_flat != self.feature_group.feature_config.pad_token
        labels_all = torch.masked_select(trg_flat, non_pad_mask)
        pred_all = self.remove_pad_3d(tower_out.hidden_rep, non_pad_mask)   
        
        loss = self.prediction_head.compute_loss(inputs=pred_all, targets=labels_all)
        logits_all = self.prediction_head(pred_all)
        outputs = {
            "loss": loss,
            "labels": labels_all,
            "predictions": logits_all,
            "model_outputs": tower_out.model_outputs,  # Keep mems, hidden states, attentions if there are in it
        }
            
        return outputs  

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(
            inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor)
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))

        return out_tensor