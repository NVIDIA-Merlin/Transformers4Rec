from typing import Optional, Callable, Any, Dict, List 
import logging
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from transformers import ElectraModel, GPT2Model, PreTrainedModel, XLNetModel

from .loss_functions import BPR, TOP1, BPR_max, BPR_max_reg, TOP1_max
from .recsys_tasks import RecSysTask


logger = logging.getLogger(__name__)


###############################################################
#                                                             #
#         Object classes for Transformers4rec inputs          #
#                                                             #
###############################################################

class Categorical(object): 
    """
    Class to build the embedding representation of categorical variable  
    """
    def __init__(self, 
        	    name : str, 
                cardinality : int, 
                is_itemid: bool = False,
                is_seq_label: bool =  False,
                is_classification_label: bool = False,
                is_regression_label: bool = False,
                log_with_preds_as_metadata: bool = True
    )
    self.name = name 
    self.cardinality = cardinality 
    self.is_itemid = is_itemid
    self.is_seq_label = is_seq_label
    self.is_classification_label = is_classification_label
    self.log_with_preds_as_metadata = log_with_preds_as_metadata


    def get_embedding_table(self, embedding_size, pad_token):
        self.feature_size = embedding_size
        self.table = nn.Embedding(
                        self.cardinality,
                        embedding_size,
                        padding_idx=pad_token,
                    ) 
    
    def get_layer_norm(self): 
        self.layer_norm = nn.LayerNorm(
                    normalized_shape=self.feature_size
                )

    def init_embedding_table(self, embeddings_init_std):
            with torch.no_grad():
                self.table.weight.normal_( 0.0,
                 embeddings_init_std)
    
    def get_projection_layer(self, output_dim, bias=True):
        self.project_layer = nn.Linear(self.feature_size,
                                   output_dim, bias=bias)

    


class Conitnuous(object): 
    """
    Class to build the embedding representation of categorical variable  
    """
    def __init__(self, 
        	    name : str, 
                dtype : str, 
                representation_type: bool = False,
                log_with_preds_as_metadata: bool =  True,
    )
    self.name = name 
    self.dtype = cardinality 
    self.representation_type = representation_type
    self.log_with_preds_as_metadata = log_with_preds_as_metadata


    def get_representation_layer(self, 
                            feature_size, 
                            embeddings_init_std,
                            numeric_features_soft_one_hot_encoding_num_embeddings=None):
        self.feature_size = feature_size
        if self.representation_type == 'soft_one_hot_encoding_num_embeddings':
            self.table = SoftEmbedding(
                            num_embeddings=numeric_features_soft_one_hot_encoding_num_embeddings,
                            embeddings_dim=feature_size,
                            embeddings_init_std=embeddings_init_std,
                        )
        else: 
            self.table = nn.Linear(1, feature_size, bias=True)

        
    def get_layer_norm(self): 
        self.layer_norm = nn.LayerNorm(
                    normalized_shape=self.feature_size
                )
    
    def get_projection_layer(self, output_dim, bias=True):
        self.project_layer = nn.Linear(self.feature_size,
                                   output_dim, bias=bias)


class FeatureProcessConfig(object):
    """
    Config to define how to represent an input sequence of group of features 
    """
    def __init__(self,
                 pad_token: str,
                 use_ohe_item_ids_inputs: bool,
                 item_embedding_dim: int, 
                 embedding_dim_from_cardinality_multiplier: float,
                 features_same_size_item_embedding: bool, 
                 mf_constrained_embeddings: bool, 
                 layer_norm_featurewise: bool,
                 layer_norm_all_features: bool,
                 item_id_embeddings_init_std: float,
                 other_embeddings_init_std: float,
                 numeric_features_project_to_embedding_dim: int, 
                 numeric_features_soft_one_hot_encoding_num_embeddings: int,
                 stochastic_shared_embeddings_replacement_prob: float,
                 input_features_aggregation: str,
                 tasks: List[str]=None, 
                 ):
        self.feature_map = feature_map
        self.pad_token = pad_token
        self.item_embedding_dim = item_embedding_dim
        self.use_ohe_item_ids_inputs =use_ohe_item_ids_inputs
        self.features_same_size_item_embedding = features_same_size_item_embedding
        self.embedding_dim_from_cardinality_multiplier = embedding_dim_from_cardinality_multiplier
        self.layer_norm_featurewise = layer_norm_featurewise
        self.layer_norm_all_features = layer_norm_all_features
        self.item_id_embeddings_init_std = item_id_embeddings_init_std
        self.other_embeddings_init_std = other_embeddings_init_std
        self.input_features_aggregation = input_features_aggregation
        self.mf_constrained_embeddings = .mf_constrained_embeddings
        self.numeric_features_project_to_embedding_dim =numeric_features_project_to_embedding_dim
        self.numeric_features_soft_one_hot_encoding_num_embeddings = numeric_features_soft_one_hot_encoding_num_embeddings
        self.tasks = tasks


class FeatureGroup(object):
    """
    Class to store the aggregated Tensor of of a group of categoricals and continuous
    defined in the same featuremap 
    Args: 
    """
    def __init__(self,
                 name: str,
                 Values: Torch.Tensor,
                 Metadata: List[str])
    self.name = name
    self.value = value
    self.metadata = metadata

    
class LabelFeature(object):
    """
    Class to store label column name of the prediction head
    Args: 
        type: type of the prediction head:  item_prediction | classification | regression
        label column: label column name
    """
    def __init__(self, 
                 type: str
                 label_column: str
                 dimension: int)
        self.type = type
        self.label_column = label_column
        self.dimension = dimension

        
class FeatureProcessOutput(object):
    """
    Class returned by FeatureProcess module with the group of sequences representations and prediction heads labels
    """
    def __init__(self, 
                 feature_groups: List[FeatureGroup],
                 label_groups: List[LabelFeature],
                 metadata_features: List[str],
                )
    
      

####################################################################################
#                                                                                  #
# FeatureProcess module : prepare multiples FeatureGroups for session-based tasks  #
#                                                                                  #
####################################################################################

class FeatureGroupProcess(PreTrainedModel): 
    """
    Process the dictionary of input tensors to prepare the sequence of interactions embeddings for Transformer blocks 
    
    Args: 
        config: HuggingFace configuration class 
        model_args: ModelArguments class specifying parameters of the neural network architecture 
        data_args: DataArguments class specifying arguments for data loading and preparation  
        feature_map: dictionary loaded from feature config yaml file specifying the name and characteristics of input and label features. 
    
    Outputs: 
        output: sequence of item interaction embeddings
        label_seq: sequence of item ids 
        classification_labels: tensor of classification targets 
        metadata_for_pred_logging: dictionary with item metadata features for prediction logging 
    """

    def __init__(self,
                name: str,
                feature_config: FeatureProcessConfig,
                feature_map: dict): 
        super(FeatureGroupProcess, self).__init__()

        # Init configs
        self.name = name 
        self.feature_map = feature_map
        self.feature_config = feature_config

        # Init input classes
        self.categoricals, self.continuous = init_from_featuremap(feature_map)

        # Init outputs 
        self.labels = []
        self.FeatureGroup = None
        self.metadata_for_pred_logging = {}

        # compute item embedding dim : 
        self.item_embedding_dim, self.itemid_name = self.get_itemid_embedding_dim(self.categoricals) 

        # Get label columns 
        for variable in self.categoricals + self.continuous: 
            if variable.is_classification_label : 
                self.labels.append(LabelFeature(type='classification', label_column=variable.name, dimension=variable.cardinality))
            elif variable.is_seq_label: 
                self.labels.append(LabelFeature(type='item_prediction', label_column=variable.name, dimension=variable.cardinality))
            elif variable.is_regression_label: 
                self.labels.append(LabelFeature(type='regression', label_column=variable.name, dimension=variable.cardinality))

        # Get representation tables of input features 
        for cat in self.categoricals:
            if cat.is_itemid: 
                cat.get_embedding_table(self.item_embedding_dim, self.feature_config.pad_token)
                cat.init_embedding_table(self.item_id_embeddings_init_std)
            else: 
                if self.feature_config.features_same_size_item_embedding:
                        cat.get_embedding_table(self.item_embedding_dim, self.feature_config.pad_token)
                else:
                    embedding_size = get_embedding_size_from_cardinality(
                            cat.cardinality
                            multiplier=self.feature_config.embedding_dim_from_cardinality_multiplier,
                        )
                    cat.get_embedding_table(self.item_embedding_dim, self.feature_config.pad_token)
                cat.init_embedding_table(self.feature_config.other_embeddings_init_std)
                if  self.feature_config.input_features_aggregation == "elementwise_sum_multiply_item_embedding":
                    cat.get_projection_layer(self.item_embedding_dim)
        
        for cont in self.continuous:
            if self.feature_config.numeric_features_project_to_embedding_dim > 0 :
                if self.features_same_size_item_embedding:
                    cont.get_representation_layer(self.item_embedding_dim, 
                                                self.feature_config.other_embeddings_init_std,
                                                self.feature_config.numeric_features_soft_one_hot_encoding_num_embeddings)
                if  self.feature_config.input_features_aggregation == "elementwise_sum_multiply_item_embedding":
                    cont.get_projection_layer(self.item_embedding_dim)
                                                
        if self.layer_norm_featurewise: 
            for variable in self.continuous + self.categoricals:
                variable.get_layer_norm() 
        
        # get name of  for meta_data 
        self.metadata_for_pred_logging = []
        for variable in self.continuous + self.categoricals:
            if variable.log_with_preds_as_metadata:
                self.metadata_for_pred_logging.append(variable.name)
        
        # compute combined_dim 
        if self.feature_config.input_features_aggregation == "elementwise_sum_multiply_item_embedding":
            self.input_combined_dim = self.item_embedding_dim
        else: 
            self.input_combined_dim = np.sum(cat.feature_size for cat in self.continuous + self.categoricals)

        # Init aggregation module 
        self.aggregate = Aggregation(self.feature_config.input_features_aggregation, self.itemid_name)

    def forward(self, inputs):
        transformed_features = OrderedDict()
        metadata_for_pred_logging =  OrderedDict()
        for variable in self.categoricals + self.continuous: 
            cdata = inputs[variable.name]
            if (
                self.feature_config.stochastic_shared_embeddings_replacement_prob > 0.0 and 
                self.training
                ):
                cdata = stochastic_swap_noise(cdata, self.feature_config.pad_token, self.feature_config.stochastic_shared_embeddings_replacement_prob)
                cdata = variable.table(cdata)
            
            if self.layer_norm_featurewise:
                cdata = variable.layer_norm(cdata)
            transformed_features[variable.name] = cdata

        
        # Create aggregated FeatureGroup class 
        output = self.aggregate(transformed_features)
        return FeatureGroup(self.name, 
                            output, 
                            metadata_for_pred_logging)


class FeatureProcess(PreTrainedModel): 
    """
    Process multiple  groups of features and return a list of classes: FeatureGroup and LabelFeature 
    """

    def __init__(self,
                names: List[str],
                feature_configs: FeatureProcessConfig,
                feature_maps: dict): 
        super(FeatureGroupProcess, self).__init__()

        # Init configs
        self.names = names
        self.feature_maps = feature_maps
        self.feature_configs = feature_configs

        # Init Feature groups 
        self.feature_groups = [FeatureGroupProcess(name, feature_config, feature_map) for 
                                (name, feature_map, feature_config) in zip(
                                    self.names, self.feature_maps, self.feature_configs
                                     )
                            ]
    def forward(self, inputs):
        feature_groups = [feat_group(inputs) for feat_group in self.feature_groups]
        labels_features = [feat_group.labels for feat_group in feature_groups]
        labels_features = sum(labels_features, [])
        metadata_for_pred_logging = [feat_group.metadata_for_pred_logging for feat_group in feature_groups]
        metadata_for_pred_logging = sum(metadata_for_pred_logging, [])
    return FeatureProcessOutput(feature_groups, 
                                labels_features,
                                metadata_for_pred_logging)


##########################################
# utils needed by FeatureProcess modules #
##########################################

def get_embedding_size_from_cardinality(cardinality, multiplier=2.0):
    # A rule-of-thumb from Google.
    embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))
    return embedding_size

class SoftEmbedding(nn.Module):
    """
    Soft-one hot encoding embedding, from https://arxiv.org/pdf/1708.00065.pdf
    """

    def __init__(self, num_embeddings, embeddings_dim, embeddings_init_std=0.05):
        super(SoftEmbedding, self).__init__()
        self.embedding_table = nn.Embedding(num_embeddings, embeddings_dim)
        with torch.no_grad():
            self.embedding_table.weight.normal_(0.0, embeddings_init_std)
        self.projection_layer = nn.Linear(1, num_embeddings, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_numeric):
        weights = self.softmax(self.projection_layer(input_numeric))
        soft_one_hot_embeddings = (
            weights.unsqueeze(-1) * self.embedding_table.weight
        ).sum(-2)
        return soft_one_hot_embeddings

class Projection(nn.Module): 
    """
    Project continuous variable to higher dimension space 
    to have the same dimension as other categoricals
    """
    def __init__(self, input_dim, output_dim, bias=True):
    super(Projection, self).__init__()
    self.project_layer = nn.Linear(input_dim,
                                   output_dim, bias=bias)

    def forward(self, input_numeric):
        return self.project_layer(input_numeric)                          
         
    

class Aggregation(nn.Module):
    def __init__(self, input_features_aggregation, itemid_name):
        self.input_features_aggregation = input_features_aggregation
        self.itemid_name = itemid_name
        
        
    def forward(transformed_features):
        if len(transformed_features) > 1:
                if self.input_features_aggregation == "concat":
                    output = torch.cat(list(transformed_features.values()), dim=-1)
                elif (
                    self.input_features_aggregation
                    == "elementwise_sum_multiply_item_embedding"
                ):
                    additional_features_sum = torch.zeros_like(
                        transformed_features[self.label_feature_name], device=self.device
                    )
                    for k, v in transformed_features.items():
                        if k != self.itemid_name:
                            if (self.feature_map[k]["dtype"] == "categorical") or (
                                self.feature_map[k]["dtype"] in ["long", "float"]
                                and self.numeric_features_project_to_embedding_dim > 0
                            ):
                                additional_features_sum += v

                    item_id_embedding = transformed_features[self.itemid_name]

                    output = item_id_embedding * (additional_features_sum + 1.0)
                else:
                    raise ValueError("Invalid value for --input_features_aggregation.")
            else:
                output = list(transformed_features.values())[0]
        return output 


def stochastic_swap_noise(cdata: torch.Tensor, 
                          pad_token: int,
                          stochastic_shared_embeddings_replacement_prob: float,
                          
                         )
    """
    Applies Stochastic replacement of sequence features 
    """ 
    with torch.no_grad():
        cdata_non_zero_mask = cdata != pad_token
        sse_prob_replacement_matrix = torch.full(
            cdata.shape,
            stochastic_shared_embeddings_replacement_prob,
            device=cdata.device,
        )
        sse_replacement_mask = (
            torch.bernoulli(sse_prob_replacement_matrix).bool()
            & cdata_non_zero_mask
        )
        n_values_to_replace = sse_replacement_mask.sum()
        
        cdata_flattened_non_zero = torch.masked_select(
            cdata, cdata_non_zero_mask
        )

        sampled_values_to_replace = cdata_flattened_non_zero[
            torch.randperm(cdata_flattened_non_zero.shape[0])
        ][:n_values_to_replace]

        cdata[sse_replacement_mask] = sampled_values_to_replace
    return cdata




    