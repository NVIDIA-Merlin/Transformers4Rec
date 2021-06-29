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

def get_embedding_size_from_cardinality(cardinality, multiplier=2.0):
    # A rule-of-thumb from Google.
    embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))
    return embedding_size

class SoftEmbedding(nn.Module):
    """
    Soft-one hot encoding embedding, from https://arxiv.org/pdf/1708.00065.pdf
    """

    def __init__(self, num_embeddings, embeddings_dim, embeddings_init_std=0.05):
        super().__init__()
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


class FeatureProcess(PreTrainedModel): 
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


    def __init__(self, config, model_args, data_args, feature_map): 
        super(FeatureProcess, self).__init__(config)

        # Init the information of the output layer with respect to the prediction task: item prediction, classification or both 
        self.target_dim = None
        self.num_classes = None
        self.label_feature_name = None
        self.label_embedding_table_name = None
        self.label_embedding_dim = None

        # Init variables from model and data arguments 
        self.feature_map = feature_map
        self.pad_token = data_args.pad_token
        self.item_embedding_dim = model_args.item_embedding_dim
        self.use_ohe_item_ids_inputs = model_args.use_ohe_item_ids_inputs
        self.features_same_size_item_embedding = (
            model_args.features_same_size_item_embedding
        )
        self.embedding_dim_from_cardinality_multiplier = (
            model_args.embedding_dim_from_cardinality_multiplier
        )
        self.layer_norm_featurewise = model_args.layer_norm_featurewise
        self.layer_norm_all_features = model_args.layer_norm_all_features
        self.item_id_embeddings_init_std = model_args.item_id_embeddings_init_std
        self.other_embeddings_init_std = model_args.other_embeddings_init_std
        self.input_features_aggregation = model_args.input_features_aggregation
        self.mf_constrained_embeddings = model_args.mf_constrained_embeddings
        self.numeric_features_project_to_embedding_dim = (
            model_args.numeric_features_project_to_embedding_dim
        )
        self.numeric_features_soft_one_hot_encoding_num_embeddings = (
            model_args.numeric_features_soft_one_hot_encoding_num_embeddings
        )

        if (
            self.numeric_features_soft_one_hot_encoding_num_embeddings > 0
            and self.numeric_features_project_to_embedding_dim == 0
        ):
            raise ValueError(
                "You must set --numeric_features_project_to_embedding_dim to a value greater than zero when using Soft One-Hot Encoding (--numeric_features_soft_one_hot_encoding_num_embeddings)"
            )
        self.stochastic_shared_embeddings_replacement_prob = (
            model_args.stochastic_shared_embeddings_replacement_prob
        )
        
        # Define dictionary of representation modules for categorical and continuous features 
        self.embedding_tables = nn.ModuleDict()
        self.numeric_to_embedding_layers = nn.ModuleDict()
        self.numeric_soft_embeddings = nn.ModuleDict()
        self.features_embedding_projection_to_item_embedding_dim_layers = (
            nn.ModuleDict()
        )
        self.features_layer_norm = nn.ModuleDict()

        self.input_combined_dim = 0


        # Init embedding tables
        for cname, cinfo in self.feature_map.items():
            # Define embedding table of categorical feature 
            if cinfo["dtype"] == "categorical":
                if self.use_ohe_item_ids_inputs:
                    feature_size = cinfo["cardinality"]
                else:
                    if "is_itemid" in cinfo and cinfo["is_itemid"]:
                        # Set itemid embeddings table
                        if model_args.item_embedding_dim is not None:
                            embedding_size = model_args.item_embedding_dim
                        # When using tying embeddings, items embeddings dimension is equal to the Transformer model output dimension
                        elif model_args.mf_constrained_embeddings:
                            embedding_size = model_args.d_model
                        else:
                            # Compute embedding dimension from the feature cardinality 
                            embedding_size = get_embedding_size_from_cardinality(
                                cinfo["cardinality"],
                                multiplier=self.embedding_dim_from_cardinality_multiplier,
                            )
                        
                        feature_size = embedding_size
                        self.item_embedding_dim = embedding_size

                        # Set itemid feature as prediction labels for item prediction task.
                        if "is_label" in cinfo and cinfo['is_label']: 
                            self.label_feature_name = cname
                            self.label_embedding_table_name = cinfo["emb_table"]
                            self.label_embedding_dim = embedding_size
                    
                    elif "is_classification_label" in cinfo and cinfo["is_classification_label"]:
                        # Get num classes for classification head  
                        self.num_classes =  cinfo["cardinality"]

                    else:
                        # Set embeddings table of other categorical features 
                        if self.features_same_size_item_embedding:
                            if self.item_embedding_dim:
                                embedding_size = self.item_embedding_dim
                                feature_size = embedding_size
                            else:
                                raise ValueError(
                                    "Make sure that the item id is the first in the YAML features config file."
                                )
                        else:
                            embedding_size = get_embedding_size_from_cardinality(
                                cinfo["cardinality"],
                                multiplier=self.embedding_dim_from_cardinality_multiplier,
                            )
                            feature_size = embedding_size

                            if (
                                self.input_features_aggregation
                                == "elementwise_sum_multiply_item_embedding"
                            ):
                                self.features_embedding_projection_to_item_embedding_dim_layers[
                                    cname
                                ] = nn.Linear(
                                    embedding_size, self.label_embedding_dim, bias=True,
                                )
                                feature_size = self.label_embedding_dim


                    self.embedding_tables[cinfo["emb_table"]] = nn.Embedding(
                        cinfo["cardinality"],
                        embedding_size,
                        padding_idx=self.pad_token,
                    )

                    # Added to initialize embeddings
                    if "is_itemid" in cinfo and cinfo["is_itemid"]:
                        embedding_init_std = self.item_id_embeddings_init_std
                    else:
                        embedding_init_std = self.other_embeddings_init_std

                    with torch.no_grad():
                        self.embedding_tables[cinfo["emb_table"]].weight.normal_(
                            0.0, embedding_init_std
                        )

                logger.info(
                    "Categ Feature: {} - Cardinality: {} - Feature Size: {}".format(
                        cname, cinfo["cardinality"], feature_size
                    )
                )
                
            # Define deep representation of continuous feature 
            elif cinfo["dtype"] in ["long", "float"]:
                if self.numeric_features_project_to_embedding_dim > 0:
                    if self.features_same_size_item_embedding:
                        if self.label_embedding_dim:
                            feature_size = self.label_embedding_dim
                        else:
                            raise ValueError(
                                "Make sure that the item id (label feature) is the first in the YAML features config file."
                            )
                    else:
                        feature_size = self.numeric_features_project_to_embedding_dim

                    if self.numeric_features_soft_one_hot_encoding_num_embeddings > 0:
                        self.numeric_soft_embeddings[cname] = SoftEmbedding(
                            num_embeddings=self.numeric_features_soft_one_hot_encoding_num_embeddings,
                            embeddings_dim=feature_size,
                            embeddings_init_std=self.other_embeddings_init_std,
                        )

                        if (
                            self.input_features_aggregation
                            == "elementwise_sum_multiply_item_embedding"
                        ):
                            self.features_embedding_projection_to_item_embedding_dim_layers[
                                cname
                            ] = nn.Linear(
                                feature_size, self.label_embedding_dim, bias=True,
                            )
                            feature_size = self.label_embedding_dim
                    else:
                        if (
                            self.input_features_aggregation
                            == "elementwise_sum_multiply_item_embedding"
                        ):
                            project_scalar_to_embedding_dim = self.label_embedding_dim
                        else:
                            project_scalar_to_embedding_dim = (
                                self.numeric_features_project_to_embedding_dim
                            )
                        feature_size = project_scalar_to_embedding_dim

                        self.numeric_to_embedding_layers[cname] = nn.Linear(
                            1, project_scalar_to_embedding_dim, bias=True
                        )

                else:
                    feature_size = 1

                logger.info(
                    "Numerical Feature: {} - Feature Size: {}".format(
                        cname, feature_size
                    )
                )

            elif cinfo["is_control"]:
                # Control features are not used as input for the model
                continue
            else:
                raise NotImplementedError

            self.input_combined_dim += feature_size

            if self.layer_norm_featurewise:
                self.features_layer_norm[cname] = nn.LayerNorm(
                    normalized_shape=feature_size
                )

            if "is_label" in cinfo and cinfo["is_label"]:
                self.target_dim = cinfo["cardinality"]
        
        if self.input_features_aggregation == "elementwise_sum_multiply_item_embedding":
            self.input_combined_dim = self.item_embedding_dim

        if self.target_dim == None and self.num_classes == None:
            raise RuntimeError("label column is not declared in feature map.")


    def forward(self, inputs):
        label_seq, output = None, []
        metadata_for_pred_logging = {}
        classification_labels = None

        transformed_features = OrderedDict()
        for cname, cinfo in self.feature_map.items():

            cdata = inputs[cname]

            if "is_label" in cinfo and cinfo["is_label"]:
                label_seq = cdata

            if cinfo["dtype"] == "categorical":
                cdata = cdata.long()

                # Applies Stochastic Shared Embeddings if training
                if (
                    self.stochastic_shared_embeddings_replacement_prob > 0.0
                    and not self.use_ohe_item_ids_inputs
                    and self.training
                ):
                    with torch.no_grad():
                        cdata_non_zero_mask = cdata != self.pad_token

                        sse_prob_replacement_matrix = torch.full(
                            cdata.shape,
                            self.stochastic_shared_embeddings_replacement_prob,
                            device=self.device,
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

                if "is_label" in cinfo and cinfo["is_label"]:
                    if self.use_ohe_item_ids_inputs:
                        cdata = torch.nn.functional.one_hot(
                            cdata, num_classes=self.target_dim
                        ).float()
                    else:
                        cdata = self.embedding_tables[cinfo["emb_table"]](cdata)

                elif  "is_classification_label" in cinfo and cinfo["is_classification_label"]:
                    # keep the original tensor of classification target  
                    classification_labels = cdata 

                else:
                    cdata = self.embedding_tables[cinfo["emb_table"]](cdata)

                    if (
                        self.input_features_aggregation
                        == "elementwise_sum_multiply_item_embedding"
                        and not self.features_same_size_item_embedding
                    ):
                        cdata = self.features_embedding_projection_to_item_embedding_dim_layers[
                            cname
                        ](
                            cdata
                        )

            elif cinfo["dtype"] in ["long", "float"]:
                if cinfo["dtype"] == "long":
                    cdata = cdata.unsqueeze(-1).long()
                elif cinfo["dtype"] == "float":
                    cdata = cdata.unsqueeze(-1).float()

                if self.numeric_features_project_to_embedding_dim > 0:
                    if self.numeric_features_soft_one_hot_encoding_num_embeddings > 0:
                        cdata = self.numeric_soft_embeddings[cname](cdata)

                        if (
                            self.input_features_aggregation
                            == "elementwise_sum_multiply_item_embedding"
                            and not self.features_same_size_item_embedding
                        ):
                            cdata = self.features_embedding_projection_to_item_embedding_dim_layers[
                                cname
                            ](
                                cdata
                            )
                    else:
                        cdata = self.numeric_to_embedding_layers[cname](cdata)

            elif cinfo["is_control"]:
                # Control features are not used as input for the model
                continue
            else:
                raise NotImplementedError

            # Applying layer norm for each feature
            if self.layer_norm_featurewise:
                cdata = self.features_layer_norm[cname](cdata)

            transformed_features[cname] = cdata

            # Keeping item metadata features
            if (
                "log_with_preds_as_metadata" in cinfo
                and cinfo["log_with_preds_as_metadata"] == True
            ):
                metadata_for_pred_logging[cname] = inputs[cname].detach()



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
                    if k != self.label_feature_name:
                        if (self.feature_map[k]["dtype"] == "categorical") or (
                            self.feature_map[k]["dtype"] in ["long", "float"]
                            and self.numeric_features_project_to_embedding_dim > 0
                        ):
                            additional_features_sum += v

                item_id_embedding = transformed_features[self.label_feature_name]

                output = item_id_embedding * (additional_features_sum + 1.0)
            else:
                raise ValueError("Invalid value for --input_features_aggregation.")
        else:
            output = list(transformed_features.values())[0]

        return output, label_seq, classification_labels, metadata_for_pred_logging