import torch
from torch import nn


class ConcatFeatures(nn.Module):
    def __init__(self, axis=-1):
        super(ConcatFeatures, self).__init__()
        self.axis = axis

    def forward(self, inputs):
        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(inputs[name])
        return torch.cat(tensors, dim=self.axis)


class StackFeature(nn.Module):
    def __init__(self, axis=-1):
        super(StackFeature, self).__init__()
        self.axis = axis

    def forward(self, inputs):
        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(inputs[name])
        return torch.stack(tensors, dim=self.axis)


class ElementwiseSum(nn.Module):
    def __init__(self):
        super(ElementwiseSum, self).__init__()

    def forward(self, inputs):
        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(inputs[name])

        return torch.stack(tensors, dim=0).sum(dim=0)


class SequenceAggregator(nn.Module):
    """
    Receive a dictionary of sequential tensors and output their aggregation as a 3d tensor.
    It supports two types of aggregation: concat and elementwise_sum_multiply_item_embedding
    """

    def __init__(
        self,
        input_features_aggregation,
        itemid_name,
        feature_map,
        numeric_features_project_to_embedding_dim,
        device,
    ):
        super(SequenceAggregator, self).__init__()
        self.itemid_name = itemid_name
        self.feature_map = feature_map
        self.device = device
        self.input_features_aggregation = input_features_aggregation
        self.numeric_features_project_to_embedding_dim = numeric_features_project_to_embedding_dim
        if input_features_aggregation == "concat":
            self.aggegator = ConcatFeatures()
        elif self.input_features_aggregation == "elementwise_sum_multiply_item_embedding":
            self.aggregator = ElementwiseSum()
            # features to sum are all categorical and projected continuous embeddings exluding itemid
            self.other_features = [
                k
                for k in feature_map.keys()
                if (self.feature_map[k]["dtype"] == "categorical")
                or (
                    self.feature_map[k]["dtype"] in ["long", "float"]
                    and self.numeric_features_project_to_embedding_dim > 0
                )
            ]

    def forward(self, transformed_features):
        if len(transformed_features) > 1:
            if self.input_features_aggregation == "concat":
                output = self.aggegator(transformed_features)
            elif self.input_features_aggregation == "elementwise_sum_multiply_item_embedding":
                additional_features_sum = {
                    k: v.long() for k, v in transformed_features.items() if k in self.other_features
                }

                item_id_embedding = transformed_features[self.itemid_name]

                output = item_id_embedding * (self.aggregator(additional_features_sum) + 1.0)
            else:
                raise ValueError("Invalid value for --input_features_aggregation.")
        else:
            output = list(transformed_features.values())[0]
        return output
