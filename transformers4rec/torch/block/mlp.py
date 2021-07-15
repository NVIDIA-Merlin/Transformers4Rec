import types

import torch

from .base import BuildableBlock, SequentialBlock


class MLPBlock(BuildableBlock):
    def __init__(
        self,
        dimensions,
        activation=torch.nn.ReLU,
        use_bias: bool = True,
        dropout=None,
        normalization=None,
        filter_features=None,
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.dropout = dropout
        self.filter_features = filter_features
        self.use_bias = use_bias
        self.activation = activation
        self.dimensions = dimensions

    def build(self, input_shape) -> SequentialBlock:
        layer_input_sizes = list(input_shape[-1:]) + list(self.dimensions[:-1])
        layer_output_sizes = self.dimensions
        sequential = [
            self._create_layer(input_size, output_size)
            for input_size, output_size in zip(layer_input_sizes, layer_output_sizes)
        ]

        return SequentialBlock(*sequential)

    def _create_layer(self, input_size, output_size):
        out = [torch.nn.Linear(input_size, output_size, bias=self.use_bias)]
        if self.activation:
            out.append(self.activation(inplace=True))
        if self.normalization:
            if self.normalization == "batch_norm":
                out.append(torch.nn.BatchNorm1d(output_size))
        if self.dropout:
            out.append(torch.nn.Dropout(self.dropout))

        output = torch.nn.Sequential(*out)

        def _get_name(self):
            return "DenseBlock"

        output._get_name = types.MethodType(_get_name, output)

        return output
