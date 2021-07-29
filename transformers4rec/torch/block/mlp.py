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

        if isinstance(dimensions, int):
            dimensions = [dimensions]

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
            self._create_layer(input_size, output_size, input_shape)
            for input_size, output_size in zip(layer_input_sizes, layer_output_sizes)
        ]

        output = SequentialBlock(*sequential)
        output.input_size = input_shape

        return output

    def _create_layer(self, dense_input_size, dense_output_size, input_shape):
        out = [torch.nn.Linear(dense_input_size, dense_output_size, bias=self.use_bias)]
        if self.activation:
            out.append(self.activation(inplace=True))
        if self.normalization:
            if self.normalization == "batch_norm":
                out.append(torch.nn.BatchNorm1d(dense_output_size))
        if self.dropout:
            out.append(torch.nn.Dropout(self.dropout))

        output = torch.nn.Sequential(*out)

        def _get_name(self):
            return "DenseBlock"

        def output_size(self):
            if len(input_shape) == 3:
                return torch.Size([input_shape[0], input_shape[1], dense_output_size])

            return torch.Size([input_shape[0], dense_output_size])

        def forward_output_size(self, input_size):
            if len(input_size) == 3:
                return torch.Size([input_size[0], input_size[1], dense_output_size])

            return torch.Size([input_size[0], dense_output_size])

        output._get_name = types.MethodType(_get_name, output)
        output.output_size = types.MethodType(output_size, output)
        output.forward_output_size = types.MethodType(forward_output_size, output)

        return output
