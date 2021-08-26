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
            DenseBlock(
                input_shape,
                input_size,
                output_size,
                activation=self.activation,
                use_bias=self.use_bias,
                dropout=self.dropout,
                normalization=self.normalization,
            )
            for input_size, output_size in zip(layer_input_sizes, layer_output_sizes)
        ]

        output = SequentialBlock(*sequential)
        output.input_size = input_shape

        return output


class DenseBlock(SequentialBlock):
    def __init__(
        self,
        input_shape,
        in_features: int,
        out_features: int,
        activation=torch.nn.ReLU,
        use_bias: bool = True,
        dropout=None,
        normalization=None,
    ):
        args = [torch.nn.Linear(in_features, out_features, bias=use_bias)]
        if activation:
            args.append(activation(inplace=True))
        if normalization:
            if normalization == "batch_norm":
                args.append(torch.nn.BatchNorm1d(out_features))
        if dropout:
            args.append(torch.nn.Dropout(self.dropout))

        super().__init__(*args)
        self._input_shape = input_shape
        self._output_size = out_features

    def _get_name(self):
        return "DenseBlock"

    def output_size(self):
        if len(self._input_shape) == 3:
            return torch.Size([self._input_shape[0], self._input_shape[1], self._output_size])

        return torch.Size([self._input_shape[0], self._output_size])

    def forward_output_size(self, input_size):
        if len(input_size) == 3:
            return torch.Size([input_size[0], input_size[1], self._output_size])

        return torch.Size([input_size[0], self._output_size])
