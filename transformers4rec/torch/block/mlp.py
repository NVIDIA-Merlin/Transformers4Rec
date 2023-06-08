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
from typing import List, Optional, Union

import torch

from .base import BuildableBlock, SequentialBlock


class MLPBlock(BuildableBlock):
    """Defines Multi-Layer Perceptron (MLP) Block by stacking
    multiple DenseBlock instances.

    Parameters
    ----------
    dimensions : int or list of int
        The dimensions of the layers in the MLP.
        If an integer is provided, a single layer MLP is created.
        If a list is provided, it must contain the size of each layer in order.
    activation : optional
        The activation function to apply after each layer.
        By default `torch.nn.ReLU`.
    use_bias : bool, optional
        Whether to add a bias term to the dense layers.
        by default True
    dropout : float, optional
        The dropout rate to apply after each layer, by default None
    normalization : str, optional
        The normalization to apply after each layer, by default None
    filter_features : List[str], optional
        List of features to select from the input., by default None
    """

    def __init__(
        self,
        dimensions,
        activation=torch.nn.ReLU,
        use_bias: bool = True,
        dropout: float = None,
        normalization: str = None,
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
    """
    A buildable dense Block to represent a fully connected layer.

    Parameters
    ----------
    input_shape : Union[List[int], torch.Size]
        The shape of the input tensor.

    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    activation : torch.nn.Module, optional
        The activation function to apply after the linear layer.
        By default `torch.nn.ReLU`.

    use_bias : bool, optional
        Whether to use bias in the layer.
        By default True.

    dropout : float, optional
        The dropout rate to apply after the dense layer, if any.
        By default is None.

    normalization : str, optional
        The type of normalization to apply after the dense layer.
        Only 'batch_norm' is supported.
        By default is None.
    """

    def __init__(
        self,
        input_shape: Union[List[int], torch.Size],
        in_features: int,
        out_features: int,
        activation=torch.nn.ReLU,
        use_bias: bool = True,
        dropout: Optional[float] = None,
        normalization=None,
    ):
        args: List[torch.nn.Module] = [torch.nn.Linear(in_features, out_features, bias=use_bias)]
        if activation:
            args.append(activation(inplace=True))
        if normalization:
            if normalization == "batch_norm":
                args.append(torch.nn.BatchNorm1d(out_features))
        if dropout:
            args.append(torch.nn.Dropout(dropout))

        super().__init__(*args)
        self._input_shape = input_shape
        self._output_size = out_features

    def _get_name(self):
        return "DenseBlock"

    def forward_output_size(self, input_size):
        return torch.Size(list(input_size[:-1]) + [self._output_size])
