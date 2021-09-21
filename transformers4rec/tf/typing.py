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

import typing
from typing import Dict

import tensorflow as tf

# TODO: Make this more generic and work with multi-hot features
TabularData = Dict[str, tf.Tensor]
TensorOrTabularData = typing.Union[tf.Tensor, TabularData]
LossReduction = typing.Callable[[typing.List[tf.Tensor]], tf.Tensor]

__all__ = [
    "TabularData",
    "TensorOrTabularData",
    "LossReduction",
]
