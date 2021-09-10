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

from .config.transformer import (
    AlbertConfig,
    ElectraConfig,
    GPT2Config,
    LongformerConfig,
    ReformerConfig,
    T4RecConfig,
    XLNetConfig,
)

# TODO check for NVTabular, and if it's installed import these from there
__all__ = [
    "T4RecConfig",
    "GPT2Config",
    "XLNetConfig",
    "LongformerConfig",
    "AlbertConfig",
    "ReformerConfig",
    "ElectraConfig",
]


try:
    from . import tf as tensorflow

    tf = tensorflow

    __all__.append("tf")
except ImportError:
    pass

try:
    from . import torch as t

    torch = t

    __all__.append("torch")
except ImportError:
    pass
