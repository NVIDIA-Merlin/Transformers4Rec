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
def is_gpu_dataloader_available() -> bool:
    try:
        import cudf
        import cupy
    except ImportError:
        cudf = None
        cupy = None
    return cudf is not None and cupy is not None


def is_pyarrow_available() -> bool:
    try:
        import pyarrow
    except ImportError:
        pyarrow = None
    return pyarrow is not None


def is_merlin_dataloader_available() -> bool:
    try:
        import merlin.dataloader
    except ImportError:
        merlin.dataloader = None
    return merlin.dataloader is not None
