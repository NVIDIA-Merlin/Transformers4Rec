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

import warnings

from merlin.schema import Tags

Tag = Tags


warnings.warn(
    "The `merlin_standard_lib.schema.tag` module has moved to `merlin.schema`. "
    "Support for importing from `merlin_standard_lib.schema.tag` is deprecated, "
    "and will be removed in a future version. Please update "
    "your imports to refer to `merlin.schema`.",
    DeprecationWarning,
)


__all__ = ["Tag"]
