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


from betterproto import Message

from .registry import Registry, RegistryMixin
from .schema import schema
from .schema.schema import ColumnSchema, Schema
from .schema.tag import Tag
from .utils import proto_utils

# Other monkey-patching
Message.HasField = proto_utils.has_field  # type: ignore
Message.copy = proto_utils.copy_better_proto_message  # type: ignore

__all__ = ["ColumnSchema", "Schema", "schema", "Tag", "Registry", "RegistryMixin"]
