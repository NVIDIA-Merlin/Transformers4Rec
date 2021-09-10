from betterproto import Message

from .registry import Registry, RegistryMixin
from .schema import schema
from .schema.schema import ColumnSchema, Schema
from .schema.tags import Tags
from .utils import proto_utils

# Other monkey-patching
Message.HasField = proto_utils.has_field
Message.copy = proto_utils.copy_better_proto_message

__all__ = ["ColumnSchema", "Schema", "schema", "Tags", "Registry", "RegistryMixin"]
