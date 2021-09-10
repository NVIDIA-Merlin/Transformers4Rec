from betterproto import Message

from .registry import Registry, RegistryMixin
from .schema import schema
from .schema.schema import ColumnSchema, Schema
from .schema.tags import Tags
from .utils import proto_utils

# Monkey-path proto-text integration
Message.to_proto_text = proto_utils.better_proto_to_proto_text
Message.from_proto_text = proto_utils.proto_text_to_better_proto

# Other monkey-patching
Message.HasField = proto_utils.has_field
Message.copy = proto_utils.copy_better_proto_message

__all__ = ["ColumnSchema", "Schema", "schema", "Tags", "Registry", "RegistryMixin"]
