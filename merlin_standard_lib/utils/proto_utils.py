import os
from typing import TypeVar

import betterproto
from betterproto import Message as BetterProtoMessage
from google.protobuf import text_format
from google.protobuf.message import Message as ProtoMessage

ProtoMessageType = TypeVar("ProtoMessageType")


def has_field(self, field_name):
    return betterproto.serialized_on_wire(getattr(self, field_name))


def copy_better_proto_message(better_proto_message: ProtoMessageType, **kwargs) -> ProtoMessageType:
    output = better_proto_message.__class__().parse(bytes(better_proto_message))
    for key, val in kwargs.items():
        setattr(output, key, val)

    return output


def better_proto_to_proto_text(
    better_proto_message: BetterProtoMessage, message: ProtoMessage
) -> str:
    message.ParseFromString(bytes(better_proto_message))

    return text_format.MessageToString(message)


def proto_text_to_better_proto(
    better_proto_message: ProtoMessageType, path_proto_text: str, message: ProtoMessage
) -> ProtoMessageType:
    proto_text = path_proto_text
    if os.path.isfile(proto_text):
        with open(path_proto_text, "rb") as f:
            proto_text = f.read()

    proto = text_format.Parse(proto_text, message)

    return better_proto_message.__class__().parse(proto.SerializeToString())
