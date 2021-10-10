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
import json
import os
from typing import TypeVar

import betterproto
from betterproto import Message as BetterProtoMessage
from google.protobuf import json_format, text_format
from google.protobuf.message import Message as ProtoMessage

ProtoMessageType = TypeVar("ProtoMessageType", bound=BetterProtoMessage)


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
        with open(path_proto_text, "r") as f:
            proto_text = f.read()

    proto = text_format.Parse(proto_text, message)

    # This is a hack because as of now we can't parse the Any representation.
    # TODO: Fix this.
    d = json_format.MessageToDict(proto)
    for f in d["feature"]:
        if "extraMetadata" in f["annotation"]:  # type: ignore
            extra_metadata = f["annotation"].pop("extraMetadata")  # type: ignore
            f["annotation"]["comment"] = [json.dumps(extra_metadata[0]["value"])]  # type: ignore
    json_str = json_format.MessageToJson(json_format.ParseDict(d, message))

    return better_proto_message.__class__().from_json(json_str)
