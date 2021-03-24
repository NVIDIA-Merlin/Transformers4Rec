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
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Mapping, Optional, Sequence, TypeVar, Union

import numpy as np

ItemId = TypeVar("ItemId", str, int)


class InstanceInfoLevel(Enum):
    INTERACTION = "interaction"
    SESSION = "session"


class FeaturesDataType(Enum):
    INT = "int32"
    LONG = "int64"
    FLOAT = "float16"
    DOUBLE = "float32"
    STR = "str"
    BOOL = "bool"

    @classmethod
    def to_numpy_dtype(cls, data_type) -> np.dtype:
        switcher = {
            cls.INT: np.int32,
            cls.LONG: np.int64,
            cls.FLOAT: np.float16,
            cls.DOUBLE: np.float32,
            cls.STR: np.str,
            cls.BOOL: np.bool,
        }

        if data_type not in switcher:
            raise Exception("Invalid data type: {}".format(data_type))

        return switcher[data_type]


class FeatureGroupType(Enum):
    ITEM_ID = "item_id"
    EVENT_TS = "event_timestamp"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    IMPLICIT_FEEDBACK = "implicit_feedback"
    ITEM_METADATA = "item_metadata"
    USER_METADATA = "user_metadata"
    EVENT_METADATA = "event_metadata"
    SEQUENTIAL_FEATURES = "sequential_features"


@dataclass
class FeatureGroups:
    item_id: str
    event_timestamp: str
    user_id: Optional[str]
    session_id: Optional[str]
    implicit_feedback: Optional[str]
    item_metadata: List[str] = field(default_factory=list)
    user_metadata: List[str] = field(default_factory=list)
    event_metadata: List[str] = field(default_factory=list)
    sequential_features: List[str] = field(default_factory=list)

    def get_feature_group(self, group_type: FeatureGroupType) -> Union[str, List[str]]:
        if not hasattr(self, group_type.value):
            raise Exception("Invalid feature group: {}".format(group_type.value))
        return getattr(self, group_type.value)


@dataclass
class FeatureTypes:
    categorical: Sequence[str] = field(default_factory=list)
    numerical: Sequence[str] = field(default_factory=list)


@dataclass
class InputDataConfig:
    schema: Mapping[str, FeaturesDataType]
    feature_groups: FeatureGroups
    feature_types: FeatureTypes
    instance_info_level: InstanceInfoLevel
    session_padded_items_value: int = field(default=0)
    positive_interactions_only: bool = field(default=False)

    def get_item_feature_names(self) -> List[str]:
        return self.feature_groups.item_metadata + [self.feature_groups.item_id]

    def get_feature_dtype(self, fname: str) -> FeaturesDataType:
        return self.schema[fname]

    def get_feature_numpy_dtype(self, fname: str) -> np.dtype:
        return FeaturesDataType.to_numpy_dtype(self.schema[fname])

    def get_feature_group(self, group_type: FeatureGroupType) -> Union[str, List[str]]:
        return self.feature_groups.get_feature_group(group_type)

    def get_features_from_type(self, feature_type: str) -> Sequence[str]:
        return getattr(self.feature_types, feature_type)
