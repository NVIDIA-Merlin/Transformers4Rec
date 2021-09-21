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


from enum import Enum
from typing import List, Union


class Tag(Enum):
    # Feature types
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    LIST = "list"
    TEXT = "text"
    TEXT_TOKENIZED = "text_tokenized"
    TIME = "time"

    # Feature context
    USER = "user"
    USER_ID = "user_id"
    ITEM = "item"
    ITEM_ID = "item_id"
    SESSION = "session"
    SESSION_ID = "session_id"
    CONTEXT = "context"

    # Target related
    TARGETS = "target"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class"
    REGRESSION = "regression"

    def __str__(self):
        return self.value

    def __eq__(self, o: object) -> bool:
        return str(o) == str(self)


TagsType = Union[List[str], List[Tag], List[Union[Tag, str]]]
