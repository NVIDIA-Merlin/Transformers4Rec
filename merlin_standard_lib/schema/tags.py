from enum import Enum


class Tags(Enum):
    # Feature types
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    LIST = "list"
    TEXT = "text"
    TEXT_TOKENIZED = "text_tokenized"
    TIME = "time"

    # Feature context
    USER = "user"
    ITEM = "item"
    ITEM_ID = "item_id"
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
