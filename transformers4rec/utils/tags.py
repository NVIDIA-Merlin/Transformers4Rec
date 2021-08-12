from enum import Enum


class DefaultTags(Enum):
    # Feature types
    CATEGORICAL = ["categorical"]
    CONTINUOUS = ["continuous"]
    LIST = ["list"]
    IMAGE = ["image"]
    TEXT = ["text"]
    TEXT_TOKENIZED = ["text_tokenized"]
    TIME = ["time"]

    # Feature context
    USER = ["user"]
    ITEM = ["item"]
    ITEM_ID = ["item", "item_id"]
    CONTEXT = ["context"]

    # Target related
    TARGETS = ["target"]
    TARGETS_BINARY = ["target", "binary"]
    TARGETS_REGRESSION = ["target", "regression"]
    TARGETS_MULTI_CLASS = ["target", "multi_class"]


class Tag:
    CATEGORICAL = DefaultTags.CATEGORICAL
    CONTINUOUS = DefaultTags.CONTINUOUS
    LIST = DefaultTags.LIST
    IMAGE = DefaultTags.IMAGE
    TEXT = DefaultTags.TEXT
    TEXT_TOKENIZED = DefaultTags.TEXT_TOKENIZED
    TIME = DefaultTags.TIME

    # Feature context
    USER = DefaultTags.USER
    ITEM = DefaultTags.ITEM
    ITEM_ID = DefaultTags.ITEM_ID
    CONTEXT = DefaultTags.CONTEXT

    # Target related
    TARGETS = DefaultTags.TARGETS
    TARGETS_BINARY = DefaultTags.TARGETS_BINARY
    TARGETS_REGRESSION = DefaultTags.TARGETS_REGRESSION
    TARGETS_MULTI_CLASS = DefaultTags.TARGETS_MULTI_CLASS

    def __init__(self, *tag):
        self.tags = tag

    @classmethod
    def parse(cls, tag, allow_list=True):
        if allow_list and isinstance(tag, list):
            return Tag(tag)
        elif isinstance(tag, DefaultTags):
            return Tag(tag.value)
        elif isinstance(tag, Tag):
            return tag
