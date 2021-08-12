try:
    from nvtabular import Schema
    from nvtabular.tag import DefaultTags, Tag
except ImportError:
    from .utils.schema import Schema
    from .utils.tags import DefaultTags, Tag

__all__ = ["Schema", "Tag", "DefaultTags"]
