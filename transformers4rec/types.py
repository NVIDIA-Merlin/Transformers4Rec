try:
    import nvtabular as nvt
    from nvtabular.tag import DefaultTags, Tag

    ColumnGroup = nvt.ColumnGroup
except ImportError:
    from .utils.columns import ColumnGroup
    from .utils.tags import DefaultTags, Tag

__all__ = ["ColumnGroup", "Tag", "DefaultTags"]
