try:
    import nvtabular as nvt
    from nvtabular.tag import Tag

    ColumnGroup = nvt.ColumnGroup
except ImportError:
    from .utils.columns import ColumnGroup
    from .utils.tags import Tag

__all__ = ["ColumnGroup", "Tag"]
