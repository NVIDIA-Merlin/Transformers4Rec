import warnings

from merlin.schema import Tags

Tag = Tags


warnings.warn(
    "The `merlin_standard_lib.schema.tag` module has moved to `merlin.schema`. "
    "Support for importing from `merlin_standard_lib.schema.tag` is deprecated, "
    "and will be removed in a future version. Please update "
    "your imports to refer to `merlin.schema`.",
    DeprecationWarning,
)


__all__ = ["Tag"]
