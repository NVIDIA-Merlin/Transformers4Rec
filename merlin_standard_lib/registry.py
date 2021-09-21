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

import abc
import re

# Camel case to snake case utils
from typing import Generic, TypeVar

_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


def camelcase_to_snakecase(name):
    s1 = _first_cap_re.sub(r"\1_\2", name)
    return _all_cap_re.sub(r"\1_\2", s1).lower()


def snakecase_to_camelcase(name):
    return "".join([w[0].upper() + w[1:] for w in name.split("_")])


def default_name(class_or_fn):
    """Default name for a class or function.

    This is the naming function by default for registries expecting classes or
    functions.

    Parameters
    ----------
    class_or_fn:
        class or function to be named.

    Returns
    -------
    Default name for registration.
    """
    return camelcase_to_snakecase(class_or_fn.__name__)


def default_object_name(obj):
    return default_name(type(obj))


class Registry:
    """
    Dict-like class for managing function registrations.

    Example usage::

        my_registry = Registry("custom_name")
        @my_registry.register
        def my_func():
          pass
        @my_registry.register()
        def another_func():
          pass
        @my_registry.register("non_default_name")
        def third_func(x, y, z):
          pass
        def foo():
          pass
        my_registry.register()(foo)
        my_registry.register("baz")(lambda (x, y): x + y)
        my_register.register("bar")
        print(list(my_registry))
        # ["my_func", "another_func", "non_default_name", "foo", "baz"]
        # (order may vary)
        print(my_registry["non_default_name"] is third_func)  # True
        print("third_func" in my_registry)                    # False
        print("bar" in my_registry)                           # False
        my_registry["non-existent_key"]                       # raises KeyError

    Optional validation, on_set callback and value transform also supported.

    Parameters
    ----------
    registry_name: str
        identifier for the given registry. Used in error msgs.
    default_key_fn: callable, optional
        function mapping value -> key for registration when a key is not provided
    validator: callable, optional
        if given, this is run before setting a given (key, value) pair. Accepts (key, value)
        and should raise if there is a problem. Overwriting existing keys is not allowed and is
        checked separately. Values are also checked to be callable separately.
    on_set: callable, optional
        callback function accepting (key, value) pair which is run after an item is successfully
        set.
    value_transformer: callable, optional
        if run, `__getitem__` will return value_transformer(key, registered_value).
    """

    def __init__(
        self,
        registry_name,
        default_key_fn=default_name,
        validator=None,
        on_set=None,
        value_transformer=(lambda k, v: v),
    ):
        self._registry = {}
        self._name = registry_name
        self._default_key_fn = default_key_fn
        self._validator = validator
        self._on_set = on_set
        self._value_transformer = value_transformer

    @classmethod
    def class_registry(
        cls, registry_name, default_key_fn=default_name, validator=None, on_set=None
    ):
        return cls(
            registry_name=registry_name,
            default_key_fn=default_key_fn,
            validator=validator,
            on_set=on_set,
            value_transformer=(lambda k, v: v()),
        )

    def default_key(self, value):
        """Default key used when key not provided. Uses function from __init__."""
        return self._default_key_fn(value)

    @property
    def name(self):
        return self._name

    def validate(self, key, value):
        """Validation function run before setting. Uses function from __init__."""
        if self._validator is not None:
            self._validator(key, value)

    def on_set(self, key, value):
        """Callback called on successful set. Uses function from __init__."""
        if self._on_set is not None:
            self._on_set(key, value)

    def __setitem__(self, key, value):
        """Validate, set, and (if successful) call `on_set` for the given item.

        Parameters
        ----------
        key:
            key to store value under. If `None`, `self.default_key(value)` is used.
        value:
            callable stored under the given key.
        Raises
        ------
            KeyError: if key is already in registry.
        """
        if key is None:
            key = self.default_key(value)
        if not isinstance(key, tuple):
            key = (key,)

        for k in key:
            if k in self:
                raise KeyError("key %s already registered in registry %s" % (k, self._name))
            if not callable(value):
                raise ValueError("value must be callable")
            self.validate(k, value)
            self._registry[k] = value
            self.on_set(k, value)

    def register(self, key_or_value=None):
        """Decorator to register a function, or registration itself.
        This is primarily intended for use as a decorator, either with or without
        a key/parentheses.

        Example Usage::

            @my_registry.register('key1')
            def value_fn(x, y, z):
              pass
            @my_registry.register()
            def another_fn(x, y):
              pass
            @my_registry.register
            def third_func():
              pass

        Note if key_or_value is provided as a non-callable, registration only
        occurs once the returned callback is called with a callable as its only
        argument::

            callback = my_registry.register('different_key')
            'different_key' in my_registry  # False
            callback(lambda (x, y): x + y)
            'different_key' in my_registry  # True

        Parameters
        ----------
        key_or_value (optional):
            key to access the registered value with, or the unction itself. If `None` (default),
            `self.default_key` will be called on `value` once the returned callback is called with
            `value` as the only arg. If `key_or_value` is itself callable, it is assumed to be the
            value and the key is given by `self.default_key(key)`.

        Returns
        -------
        decorated callback, or callback generated a decorated function.
        """

        def decorator(value, key):
            self[key] = value
            return value

        # Handle if decorator was used without parens
        if callable(key_or_value):
            return decorator(value=key_or_value, key=None)
        else:
            return lambda value: decorator(value, key=key_or_value)

    def register_with_multiple_names(self, *names):
        return self.register(names)

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(
                "%s never registered with registry %s. Available:\n %s"
                % (key, self.name, display_list_by_prefix(sorted(self), 4))
            )
        value = self._registry[key]
        return self._value_transformer(key, value)

    def __contains__(self, key):
        return key in self._registry

    def keys(self):
        return self._registry.keys()

    def values(self):
        return (self[k] for k in self)  # complicated because of transformer

    def items(self):
        return ((k, self[k]) for k in self)  # complicated because of transformer

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def _clear(self):
        self._registry.clear()

    def get(self, key, default=None):
        return self[key] if key in self else default

    def parse(self, class_or_str):
        if isinstance(class_or_str, str):
            return self[class_or_str]

        return class_or_str


RegistryClassT = TypeVar("RegistryClassT")


class RegistryMixin(Generic[RegistryClassT], abc.ABC):
    @classmethod
    def parse(cls, class_or_str) -> RegistryClassT:
        output: RegistryClassT = cls.registry().parse(class_or_str)

        return output

    @classmethod
    def registry(cls) -> Registry:
        raise NotImplementedError


def display_list_by_prefix(names_list, starting_spaces=0):
    """Creates a help string for names_list grouped by prefix."""
    cur_prefix, result_lines = None, []
    space = " " * starting_spaces
    for name in sorted(names_list):
        split = name.split("_", 1)
        prefix = split[0]
        if cur_prefix != prefix:
            result_lines.append(space + prefix + ":")
            cur_prefix = prefix
        result_lines.append(space + "  * " + name)
    return "\n".join(result_lines)
