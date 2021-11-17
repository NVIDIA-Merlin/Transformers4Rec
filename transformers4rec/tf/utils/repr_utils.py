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

from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def dict_wrapper_repr(self):
    child_lines = []

    for key, child in dict(self).items():
        if isinstance(child, FeatureConfig):
            child = child.table
        mod_str = repr(child)
        mod_str = _addindent(mod_str, 2)
        child_lines.append("(" + key + "): " + mod_str)

    main_str = "Dict("
    if child_lines:
        main_str += "\n  " + "\n  ".join(child_lines) + "\n"

    main_str += ")"

    return main_str


def list_wrapper_repr(self):
    child_lines = []

    for index, item in enumerate(self):
        mod_str = repr(item)
        mod_str = _addindent(mod_str, 2)
        child_lines.append("(" + str(index) + "): " + mod_str)

    main_str = "List("
    if child_lines:
        main_str += "\n  " + "\n  ".join(child_lines) + "\n"

    main_str += ")"

    return main_str


def _layer_repr(self, track_children=True):
    extra_lines = []
    extra_repr = self.repr_extra() if getattr(self, "repr_extra", None) else None
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []

    if track_children:
        to_remove = self.repr_ignore() if getattr(self, "repr_ignore", None) else []
        children = [
            x for x in self._self_unconditional_checkpoint_dependencies if x.name not in to_remove
        ]
        to_add = self.repr_add() if getattr(self, "repr_add", None) else []
        if to_add:
            children = children + to_add

        for key, child in children:
            if child is not None:
                continue
            mod_str = repr(child)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
    lines = extra_lines + child_lines

    name = self._get_name() if getattr(self, "_get_name", None) else self.__class__.__name__
    main_str = name + "("
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    main_str += ")"

    return main_str


def layer_repr(self):
    return _layer_repr(self, track_children=True)


def layer_repr_no_children(self):
    return _layer_repr(self, track_children=False)


def dense_extra_repr(self):
    return ", ".join(
        [
            str(self.units),
            f"activation={self.activation.__name__}",
            f"use_bias={str(self.use_bias)}",
        ]
    )
