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


def docstring_parameter(*args, extra_padding=None, **kwargs):
    def dec(obj):
        if extra_padding:

            def pad(value):
                return ("\n" + " " * extra_padding).join(value.split("\n"))

            nonlocal args, kwargs
            kwargs = {key: pad(value) for key, value in kwargs.items()}
            args = [pad(value) for value in args]
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec
