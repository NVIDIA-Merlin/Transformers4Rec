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

from abc import ABC
from functools import reduce
from typing import Dict, List, Optional, Union

import tensorflow as tf

from merlin_standard_lib import Registry, RegistryMixin, Schema
from merlin_standard_lib.utils.doc_utils import docstring_parameter
from transformers4rec.config.schema import SchemaMixin

from ..block.base import Block, SequentialBlock
from ..typing import TabularData, TensorOrTabularData
from ..utils.tf_utils import (
    calculate_batch_size_from_input_shapes,
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)

tabular_transformation_registry: Registry = Registry.class_registry("tf.tabular_transformations")
tabular_aggregation_registry: Registry = Registry.class_registry("tf.tabular_aggregations")


class TabularTransformation(
    SchemaMixin, tf.keras.layers.Layer, RegistryMixin["TabularTransformation"], ABC
):
    """Transformation that takes in `TabularData` and outputs `TabularData`."""

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        raise NotImplementedError()

    @classmethod
    def registry(cls) -> Registry:
        return tabular_transformation_registry


class TabularAggregation(
    SchemaMixin, tf.keras.layers.Layer, RegistryMixin["TabularAggregation"], ABC
):
    """Aggregation of `TabularData` that outputs a single `Tensor`"""

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    @classmethod
    def registry(cls) -> Registry:
        return tabular_aggregation_registry

    def _expand_non_sequential_features(self, inputs: TabularData) -> TabularData:
        inputs_sizes = {k: v.shape for k, v in inputs.items()}
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(inputs_sizes)

        if len(seq_features_shapes) > 0:
            non_seq_features = set(inputs.keys()).difference(set(seq_features_shapes.keys()))
            for fname in non_seq_features:
                # Including the 2nd dim and repeating for the sequence length
                inputs[fname] = tf.tile(tf.expand_dims(inputs[fname], 1), (1, sequence_length, 1))

        return inputs

    def _get_seq_features_shapes(self, inputs_sizes: Dict[str, tf.TensorShape]):
        seq_features_shapes = dict()
        for fname, fshape in inputs_sizes.items():
            # Saves the shapes of sequential features
            if len(fshape) >= 3:
                seq_features_shapes[fname] = tuple(fshape[:2])

        sequence_length = 0
        if len(seq_features_shapes) > 0:
            if len(set(seq_features_shapes.values())) > 1:
                raise ValueError(
                    "All sequential features must share the same shape in the first two dims "
                    "(batch_size, seq_length): {}".format(seq_features_shapes)
                )

            sequence_length = list(seq_features_shapes.values())[0][1]

        return seq_features_shapes, sequence_length

    def _check_concat_shapes(self, inputs: TabularData):
        input_sizes = {k: v.shape for k, v in inputs.items()}
        if len(set([tuple(v[:-1]) for v in input_sizes.values()])) > 1:
            raise Exception(
                "All features dimensions except the last one must match: {}".format(input_sizes)
            )

    def _get_agg_output_size(self, input_size, agg_dim):
        batch_size = calculate_batch_size_from_input_shapes(input_size)
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(input_size)

        if len(seq_features_shapes) > 0:
            return (
                batch_size,
                sequence_length,
                agg_dim,
            )

        else:
            return (batch_size, agg_dim)


TabularTransformationType = Union[str, TabularTransformation]
TabularTransformationsType = Union[TabularTransformationType, List[TabularTransformationType]]
TabularAggregationType = Union[str, TabularAggregation]


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class SequentialTabularTransformations(SequentialBlock):
    """A sequential container, modules will be added to it in the order they are passed in.

    Parameters
    ----------
    transformation: TabularTransformationType
        transformations that are passed in here will be called in order.
    """

    def __init__(self, transformation: TabularTransformationsType):
        if len(transformation) == 1 and isinstance(transformation[0], list):
            transformation = transformation[0]
        if not isinstance(transformation, (list, tuple)):
            transformation = [transformation]
        super().__init__([TabularTransformation.parse(t) for t in transformation])

    def append(self, transformation):
        self.transformations.append(TabularTransformation.parse(transformation))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        layers = [
            tf.keras.utils.deserialize_keras_object(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        return SequentialTabularTransformations(layers)


TABULAR_MODULE_PARAMS_DOCSTRING = """
    pre: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs when the module is called (so **before** `call`).
    post: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs after the module is called (so **after** `call`).
    aggregation: Union[str, TabularAggregation], optional
        Aggregation to apply after processing the `call`-method to output a single Tensor.

        Next to providing a class that extends TabularAggregation, it's also possible to provide
        the name that the class is registered in the `tabular_aggregation_registry`. Out of the box
        this contains: "concat", "stack", "element-wise-sum" &
        "element-wise-sum-item-multi".
    schema: Optional[DatasetSchema]
        DatasetSchema containing the columns used in this block.
    name: Optional[str]
        Name of the layer.
"""


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class TabularBlock(Block):
    """Layer that's specialized for tabular-data by integrating many often used operations.

    Note, when extending this class, typically you want to overwrite the `compute_call_output_shape`
    method instead of the normal `compute_output_shape`. This because a Block can contain pre- and
    post-processing and the output-shapes are handled automatically in `compute_output_shape`. The
    output of `compute_call_output_shape` should be the shape that's outputted by the `call`-method.

    Parameters
    ----------
    {tabular_module_parameters}
    """

    def __init__(
        self,
        pre: Optional[TabularTransformationsType] = None,
        post: Optional[TabularTransformationsType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.input_size = None
        self.set_pre(pre)
        self.set_post(post)
        self.set_aggregation(aggregation)

        if schema:
            self.set_schema(schema)

    @classmethod
    def from_schema(cls, schema: Schema, tags=None, **kwargs) -> Optional["TabularBlock"]:
        """Instantiate a TabularLayer instance from a DatasetSchema.

        Parameters
        ----------
        schema
        tags
        kwargs

        Returns
        -------
        Optional[TabularModule]
        """
        schema_copy = schema.copy()
        if tags:
            schema_copy = schema_copy.select_by_tag(tags)

        if not schema_copy.column_names:
            return None

        return cls.from_features(schema_copy.column_names, schema=schema_copy, **kwargs)

    @classmethod
    @docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING, extra_padding=4)
    def from_features(
        cls,
        features: List[str],
        pre: Optional[TabularTransformationsType] = None,
        post: Optional[TabularTransformationsType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        name=None,
        **kwargs,
    ) -> "TabularBlock":
        """
        Initializes a TabularLayer instance where the contents of features will be filtered out

        Parameters
        ----------
        features: List[str]
            A list of feature-names that will be used as the first pre-processing op to filter out
            all other features not in this list.
        {tabular_module_parameters}

        Returns
        -------
        TabularModule
        """
        pre = [FilterFeatures(features), pre] if pre else FilterFeatures(features)  # type: ignore

        return cls(pre=pre, post=post, aggregation=aggregation, name=name, **kwargs)

    def pre_call(
        self, inputs: TabularData, transformations: Optional[TabularTransformationsType] = None
    ) -> TabularData:
        """Method that's typically called before the forward method for pre-processing.

        Parameters
        ----------
        inputs: TabularData
             input-data, typically the output of the forward method.
        transformations: TabularTransformationsType, optional

        Returns
        -------
        TabularData
        """
        return self._maybe_apply_transformations(
            inputs, transformations=transformations or self.pre
        )

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        return inputs

    def post_call(
        self,
        inputs: TabularData,
        transformations: Optional[TabularTransformationsType] = None,
        merge_with: Union["TabularBlock", List["TabularBlock"]] = None,
        aggregation: Optional[TabularAggregationType] = None,
    ) -> TensorOrTabularData:
        """Method that's typically called after the forward method for post-processing.

        Parameters
        ----------
        inputs: TabularData
            input-data, typically the output of the forward method.
        transformations: TabularTransformationType, optional
            Transformations to apply on the input data.
        merge_with: Union[TabularModule, List[TabularModule]], optional
            Other TabularModule's to call and merge the outputs with.
        aggregation: TabularAggregationType, optional
            Aggregation to aggregate the output to a single Tensor.

        Returns
        -------
        TensorOrTabularData (Tensor when aggregation is set, else TabularData)
        """
        _aggregation: Optional[TabularAggregation] = None
        if aggregation:
            _aggregation = TabularAggregation.parse(aggregation)
        _aggregation = _aggregation or getattr(self, "aggregation", None)

        outputs = inputs
        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        outputs = self._maybe_apply_transformations(
            outputs, transformations=transformations or self.post
        )

        if _aggregation:
            schema = getattr(self, "schema", None)
            _aggregation.set_schema(schema)
            return _aggregation(outputs)

        return outputs

    def __call__(  # type: ignore
        self,
        inputs: TabularData,
        *args,
        pre: Optional[TabularTransformationsType] = None,
        post: Optional[TabularTransformationsType] = None,
        merge_with: Union["TabularBlock", List["TabularBlock"]] = None,
        aggregation: Optional[TabularAggregationType] = None,
        **kwargs,
    ) -> TensorOrTabularData:
        """We overwrite the call method in order to be able to do pre- and post-processing.

        Parameters
        ----------
        inputs: TabularData
            Input TabularData.
        pre: TabularTransformationsType, optional
            Transformations to apply before calling the forward method. If pre is None, this method
            will check if `self.pre` is set.
        post: TabularTransformationsType, optional
            Transformations to apply after calling the forward method. If post is None, this method
            will check if `self.post` is set.
        merge_with: Union[TabularModule, List[TabularModule]]
            Other TabularModule's to call and merge the outputs with.
        aggregation: TabularAggregationType, optional
            Aggregation to aggregate the output to a single Tensor.

        Returns
        -------
        TensorOrTabularData (Tensor when aggregation is set, else TabularData)
        """
        inputs = self.pre_call(inputs, transformations=pre)

        # This will call the `forward` method implemented by the super class.
        outputs = super().__call__(inputs, *args, **kwargs)  # noqa

        if isinstance(outputs, dict):
            outputs = self.post_call(
                outputs, transformations=post, merge_with=merge_with, aggregation=aggregation
            )

        return outputs

    def _maybe_apply_transformations(
        self,
        inputs: TabularData,
        transformations: Optional[TabularTransformationsType] = None,
    ) -> TabularData:
        """Apply transformations to the inputs if these are defined.

        Parameters
        ----------
        inputs
        transformations

        Returns
        -------

        """
        if transformations:
            transformations = TabularTransformation.parse(transformations)
            return transformations(inputs)

        return inputs

    def compute_call_output_shape(self, input_shapes):
        return input_shapes

    def compute_output_shape(self, input_shapes):
        if self.pre:
            input_shapes = self.pre.compute_output_shape(input_shapes)

        output_shapes = self._check_post_output_size(self.compute_call_output_shape(input_shapes))

        return output_shapes

    def get_config(self):
        config = super(TabularBlock, self).get_config()
        config = maybe_serialize_keras_objects(self, config, ["pre", "post", "aggregation"])

        if self.schema:
            config["schema"] = self.schema.to_json()

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = Schema().from_json(config["schema"])

        return super().from_config(config)

    def _check_post_output_size(self, input_shapes):
        output_shapes = input_shapes

        if isinstance(output_shapes, dict):
            if self.post:
                output_shapes = self.post.compute_output_shape(output_shapes)
            if self.aggregation:
                schema = getattr(self, "schema", None)
                self.aggregation.set_schema(schema)
                output_shapes = self.aggregation.compute_output_shape(output_shapes)

        return output_shapes

    def apply_to_all(self, inputs, columns_to_filter=None):
        if columns_to_filter:
            inputs = FilterFeatures(columns_to_filter)(inputs)
        outputs = tf.nest.map_structure(self, inputs)

        return outputs

    def set_schema(self, schema=None):
        self._maybe_set_schema(self.pre, schema)
        self._maybe_set_schema(self.post, schema)
        self._maybe_set_schema(self.aggregation, schema)

        return super().set_schema(schema)

    def set_pre(self, value: Optional[TabularTransformationsType]):
        if value and isinstance(value, SequentialTabularTransformations):
            self._pre: Optional[SequentialTabularTransformations] = value
        elif value and isinstance(value, (tf.keras.layers.Layer, list)):
            self._pre = SequentialTabularTransformations(value)
        else:
            self._pre = None

    @property
    def pre(self) -> Optional[SequentialTabularTransformations]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._pre

    @property
    def post(self) -> Optional[SequentialTabularTransformations]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._post

    def set_post(self, value: Optional[TabularTransformationsType]):
        if value and isinstance(value, SequentialTabularTransformations):
            self._post: Optional[SequentialTabularTransformations] = value
        elif value and isinstance(value, (tf.keras.layers.Layer, list)):
            self._post = SequentialTabularTransformations(value)
        else:
            self._post = None

    @property
    def aggregation(self) -> Optional[TabularAggregation]:
        """

        Returns
        -------
        TabularAggregation, optional
        """
        return self._aggregation

    def set_aggregation(self, value: Optional[Union[str, TabularAggregation]]):
        """

        Parameters
        ----------
        value
        """
        if value:
            self._aggregation: Optional[TabularAggregation] = TabularAggregation.parse(value)
        else:
            self._aggregation = None

    def repr_ignore(self):
        return []

    def repr_extra(self):
        return []

    def repr_add(self):
        return []

    @staticmethod
    def calculate_batch_size_from_input_shapes(input_shapes):
        return calculate_batch_size_from_input_shapes(input_shapes)

    def __rrshift__(self, other):
        from ..block.base import right_shift_layer

        return right_shift_layer(self, other)


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class FilterFeatures(TabularTransformation):
    """Transformation that filters out certain features from `TabularData`."

    Parameters
    ----------
    to_include: List[str]
        List of features to include in the result of calling the module
    pop: bool
        Boolean indicating whether to pop the features to exclude from the inputs dictionary.
    """

    def __init__(
        self, to_include, trainable=False, name=None, dtype=None, dynamic=False, pop=False, **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.to_include = to_include
        self.pop = pop

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        """Filter out features from inputs.

        Parameters
        ----------
        inputs: TabularData
            Input dictionary containing features to filter.

        Returns Filtered TabularData that only contains the feature-names in `self.to_include`.
        -------

        """
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.to_include}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    def compute_output_shape(self, input_shape):
        return {k: v for k, v in input_shape.items() if k in self.to_include}

    def get_config(self):
        config = super().get_config()
        config["to_include"] = self.to_include

        return config


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class MergeTabular(TabularBlock):
    """Merge multiple TabularModule's into a single output of TabularData.

    Parameters
    ----------
    blocks_to_merge: Union[TabularModule, Dict[str, TabularBlock]]
        TabularBlocks to merge into, this can also be one or multiple dictionaries keyed by the
        name the module should have.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        *blocks_to_merge: Union[TabularBlock, Dict[str, TabularBlock]],
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
        self.to_merge: Union[List[TabularBlock], Dict[str, TabularBlock]]
        if all(isinstance(x, dict) for x in blocks_to_merge):
            to_merge: Dict[str, TabularBlock] = reduce(
                lambda a, b: dict(a, **b), blocks_to_merge
            )  # type: ignore
            self.to_merge = to_merge
        elif all(isinstance(x, tf.keras.layers.Layer) for x in blocks_to_merge):
            self.to_merge = list(blocks_to_merge)  # type: ignore
        else:
            raise ValueError(
                "Please provide one or multiple layer's to merge or "
                f"dictionaries of layer. got: {blocks_to_merge}"
            )

        # Merge schemas if necessary.
        if not schema and all(getattr(m, "schema", False) for m in self.merge_values):
            s = reduce(lambda a, b: a + b, [m.schema for m in self.merge_values])  # type: ignore
            self.set_schema(s)

    def build(self, input_shape):
        if isinstance(self.to_merge, dict):
            layers = self.to_merge.values()
        else:
            layers = self.to_merge

        for layer in layers:
            layer.build(input_shape)

    @property
    def merge_values(self) -> List[tf.keras.layers.Layer]:
        if isinstance(self.to_merge, dict):
            return list(self.to_merge.values())

        return self.to_merge

    @property
    def to_merge_dict(self) -> Dict[str, tf.keras.layers.Layer]:
        if isinstance(self.to_merge, dict):
            return self.to_merge

        return {str(i): m for i, m in enumerate(self.to_merge)}

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {}
        for layer in self.merge_values:
            outputs.update(layer(inputs))

        return outputs

    def compute_call_output_shape(self, input_shape):
        output_shapes = {}

        for layer in self.merge_values:
            output_shapes.update(layer.compute_output_shape(input_shape))

        return output_shapes

    def get_config(self):
        return maybe_serialize_keras_objects(
            self, super(MergeTabular, self).get_config(), ["to_merge"]
        )


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class AsTabular(tf.keras.layers.Layer):
    """Converts a Tensor to TabularData by converting it to a dictionary.

    Parameters
    ----------
    output_name: str
        Name that should be used as the key in the output dictionary.
    name: str
        Name of the layer.
    """

    def __init__(self, output_name: str, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_name = output_name

    def call(self, inputs, **kwargs):
        return {self.output_name: inputs}

    def get_config(self):
        config = super(AsTabular, self).get_config()
        config["output_name"] = self.output_name

        return config


def merge_tabular(self, other, aggregation=None, **kwargs):
    return MergeTabular(self, other, aggregation=aggregation, **kwargs)


TabularBlock.__add__ = merge_tabular
TabularBlock.merge = merge_tabular
