from abc import ABC
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Optional, Union

import tensorflow as tf

from ...types import DatasetSchema
from ...utils.misc_utils import docstring_parameter
from ...utils.registry import Registry, RegistryMixin
from ...utils.schema import SchemaMixin
from ..block.base import Block, SequentialBlock
from ..typing import TabularData, TensorOrTabularData
from ..utils.tf_utils import calculate_batch_size_from_input_shapes

tabular_transformation_registry: Registry = Registry.class_registry("tf.tabular_transformations")
tabular_aggregation_registry: Registry = Registry.class_registry("tf.tabular_aggregations")


class TabularTransformation(SchemaMixin, tf.keras.layers.Layer, RegistryMixin, ABC):
    """Transformation that takes in `TabularData` and outputs `TabularData`."""

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        raise NotImplementedError()

    @classmethod
    def registry(cls) -> Registry:
        return tabular_transformation_registry


class TabularAggregation(SchemaMixin, tf.keras.layers.Layer, RegistryMixin, ABC):
    """Aggregation of `TabularData` that outputs a single `Tensor`"""

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    @classmethod
    def registry(cls) -> Registry:
        return tabular_aggregation_registry


TabularTransformationType = Union[
    str, TabularTransformation, List[str], List[TabularTransformation]
]
TabularAggregationType = Union[str, TabularAggregation]


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class SequentialTabularTransformations(SequentialBlock):
    """A sequential container, modules will be added to it in the order they are passed in.

    Parameters
    ----------
    transformation: TabularTransformationType
        transformations that are passed in here will be called in order.
    """

    def __init__(self, *transformation: TabularTransformationType):
        if len(transformation) == 1 and isinstance(transformation, list):
            transformation = transformation[0]
        super().__init__([TabularTransformation.parse(t) for t in transformation])

    def append(self, transformation):
        self.transformations.append(TabularTransformation.parse(transformation))


TABULAR_MODULE_PARAMS_DOCSTRING = """
    pre: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs when the module is called (so **before** `call`).
    post: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs after the module is called (so **after** `call`).
    aggregation: Union[str, TabularAggregation], optional
        Aggregation to apply after processing the `call`-method to output a single Tensor.

        Next to providing a class that extends TabularAggregation, it's also possible to provide
        the name that the class is registered in the `tabular_aggregation_registry`. Out of the box
        this contains: "concat", "stack", "sequential-concat", "element-wise-sum" &
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
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[DatasetSchema] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.input_size = None
        self.set_pre(pre)
        self.set_post(post)
        self.set_aggregation(aggregation)

        if schema:
            self.set_schema(schema)

    @classmethod
    def from_schema(cls, schema: DatasetSchema, tags=None, **kwargs) -> Optional["TabularBlock"]:
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
        _schema = deepcopy(schema)
        if tags:
            _schema = _schema.select_by_tag(tags)

        if not _schema.columns:
            return None

        return cls.from_features(_schema.column_names, schema=_schema, **kwargs)

    @classmethod
    @docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
    def from_features(
        cls,
        features: List[str],
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        name=None,
        **kwargs
    ) -> "TabularBlock":
        """Initializes a TabularLayer instance where the contents of features will be filtered
            out

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
        pre = [FilterFeatures(features), pre] if pre else FilterFeatures(features)

        return cls(pre=pre, post=post, aggregation=aggregation, name=name, **kwargs)

    def pre_call(
        self, inputs: TabularData, transformations: Optional[TabularAggregationType] = None
    ) -> TabularData:
        """Method that's typically called before the forward method for pre-processing.

        Parameters
        ----------
        inputs: TabularData
             input-data, typically the output of the forward method.
        transformations: TabularAggregationType, optional

        Returns
        -------
        TabularData
        """
        return self._maybe_apply_transformations(
            inputs, transformations=transformations or self.pre
        )

    def call(self, x: TabularData, *args, **kwargs) -> TabularData:
        return x

    def post_call(
        self,
        inputs: TabularData,
        transformations: Optional[TabularTransformationType] = None,
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
        if aggregation:
            aggregation = TabularAggregation.parse(aggregation)
        aggregation = aggregation or getattr(self, "aggregation", None)

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

        if aggregation:
            return aggregation(outputs)

        return outputs

    def __call__(
        self,
        inputs: TabularData,
        *args,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        merge_with: Union["TabularBlock", List["TabularBlock"]] = None,
        aggregation: Optional[TabularAggregationType] = None,
        **kwargs
    ) -> TensorOrTabularData:
        """We overwrite the call method in order to be able to do pre- and post-processing.

        Parameters
        ----------
        inputs: TabularData
            Input TabularData.
        pre: TabularTransformationType, optional
            Transformations to apply before calling the forward method. If pre is None, this method
            will check if `self.pre` is set.
        post: TabularTransformationType, optional
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
        transformations: Optional[TabularTransformationType] = None,
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

        if self.pre:
            config["pre"] = tf.keras.utils.serialize_keras_object(self.pre)
        if self.post:
            config["post"] = tf.keras.utils.serialize_keras_object(self.post)
        if self.aggregation:
            config["aggregation"] = tf.keras.utils.serialize_keras_object(self.aggregation)
        if self.schema:
            config["schema"] = self.schema.to_proto_str()

        return config

    @classmethod
    def from_config(cls, config):
        if "schema" in config:
            config["schema"] = DatasetSchema.from_proto(config["schema"])
        if "pre" in config:
            config["pre"] = tf.keras.utils.deserialize_keras_object(config["pre"])
        if "post" in config:
            config["post"] = tf.keras.utils.deserialize_keras_object(config["post"])
        if "aggregation" in config:
            config["aggregation"] = tf.keras.utils.deserialize_keras_object(config["aggregation"])

        return super().from_config(config)

    def _check_post_output_size(self, input_shapes):
        output_shapes = input_shapes

        if isinstance(output_shapes, dict):
            if self.post:
                output_shapes = self.post.compute_output_shape(output_shapes)
            if self.aggregation:
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

    def set_pre(
        self, value: Union[str, TabularTransformation, List[str], List[TabularTransformation]]
    ):
        if value:
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

    def set_post(
        self, value: Union[str, TabularTransformation, List[str], List[TabularTransformation]]
    ):
        if value:
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
            self._aggregation = TabularAggregation.parse(value)
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
        return {
            "to_include": self.to_include,
        }


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
        schema: Optional[DatasetSchema] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
        if all(isinstance(x, dict) for x in blocks_to_merge):
            blocks_to_merge = reduce(lambda a, b: dict(a, **b), blocks_to_merge)
            self.to_merge = blocks_to_merge
        else:
            self.to_merge = list(blocks_to_merge)

        # Merge schemas if necessary.
        if not schema and all(getattr(m, "schema", False) for m in self.merge_values):
            self.set_schema(reduce(lambda a, b: a + b, [m.schema for m in self.merge_values]))

    @property
    def merge_values(self):
        if isinstance(self.to_merge, dict):
            return list(self.to_merge.values())

        return self.to_merge

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
        return {"merge_layers": tf.keras.utils.serialize_keras_object(self.merge_layers)}


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
        return {
            "axis": self.axis,
        }


def merge_tabular(self, other, aggregation=None, **kwargs):
    return MergeTabular(self, other, aggregation=aggregation, **kwargs)


TabularBlock.__add__ = merge_tabular
TabularBlock.merge = merge_tabular
