from abc import ABC
from functools import reduce
from typing import Dict, List, Optional, Union

import tensorflow as tf

from ...types import DatasetSchema
from ...utils.misc_utils import docstring_parameter
from ...utils.registry import Registry
from ..block.base import Block, SequentialBlock
from ..typing import TabularData, TensorOrTabularData
from ..utils.tf_utils import SchemaMixin, calculate_batch_size_from_input_shapes

tabular_transformation_registry: Registry = Registry.class_registry("tf.tabular_transformations")
tabular_aggregation_registry: Registry = Registry.class_registry("tf.tabular_aggregations")


class TabularTransformation(SchemaMixin, tf.keras.layers.Layer, ABC):
    """Transformation that takes in `TabularData` and outputs `TabularData`."""

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        raise NotImplementedError()

    @classmethod
    def parse(cls, class_or_str):
        return tabular_transformation_registry.parse(class_or_str)


class TabularAggregation(SchemaMixin, tf.keras.layers.Layer, ABC):
    """Aggregation of `TabularData` that outputs a single `Tensor`"""

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    @classmethod
    def parse(cls, class_or_str):
        return tabular_aggregation_registry.parse(class_or_str)


TabularTransformationType = Union[
    str, TabularTransformation, List[str], List[TabularTransformation]
]
TabularAggregationType = Union[str, TabularAggregation]


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
        super().__init__(*[TabularTransformation.parse(t) for t in transformation])

    def append(self, transformation):
        self.transformations.append(TabularTransformation.parse(transformation))


TABULAR_MODULE_PARAMS_DOCSTRING = """
    pre: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs when the module is called (so **before** `call`).
    post: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs after the module is called (so **after** `call`).
    aggregation: Union[str, TabularAggregation], optional
        Aggregation to apply after processing the `call`-method to output a single Tensor.
"""


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
class TabularBlock(Block):
    """Layer that's specialized for tabular-data by integrating many often used operations.

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
        **kwargs
    ):
        super().__init__(**kwargs)
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
        if tags:
            schema = schema.select_by_tag(tags)

        if not schema.columns:
            return None

        return cls.from_features(schema.column_names, schema=schema, **kwargs)

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
        raise NotImplementedError()

    def compute_output_shape(self, input_shapes):
        if self.pre:
            input_shapes = self.pre.compute_output_shape(input_shapes)

        output_shapes = self._check_post_output_size(self.compute_call_output_shape(input_shapes))

        return output_shapes

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
    def __init__(
        self, to_include, trainable=False, name=None, dtype=None, dynamic=False, pop=False, **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.to_include = to_include
        self.pop = pop

    def call(self, inputs, **kwargs):
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
    def __init__(
        self,
        *blocks_to_merge: Union[TabularBlock, Dict[str, TabularBlock]],
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        name=None,
        **kwargs
    ):
        super().__init__(pre=pre, post=post, aggregation=aggregation, name=name, **kwargs)
        if all(isinstance(x, dict) for x in blocks_to_merge):
            blocks_to_merge = reduce(lambda a, b: dict(a, **b), blocks_to_merge)
            self.to_merge = blocks_to_merge
        else:
            self.to_merge = list(blocks_to_merge)

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
    def __init__(
        self, output_name, trainable=False, name=None, dtype=None, dynamic=False, **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
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


class AsSparseFeatures(TabularTransformation):
    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                outputs[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()
            else:
                outputs[name] = val

        return outputs


class AsDenseFeatures(TabularTransformation):
    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                outputs[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_tensor()
            else:
                outputs[name] = val

        return outputs
