import tensorflow as tf


def get_output_sizes_from_schema(schema, batch_size=0, max_sequence_length=None):
    sizes = {}
    for feature in schema.feature:
        name = feature.name
        if feature.HasField("value_count"):
            sizes[name] = tf.TensorShape(
                [
                    batch_size,
                    max_sequence_length if max_sequence_length else feature.value_count.max,
                ]
            )
        elif feature.HasField("shape"):
            sizes[name] = tf.TensorShape([batch_size] + [d.size for d in feature.shape.dim])
        else:
            sizes[name] = tf.TensorShape([batch_size, 1])

    return sizes


def calculate_batch_size_from_input_shapes(input_shapes):
    return [i for i in input_shapes.values() if not isinstance(i, tuple)][0][0]


def maybe_serialize_keras_objects(self, config, maybe_serialize_keys):
    for key in maybe_serialize_keys:
        maybe_value = getattr(self, key, None)
        if maybe_value:
            config[key] = tf.keras.utils.serialize_keras_object(maybe_value)

    return config


def maybe_deserialize_keras_objects(config, maybe_deserialize_keys):
    for key in maybe_deserialize_keys:
        if key in config:
            config[key] = tf.keras.utils.deserialize_keras_object(config[key])

    return config
