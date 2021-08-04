import torch


def get_output_sizes_from_schema(schema, batch_size=-1, max_sequence_length=None):
    sizes = {}
    for feature in schema.feature:
        name = feature.name
        if feature.HasField("value_count"):
            sizes[name] = torch.Size(
                [
                    batch_size,
                    max_sequence_length if max_sequence_length else feature.value_count.max,
                ]
            )
        elif feature.HasField("shape"):
            sizes[name] = torch.Size([batch_size] + [d.size for d in feature.shape.dim])
        else:
            sizes[name] = torch.Size([batch_size, 1])

    return sizes


def calculate_batch_size_from_input_size(input_size):
    return [i for i in input_size.values() if isinstance(i, torch.Size)][0][0]
