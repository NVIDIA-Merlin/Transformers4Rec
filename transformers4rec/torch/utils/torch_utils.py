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
    if isinstance(input_size, dict):
        input_size = [i for i in input_size.values() if isinstance(i, torch.Size)][0]

    return input_size[0]


def check_inputs(ks, scores, labels):
    if len(ks.shape) > 1:
        raise ValueError("ks should be a 1-dimensional tensor")

    if len(scores.shape) != 2:
        raise ValueError("scores must be a 2-dimensional tensor")

    if len(labels.shape) != 2:
        raise ValueError("labels must be a 2-dimensional tensor")

    if scores.shape != labels.shape:
        raise ValueError("scores and labels must be the same shape")

    return (
        ks.to(dtype=torch.int32, device=scores.device),
        scores.to(dtype=torch.float32, device=scores.device),
        labels.to(dtype=torch.float32, device=scores.device),
    )


def extract_topk(ks, scores, labels):
    max_k = int(max(ks))
    topk_scores, topk_indices = torch.topk(scores, max_k)
    topk_labels = torch.gather(labels, 1, topk_indices)
    return topk_scores, topk_indices, topk_labels


def create_output_placeholder(scores, ks):
    return torch.zeros(scores.shape[0], len(ks)).to(device=scores.device, dtype=torch.float32)


def tranform_label_to_onehot(labels, vocab_size):
    return torch.nn.functional.one_hot(labels.reshape(-1), vocab_size).detach()
