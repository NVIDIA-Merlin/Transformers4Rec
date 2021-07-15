import torch


def calculate_batch_size_from_input_size(input_size):
    return [i for i in input_size.values() if isinstance(i, torch.Size)][0][0]
