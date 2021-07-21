import torch

from ..utils.registry import Registry
from .tabular import TabularModule
from .typing import TensorOrTabularData

augmentation: Registry = Registry.class_registry("torch.augmentation")


class DataAugmentation(TabularModule):
    def augment(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, inputs: TensorOrTabularData, **kwargs) -> TensorOrTabularData:
        if isinstance(inputs, dict):
            return {key: self.augment(val) for key, val in inputs.items()}

        return self.augment(inputs)

    def forward_output_size(self, input_size):
        return input_size


@augmentation.register_with_multiple_names("stochastic-swap-noise", "ssn")
class StochasticSwapNoise(TabularModule):
    """
    Applies Stochastic replacement of sequence features
    """

    def __init__(self, pad_token, replacement_prob, aggregation=None):
        super().__init__(aggregation)
        self.pad_token = pad_token
        self.replacement_prob = replacement_prob

    def augment(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cdata_non_zero_mask = input_tensor != self.pad_token
            sse_prob_replacement_matrix = torch.full(
                input_tensor.shape,
                self.replacement_prob,
                device=input_tensor.device,
            )
            sse_replacement_mask = (
                torch.bernoulli(sse_prob_replacement_matrix).bool() & cdata_non_zero_mask
            )
            n_values_to_replace = sse_replacement_mask.sum()

            cdata_flattened_non_zero = torch.masked_select(input_tensor, cdata_non_zero_mask)

            sampled_values_to_replace = cdata_flattened_non_zero[
                torch.randperm(cdata_flattened_non_zero.shape[0])
            ][:n_values_to_replace]

            input_tensor[sse_replacement_mask] = sampled_values_to_replace

        return input_tensor
