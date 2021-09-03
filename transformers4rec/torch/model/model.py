import inspect
from typing import Dict, List, Optional, Type, Union

import numpy as np
import torch
from tqdm import tqdm

from ..typing import TensorOrTabularData
from .head import Head


class Model(torch.nn.Module):
    """Model class that can aggregate one of multiple heads.

    Parameters
    ----------
    head: Head
        One or more heads of the model.
    head_weights: List[float], optional
        Weight-value to use for each head.
    head_reduction: str, optional
        How to reduce the losses into a single tensor when multiple heads are used.
    optimizer: Type[torch.optim.Optimizer]
        Optimizer-class to use during fitting
    name: str, optional
        Name of the model.
    """

    def __init__(
        self,
        *head: Head,
        head_weights: Optional[List[float]] = None,
        head_reduction: Optional[str] = "mean",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        name=None
    ):
        if head_weights:
            if not isinstance(head_weights, list):
                raise ValueError("`head_weights` must be a list")
            if not len(head_weights) == len(head):
                raise ValueError(
                    "`head_weights` needs to have the same length " "as the number of heads"
                )

        super().__init__()

        self.name = name
        self.heads = torch.nn.ModuleList(head)
        self.head_weights = head_weights or [1.0] * len(head)
        self.head_reduction = head_reduction
        self.optimizer = optimizer

    def forward(self, inputs: TensorOrTabularData, **kwargs):
        # TODO: Optimize this
        outputs = {}
        for head in self.heads:
            outputs.update(head(inputs, call_body=True, always_output_dict=True))

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]

        return outputs

    def compute_loss(self, inputs, targets, compute_metrics=True, **kwargs) -> torch.Tensor:
        losses = []

        for i, head in enumerate(self.heads):
            loss = head.compute_loss(
                inputs, targets, call_body=True, compute_metrics=compute_metrics, **kwargs
            )
            losses.append(loss * self.head_weights[i])

        loss_tensor = torch.stack(losses)

        return getattr(loss_tensor, self.head_reduction)()

    def calculate_metrics(
        self, inputs, targets, mode="val"
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        outputs = {}
        for head in self.heads:
            outputs.update(head.calculate_metrics(inputs, targets, mode=mode, call_body=True))

        return outputs

    def compute_metrics(self, mode=None):
        metrics = {}
        for head in self.heads:
            metrics.update(head.compute_metrics(mode=mode))

        return metrics

    def reset_metrics(self):
        for head in self.heads:
            head.reset_metrics()

    def to_lightning(self):
        import pytorch_lightning as pl

        parent_self = self

        class BlockWithHeadLightning(pl.LightningModule):
            def __init__(self):
                super(BlockWithHeadLightning, self).__init__()
                self.parent = parent_self

            def forward(self, inputs, *args, **kwargs):
                return self.parent(inputs, *args, **kwargs)

            def training_step(self, batch, batch_idx):
                loss = self.parent.compute_loss(*batch)
                self.log("train_loss", loss)

                return loss

            def configure_optimizers(self):
                optimizer = self.parent.optimizer(self.parent.parameters(), lr=1e-3)

                return optimizer

        return BlockWithHeadLightning()

    def fit(
        self,
        dataloader,
        optimizer=torch.optim.Adam,
        eval_dataloader=None,
        num_epochs=1,
        amp=False,
        train=True,
        verbose=True,
    ):
        if isinstance(dataloader, torch.utils.data.DataLoader):
            dataset = dataloader.dataset
        else:
            dataset = dataloader

        if inspect.isclass(optimizer):
            optimizer = optimizer(self.parameters())

        self.train(mode=train)
        epoch_losses = []
        with torch.set_grad_enabled(mode=train):
            for epoch in range(num_epochs):
                losses = []
                batch_iterator = enumerate(iter(dataset))
                if verbose:
                    batch_iterator = tqdm(batch_iterator)
                for batch_idx, (x, y) in batch_iterator:
                    if amp:
                        with torch.cuda.amp.autocast():
                            loss = self.compute_loss(x, y)
                    else:
                        loss = self.compute_loss(x, y)

                    losses.append(float(loss))

                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                if verbose:
                    print(self.compute_metrics(mode="train"))
                    if eval_dataloader:
                        print(self.evaluate(eval_dataloader, verbose=False))
                epoch_losses.append(np.mean(losses))

        return np.array(epoch_losses)

    def evaluate(self, dataloader, verbose=True, mode="eval"):
        if isinstance(dataloader, torch.utils.data.DataLoader):
            dataset = dataloader.dataset
        else:
            dataset = dataloader

        batch_iterator = enumerate(iter(dataset))
        if verbose:
            batch_iterator = tqdm(batch_iterator)
        self.reset_metrics()
        for batch_idx, (x, y) in batch_iterator:
            self.calculate_metrics(x, y, mode=mode)

        return self.compute_metrics(mode=mode)

    def _get_name(self):
        if self.name:
            return self.name

        return super(Model, self)._get_name()
