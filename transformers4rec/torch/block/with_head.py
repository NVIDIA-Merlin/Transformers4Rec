import inspect
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ..head import Head
from .base import BlockType


class BlockWithHead(torch.nn.Module):
    def __init__(self, block: BlockType, head: Head, optimizer=torch.optim.Adam, model_name=None):
        super().__init__()
        self.model_name = model_name
        self.block = block
        self.head = head
        self.optimizer = optimizer

    def forward(self, inputs, *args, **kwargs):
        return self.head(self.block(inputs, *args, **kwargs), **kwargs)

    def compute_loss(self, inputs, targets) -> torch.Tensor:
        block_outputs = self.block(inputs)
        return self.head.compute_loss(block_outputs, targets)

    def calculate_metrics(self, inputs, targets, mode="val") -> Dict[str, torch.Tensor]:
        block_outputs = self.block(inputs)
        return self.head.calculate_metrics(block_outputs, targets, mode=mode)

    def compute_metrics(self, mode=None):
        return self.head.compute_metrics(mode=mode)

    def reset_metrics(self):
        return self.head.reset_metrics()

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
        if self.model_name:
            return self.model_name

        return f"{self.block._get_name()}WithHead"
