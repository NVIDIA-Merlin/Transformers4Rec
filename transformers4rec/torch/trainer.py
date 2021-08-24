from collections.abc import Sized

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from ..config.trainer import T4RecTrainerConfig

logger = logging.get_logger(__name__)


class T4recTrainer(Trainer):
    """
    An :class:`~transformers.Trainer` specialized for sequential recommendation
    including (session-based and sequtial recommendation)
    """

    def __init__(self, model: torch.nn.Module, args: T4RecTrainerConfig, **kwargs):

        self.past_global_steps = 0

        recsys_callback = RecSysTrainerCallback(self)
        mock_dataset = DatasetMock()

        super(T4recTrainer, self).__init__(
            model=model,
            args=args,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            callbacks=[recsys_callback],
            **kwargs,
        )

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def set_train_dataloader(self, dataloader: DataLoader):
        # Check that values are consistent between data-loader
        # and TrainingArg class
        print(dataloader.batch_size)
        assert (
            dataloader.batch_size == self.args.train_batch_size
        ), "batch size of dataloader {} should match ".format(dataloader.batch_size)
        "train batch size of T4RecTrainerConfig {}".format(self.args.train_batch_size)

        assert (
            dataloader.drop_last == self.args.dataloader_drop_last
        ), "Make sure drop_last is set to '{}' ".format(dataloader.drop_last)
        "in dataloader.drop_last and T4RecTrainerConfig.dataloader_drop_last"

        self.train_dataloader = dataloader

    def set_eval_dataloader(self, dataloader: DataLoader):
        assert (
            dataloader.batch_size == self.args.eval_batch_size
        ), "batch size of dataloader {} should match ".format(dataloader.batch_size)
        "eval batch size of T4RecTrainerConfig {}".format(self.args.eval_batch_size)

        self.eval_dataloader = dataloader


# Mock to inform HF Trainer that the dataset is sized, and can be obtained via the data loader
# This is needed because we are decoupling dataloading from the trainer
class DatasetMock(Dataset, Sized):
    def __init__(self, nsteps=1):
        self.nsteps = nsteps

    def __len__(self):
        return self.nsteps


class RecSysTrainerCallback(TrainerCallback):
    """
    An :class:`~transformers.TrainerCallback` that changes the state of the Trainer
    on specific hooks for the purpose of the RecSysTrainer
    """

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        pass

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Increments the global steps for logging with the global steps of the last train()
        self.trainer._increment_past_global_steps(state.global_step)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Evaluates on eval set
        # self.trainer.evaluate()
        pass
