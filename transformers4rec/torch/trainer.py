# type: ignore

#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import collections
import inspect
import random
import re
from collections.abc import Sized
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Trainer as BaseTrainer
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalLoopOutput, SchedulerType
from transformers.utils import logging

from merlin_standard_lib import Schema

from ..config.trainer import T4RecTrainingArguments
from .model.base import Model
from .utils.data_utils import T4RecDataLoader

logger = logging.get_logger(__name__)


class Trainer(BaseTrainer):
    """
    An :class:`~transformers.Trainer` specialized for sequential recommendation
    including (session-based and sequtial recommendation)

    Parameters
    ----------
    model: Model
        The Model defined using Transformers4Rec api.
    args: T4RecTrainingArguments
        The training arguments needed to setup training and evaluation
        experiments.
    schema: Optional[Dataset.schema], optional
        The schema object including features to use and their properties.
        by default None
    train_dataset_or_path: Optional[Union[str, Dataset]], optional
        Path of parquet files or DataSet to use for training.
        by default None
    eval_dataset_or_path: Optional[str, Dataset], optional
        Path of parquet files or DataSet to use for evaluation.
        by default None
    train_dataloader: Optional[DataLoader], optional
        The data generator to use for training.
        by default None
    eval_dataloader: Optional[DataLoader], optional
        The data generator to use for evaluation.
        by default None
    compute_metrics: Optional[bool], optional
        Whether to compute metrics defined by Model class or not.
        by default None
    incremental_logging: bool
        Whether to enable incremental logging or not. If True, it ensures that
        global steps are incremented over many `trainer.train()` calls, so that
        train and eval metrics steps do not overlap and can be seen properly
        in reports like W&B and Tensorboard
    """

    def __init__(
        self,
        model: Model,
        args: T4RecTrainingArguments,
        schema: Schema = None,
        train_dataset_or_path=None,
        eval_dataset_or_path=None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics=None,
        incremental_logging: bool = False,
        **kwargs,
    ):

        mock_dataset = DatasetMock()
        hf_model = HFWrapper(model)

        self.incremental_logging = incremental_logging
        if self.incremental_logging:
            self.past_global_steps = 0
            incremental_logging_callback = IncrementalLoggingCallback(self)
            callbacks.append(incremental_logging_callback)

        super(Trainer, self).__init__(
            model=hf_model,
            args=args,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            callbacks=callbacks,
            **kwargs,
        )

        self.compute_metrics = compute_metrics
        self.train_dataset_or_path = train_dataset_or_path
        self.eval_dataset_or_path = eval_dataset_or_path
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.schema = schema
        self.incremental_logging = incremental_logging

    def get_train_dataloader(self):
        """
        Set the train dataloader to use by Trainer.
        It supports user defined data-loader set as an attribute in the constructor.
        When the attribute is None, The data-loader is defined using train_dataset
        and the `data_loader_engine` specified in Training Arguments.
        """
        if self.train_dataloader is not None:
            return self.train_dataloader

        assert self.schema is not None, "schema is required to generate Train Dataloader"
        return T4RecDataLoader.parse(self.args.data_loader_engine).from_schema(
            self.schema,
            self.train_dataset_or_path,
            self.args.per_device_train_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            drop_last=self.args.dataloader_drop_last,
            shuffle=True,
            shuffle_buffer_size=self.args.shuffle_buffer_size,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Set the eval dataloader to use by Trainer.
        It supports user defined data-loader set as an attribute in the constructor.
        When the attribute is None, The data-loader is defined using eval_dataset
        and the `data_loader_engine` specified in Training Arguments.
        """
        if self.eval_dataloader is not None:
            return self.eval_dataloader

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert self.schema is not None, "schema is required to generate Eval Dataloader"
        return T4RecDataLoader.parse(self.args.data_loader_engine).from_schema(
            self.schema,
            self.eval_dataset_or_path,
            self.args.per_device_eval_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            drop_last=self.args.dataloader_drop_last,
            shuffle=False,
            shuffle_buffer_size=self.args.shuffle_buffer_size,
        )

    def num_examples(self, dataloader: DataLoader):
        """
        Overriding :obj:`Trainer.num_examples()` method because
        the data loaders for this project do not return the dataset size,
        but the number of steps. So we estimate the dataset size here
        by multiplying the number of steps * batch size
        """
        """
        if dataloader == self.get_train_dataloader():
            batch_size = self.args.per_device_train_batch_size
        else:
            batch_size = self.args.per_device_eval_batch_size
        """
        return len(dataloader) * dataloader._batch_size

    def reset_lr_scheduler(self) -> None:
        """
        Resets the LR scheduler of the previous :obj:`Trainer.train()` call,
        so that a new LR scheduler one is created by the next :obj:`Trainer.train()` call.
        This is important for LR schedules like `get_linear_schedule_with_warmup()`
        which decays LR to 0 in the end of the train
        """
        self.lr_scheduler = None

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        # flexibility in scheduler with num_cycles as hyperparams
        if self.lr_scheduler is None:
            self.lr_scheduler = self.get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.args.learning_rate_num_cosine_cycles_by_epoch
                * self.args.num_train_epochs,
            )

    # Override the method get_scheduler to accept num_cycle params ?
    # The advantage is to use the unified HF API with many scheduler
    # we can also send a PR to HF ?
    @staticmethod
    def get_scheduler(
        name: Union[str, SchedulerType],
        optimizer: Optimizer,
        num_warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
        num_cycles: Optional[int] = 0.5,
    ):
        """
        Unified API to get any scheduler from its name.

        Parameters
        ----------
        name: (:obj:`str` or `:obj:`SchedulerType`)
            The name of the scheduler to use.
        optimizer: (:obj:`torch.optim.Optimizer`)
            The optimizer that will be used during training.
        num_warmup_steps: (:obj:`int`, `optional`)
            The number of warmup steps to do. This is not required by all schedulers
            (hence the argument being optional),
            the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps: (:obj:`int`, `optional`)
            The number of training steps to do. This is not required by all schedulers
            (hence the argument being optional),
            the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles: (:obj:`int`, `optional`)
            The number of waves in the cosine schedule /
            hard restarts to use for cosine scheduler
        """
        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
        if name == SchedulerType.CONSTANT:
            return schedule_func(optimizer)

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        if "num_cycles" in inspect.signature(schedule_func).parameters:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )

        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[float],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, Any]],
    ]:
        """
        Overriding :obj:`Trainer.prediction_step()`
        to provide more flexibility to unpack results from the model,
        like returning labels that are not exactly one input feature
        model
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(inputs, training=False)
            else:
                outputs = model(inputs, training=False)

            loss = outputs["loss"].mean().detach()

        if prediction_loss_only:
            return (loss, None, None, None)

        predictions = outputs["predictions"].detach()
        labels = outputs["labels"].detach()

        # TODO: define metadata dict in the model for logging
        # other_outputs = {
        #    k: v.detach() if isinstance(v, torch.Tensor) else v
        #    for k, v in outputs.items()
        #    if k not in ignore_keys + ["loss", "predictions", "labels"]
        # }
        other_outputs = None

        return (loss, predictions, labels, other_outputs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: Optional[str] = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding :obj:`Trainer.prediction_loop()`
        (shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`)
        to provide more flexibility to work with streaming metrics
        (computed at each eval batch) and
        to log with the outputs of the model
        (e.g. prediction scores, prediction metadata, attention weights)

        Parameters
        ----------
        dataloader: DataLoader
            DataLoader object to use to iterate over evaluation data
        description: str
            Parameter to describe the evaluation experiment.
            e.g: `Prediction`, `test`
        prediction_loss_only: Optional[bool]
            Whether or not to return the loss only.
            by default None
        ignore_keys: Optional[List[str]]
            Columns not accepted by the ``model.forward()`` method
            are automatically removed.
            by default None
        metric_key_prefix: Optional[str]
            Prefix to use when logging evaluation metrics.
            by default `eval`
        """
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        # set the model
        model = self.model.module
        # reset metrics for the dataset (Train, Valid or Test)
        if self.compute_metrics:
            model.reset_metrics()

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")

        batch_size = dataloader._batch_size

        logger.info("***** Running %s *****", description)
        logger.info("  Batch size = %d", batch_size)

        preds_item_ids_scores_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        if metric_key_prefix == "train" and self.args.eval_steps_on_train_set:
            num_examples = self.args.eval_steps_on_train_set * batch_size
        else:
            num_examples = self.num_examples(dataloader)

        logger.info("  Num sessions (examples) = %d", num_examples)

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_item_ids_scores_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds_item_ids_scores = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Iterate over dataloader
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Limits the number of evaluation steps on train set (which is usually larger)
            if (
                metric_key_prefix == "train"
                and self.args.eval_steps_on_train_set > 0
                and step + 1 > self.args.eval_steps_on_train_set
            ):
                break

            loss, preds, labels, outputs = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            # Updates metrics
            # TODO: compute metrics each N eval_steps to speedup evaluation
            metrics_results_detailed = None
            if self.compute_metrics:
                if step % self.args.compute_metrics_each_n_steps == 0:
                    metrics_results_detailed = model.calculate_metrics(
                        preds, labels, mode=metric_key_prefix, forward=False, call_body=False
                    )

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = (
                    losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
                )
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=0)
                )
            if preds is not None and self.args.predict_top_k > 0:
                preds_sorted_item_scores, preds_sorted_item_ids = torch.topk(
                    preds, k=self.args.predict_top_k, dim=-1
                )
                self._maybe_log_predictions(
                    labels,
                    preds_sorted_item_ids,
                    preds_sorted_item_scores,
                    # outputs["pred_metadata"],
                    metrics_results_detailed,
                    metric_key_prefix,
                )
                # The output predictions will be a tuple with the ranked top-n item ids,
                # and item recommendation scores
                preds_item_ids_scores = (
                    preds_sorted_item_ids,
                    preds_sorted_item_scores,
                )
                preds_item_ids_scores_host = (
                    preds_item_ids_scores
                    if preds_item_ids_scores_host is None
                    else nested_concat(
                        preds_item_ids_scores_host,
                        preds_item_ids_scores,
                    )
                )

            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU
            # if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=0)
                    )
                if preds_item_ids_scores_host is not None:
                    preds_item_ids_scores = nested_numpify(preds_item_ids_scores_host)
                    all_preds_item_ids_scores = (
                        preds_item_ids_scores
                        if all_preds_item_ids_scores is None
                        else nested_concat(
                            all_preds_item_ids_scores,
                            preds_item_ids_scores,
                        )
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_item_ids_scores_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels if all_labels is None else nested_concat(all_labels, labels, padding_index=0)
            )
        if preds_item_ids_scores_host is not None:
            preds_item_ids_scores = nested_numpify(preds_item_ids_scores_host)
            all_preds_item_ids_scores = (
                preds_item_ids_scores
                if all_preds_item_ids_scores is None
                else nested_concat(
                    all_preds_item_ids_scores,
                    preds_item_ids_scores,
                )
            )
        # Get Number of samples :
        # the data loaders for this project do not return the dataset size,
        num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size
        # and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds_item_ids_scores is not None:
            all_preds_item_ids_scores = nested_truncate(all_preds_item_ids_scores, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Get metrics :
        metrics = {}
        # Computing the metrics results as the average of all steps
        if self.compute_metrics:
            streaming_metrics_results = model.compute_metrics(mode=metric_key_prefix)
            streaming_metrics_results_flattened = process_metrics(
                streaming_metrics_results, prefix=metric_key_prefix + "/"
            )

            metrics = {**metrics, **streaming_metrics_results_flattened}

        metrics[f"{metric_key_prefix}/loss"] = all_losses.mean().item()

        return EvalLoopOutput(
            predictions=all_preds_item_ids_scores,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_examples,
        )

    def _save_model_and_checkpoint(self, save_model_class=False):
        """
        Save the serialized model + trainer and random states.

        Parameters
        ----------
        save_model_class: Optional[bool]
            Whether to save the Model class or not.
            by default False
        """
        import os

        try:
            import cloudpickle
        except ImportError:
            cloudpickle = None

        logger.info("Saving model...")
        output_dir = os.path.join(
            self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )

        # save model parameters
        self._save_checkpoint(self.model, trial=None, metrics=None)
        # save the serialized model
        if save_model_class:
            # TODO : fix serialization of DatasetSchema object
            if cloudpickle is None:
                raise ValueError("cloudpickle is required to save model class")

            with open(os.path.join(output_dir, "model_class.pkl"), "wb") as out:
                cloudpickle.dump(self.model.module, out)

    def load_model_trainer_states_from_checkpoint(self, checkpoint_path, model=None):
        """
        This method loads the checkpoints states of the model, trainer and random states.
        If model is None the serialized model class is loaded from checkpoint.
        It does not loads the optimizer and LR scheduler states (for that call trainer.train()
        with resume_from_checkpoint argument for a complete load)

        Parameters
        ----------
        checkpoint_path: str
            Path to the checkpoint directory.
        model: Optional[Model]
            Model class used by Trainer. by default None
        """
        import os

        if model is None:
            try:
                import cloudpickle
            except ImportError:
                raise ImportError("cloudpickle is required to load model class")
            logger.info("Loading model class")
            model = cloudpickle.load(open(os.path.join(checkpoint_path, "model_class.pkl"), "rb"))

        self.model = HFWrapper(model)
        logger.info("Loading weights of previously trained model")
        # Restoring model weights
        self.model.load_state_dict(
            # torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin"))
            torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
        )
        # Restoring random state
        rng_file = os.path.join(checkpoint_path, "rng_state.pth")
        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
        # Restoring AMP scaler
        if self.use_amp:
            self.scaler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scaler.pt")))

    @property
    def log_predictions_callback(self) -> Callable:
        return self.__log_predictions_callback

    @log_predictions_callback.setter
    def log_predictions_callback(self, var: Callable):
        self.__log_predictions_callback = var

    def _maybe_log_predictions(
        self,
        labels: torch.Tensor,
        pred_item_ids: torch.Tensor,
        pred_item_scores: torch.Tensor,
        metrics: Dict[str, np.ndarray],
        metric_key_prefix: str,
    ):
        """
        If --log_predictions is enabled, calls a callback function to
        log predicted item ids, scores, metadata and metrics.
        Parameters
        ----------
        labels: torch.Tensor
            True labels.
        pred_item_ids: torch.Tensor
            The predicted items ids. if top_k is set:
            we return to top-k items for each
            next-item prediction.
        pred_item_scores: torch.Tensor
            The prediction scores, if top_k is set:
            we return to top-k predictions for each
            next-item prediction.
        metrics: Dict[str, np.ndarray]
            Dictionary of metrics computed by Model.
        metric_key_prefix: str
            Prefix to use when logging evaluation metrics.
            by default `eval`
        """
        # TODO Add pred_metadata: Dict[str, torch.Tensor],

        if self.args.log_predictions and self.log_predictions_callback is not None:
            # Converting torch Tensors to NumPy and callback predictions logging function
            # preds_metadata = {k: v.cpu().numpy() for k, v in pred_metadata.items()}

            self.log_predictions_callback(
                labels=labels.cpu().numpy(),
                pred_item_ids=pred_item_ids.cpu().numpy(),
                pred_item_scores=pred_item_scores.cpu()
                .numpy()
                .astype(np.float32),  # Because it is float16 when --fp16
                # preds_metadata=preds_metadata,
                metrics=metrics,
                dataset_type=metric_key_prefix,
            )

    def _increment_past_global_steps(self, current_global_step: int):
        self.past_global_steps += current_global_step

    def _get_general_global_step(self) -> int:
        general_global_step = self.past_global_steps
        if self.model.training:
            general_global_step += self.state.global_step

        return general_global_step

    def log(self, logs: Dict[str, float]) -> None:
        # Ensuring that eval metrics are prefixed as "eval_" so that the HF integration loggers
        # do not prefix metrics names with 'train/' (as 'train/' is always added when not eval)
        logs = {re.sub("^eval/", "eval_", k).replace("train/", ""): v for k, v in logs.items()}

        if not self.incremental_logging:
            super().log(logs)
        else:
            # If Incremental logging is enabled, ensures that global steps are always
            # incremented after train() calls
            # so that metrics are logger with no overlap on W&B and Tensorboard
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)

            # As state.global_step is also used for the learning rate schedules,
            # we create a copy only for logging
            state_copy = deepcopy(self.state)
            state_copy.global_step = self._get_general_global_step()

            output = {**logs, **{"step": state_copy.global_step}}
            self.state.log_history.append(output)
            self.control = self.callback_handler.on_log(self.args, state_copy, self.control, logs)


def process_metrics(metrics, prefix="", to_cpu=True):
    metrics_proc = {}
    for root_key, root_value in metrics.items():
        if isinstance(root_value, dict):
            flattened_metrics = process_metrics(root_value, prefix=prefix, to_cpu=to_cpu)
            metrics_proc = {**metrics_proc, **flattened_metrics}
        else:
            value = root_value.cpu().numpy().item() if to_cpu else root_value
            metrics_proc[f"{prefix}{root_key}"] = value
    return metrics_proc


class IncrementalLoggingCallback(TrainerCallback):
    """
    An :class:`~transformers.TrainerCallback` that changes the state of the Trainer
    on specific hooks for the purpose of the incremental logging
    Parameters
    ----------
    trainer: Trainer
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


class DatasetMock(Dataset, Sized):
    """
    Mock to inform HF Trainer that the dataset is sized,
    and can be obtained via the generated/provided data loader
    """

    def __init__(self, nsteps=1):
        self.nsteps = nsteps

    def __len__(self):
        return self.nsteps


class HFWrapper(torch.nn.Module):
    """
    Prepare the signature of the forward method
    as required by HF Trainer
    """

    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        inputs = kwargs
        return self.module(inputs, *args)
