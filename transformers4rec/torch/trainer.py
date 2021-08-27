import collections
import gc
import inspect
from collections.abc import Sized
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Trainer
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import DistributedTensorGatherer, nested_concat
from transformers.trainer_utils import EvalLoopOutput, SchedulerType
from transformers.utils import logging

from ..config.trainer import T4RecTrainingArguments
from .model import Model

logger = logging.get_logger(__name__)


class Trainer(Trainer):
    """
    An :class:`~transformers.Trainer` specialized for sequential recommendation
    including (session-based and sequtial recommendation)
    """

    def __init__(
        self,
        model: Model,
        args: T4RecTrainingArguments,
        train_dataloader: Optional[Dataset] = None,
        eval_dataloader: Optional[Dataset] = None,
        compute_metrics: Optional[bool] = None,
        **kwargs,
    ):
        """
        Parameters:
        -----------
            #TODO
        """

        self.past_global_steps = 0

        t4rec_callback = T4RecTrainerCallback(self)
        mock_dataset = DatasetMock()
        hf_model = HFWrapper(model)

        super(Trainer, self).__init__(
            model=hf_model,
            args=args,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            callbacks=[t4rec_callback],
            **kwargs,
        )
        self.compute_metrics = compute_metrics

        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        return self.eval_dataloader

    def set_train_dataloader(self, dataloader: DataLoader):
        # Check that values are consistent between data-loader
        # and TrainingArg class
        assert (
            dataloader._batch_size == self.args.per_device_train_batch_size
        ), "batch size of dataloader {} should match ".format(dataloader._batch_size)
        "train batch size of T4RecTrainingArguments {}".format(
            self.args.per_device_train_batch_size
        )

        assert (
            dataloader.drop_last == self.args.dataloader_drop_last
        ), "Make sure drop_last is set to '{}' ".format(dataloader.drop_last)
        "in dataloader.drop_last and T4RecTrainingArguments.dataloader_drop_last"

        self.train_dataloader = dataloader

    def set_eval_dataloader(self, dataloader: DataLoader):
        assert (
            dataloader._batch_size == self.args.per_device_eval_batch_size
        ), "batch size of dataloader {} should match ".format(dataloader._batch_size)
        "eval batch size of T4RecTrainingArguments {}".format(self.args.per_device_eval_batch_size)

        self.eval_dataloader = dataloader

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

    def _increment_past_global_steps(self, current_global_step: int):
        self.past_global_steps += current_global_step

    def _get_general_global_step(self) -> int:
        general_global_step = self.past_global_steps
        if self.model.training:
            general_global_step += self.state.global_step
        return general_global_step

    def reset_lr_scheduler(self) -> None:
        """
        Resets the LR scheduler of the previous :obj:`Trainer.train()` call,
        so that a new LR scheduler one is created by the next :obj:`Trainer.train()` call.
        This is important for LR schedules like `get_linear_schedule_with_warmup()`
        which decays LR to 0 in the end of the train
        """
        self.lr_scheduler = None

    def create_scheduler(self, num_training_steps: int):
        # flexibility in scheduler with num_cycles as hyperparams
        if self.lr_scheduler is None:
            self.lr_scheduler = self.get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.args.learning_rate_num_cosine_cycles_by_epoch
                * self.args.num_train_epochs,
            )

    # What if we override the method get_scheduler to accept num_cycle params ?
    # The advantage is to use the unified HF API offering a variety
    # of scheduler
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
        Args:
            name (:obj:`str` or `:obj:`SchedulerType`):
                The name of the scheduler to use.
            optimizer (:obj:`torch.optim.Optimizer`):
                The optimizer that will be used during training.
            num_warmup_steps (:obj:`int`, `optional`):
                The number of warmup steps to do. This is not required by all schedulers
                (hence the argument being optional),
                the function will raise an error if it's unset and the scheduler type requires it.
            num_training_steps (:obj:`int`, `optional`):
                The number of training steps to do. This is not required by all schedulers
                (hence the argument being optional),
                the function will raise an error if it's unset and the scheduler type requires it.
            num_cycles: (:obj:`int`, `optional`):
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
        model: List[torch.nn.Module],
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
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            loss = outputs["loss"].mean().detach()

        if prediction_loss_only:
            return (loss, None, None, None)

        predictions = outputs["predictions"].detach()
        labels = outputs["labels"].detach()

        # TODO: define metadata dict in the model
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
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding :obj:`Trainer.prediction_loop()`
        (shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`)
        to provide more flexibility to work with streaming metrics
        (computed at each eval batch) and
        to log with the outputs of the model
        (e.g. prediction scores, prediction metadata, attention weights)
        """
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        # set the model
        model = self.model
        # reset metrics for the dataset (Train, Valid or Test)
        if self.compute_metrics:
            model._model.reset_metrics()

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")

        batch_size = dataloader._batch_size

        logger.info("***** Running %s *****", description)
        logger.info("  Batch size = %d", batch_size)

        preds_item_ids_scores_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        losses = []

        # TODO support distributed prediction loop
        world_size = 1

        if metric_key_prefix == "train" and self.args.eval_steps_on_train_set:
            num_examples = self.args.eval_steps_on_train_set * batch_size
        else:
            num_examples = self.num_examples(dataloader)

        logger.info("  Num sessions (examples) = %d", num_examples)

        PADDING_INDEX = -100  # TODO get padding_index from model
        preds_item_ids_scores_gatherer = DistributedTensorGatherer(
            world_size,
            num_examples,
            make_multiple_of=batch_size,
            padding_index=PADDING_INDEX,
        )
        labels_gatherer = DistributedTensorGatherer(
            world_size,
            num_examples,
            make_multiple_of=batch_size,
            padding_index=PADDING_INDEX,
        )

        model.eval()

        if self.args.past_index >= 0:
            self._past = None
        self.callback_handler.eval_dataloader = dataloader

        # Iterate over dataloader
        for step, inputs in enumerate(dataloader):
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

            losses.append(loss.item())

            # Updates metrics
            # TODO: compute metrics each N eval_steps to speedup evaluation
            metrics_results_detailed = None
            if self.compute_metrics:
                metrics_results_detailed = model._model.calculate_metrics(
                    inputs, labels, mode=metric_key_prefix
                )

            if labels is not None:
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=PADDING_INDEX)
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
                        padding_index=-100,
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
                if not prediction_loss_only:
                    preds_item_ids_scores_gatherer.add_arrays(
                        self._gather_and_numpify(
                            preds_item_ids_scores_host, "preds_item_ids_scores"
                        )
                    )
                    labels_gatherer.add_arrays(
                        self._gather_and_numpify(labels_host, "eval_label_ids")
                    )

                # Set back to None to begin a new accumulation
                preds_item_ids_scores_host, labels_host = None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if not prediction_loss_only:
            preds_item_ids_scores_gatherer.add_arrays(
                self._gather_and_numpify(preds_item_ids_scores_host, "preds_item_ids_scores")
            )
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

            preds_item_ids_scores = (
                preds_item_ids_scores_gatherer.finalize() if not prediction_loss_only else None
            )
            label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
            if label_ids is not None:
                # Truncating labels and predictions (because the last batch is usually not complete)
                valid_preds_mask = label_ids != PADDING_INDEX
                label_ids = label_ids[valid_preds_mask]
            if isinstance(preds_item_ids_scores, tuple):
                preds_item_ids_scores = tuple(
                    [pred_section[valid_preds_mask] for pred_section in preds_item_ids_scores]
                )
            else:
                preds_item_ids_scores = preds_item_ids_scores[valid_preds_mask]

        metrics = {}
        # Computing the metrics results as the average of all steps
        if self.compute_metrics:
            streaming_metrics_results = model._model.compute_metrics(mode=metric_key_prefix)
            metrics = {**metrics, **streaming_metrics_results}
        metrics[f"{metric_key_prefix}_loss"] = np.mean(losses)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key).cpu().numpy()

        return EvalLoopOutput(
            predictions=preds_item_ids_scores,
            label_ids=label_ids,
            metrics=metrics,
            num_samples=num_examples,
        )

    def _save_model_checkpoint(rec_model, trial=None, metrics=None):
        """
        Save the serialized model + optimizer states
        """
        """
        Previsous code :  Only state dict is saved  ==>  The HF model class is known before hand
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        """
        # TODO: serialize the model class and save it as well
        # torch.save(dict(model=model, model_state=model.state_dict()), "/path/model.tar")
        pass

    def load_model_trainer_states_from_checkpoint(checkpoint_path, rec_model, trainer):
        """
        This method loads from checkpoints states of the model, trainer and random states.
        It does not loads the optimizer and LR scheduler states (for that call trainer.train()
        with resume_from_checkpoint argument for a complete load)
        """
        # TODO : load serialized model and its states.
        pass

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
        # pred_metadata: Dict[str, torch.Tensor],
        metrics: Dict[str, np.ndarray],
        metric_key_prefix: str,
    ):
        """
        If --log_predictions is enabled, calls a callback function to
        log predicted item ids, scores, metadata and metrics
        """

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

    def log(self, logs: Dict[str, float]) -> None:
        """
        Overriding :obj:`Trainer.log()` to ensure that the global step
        used for logging to W&B and Tensorboard is incremental
        across multiple :obj:`Trainer.train()` methods
        so that the logged values do not overlap
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch

        # Incremental global steps across train() calls so that
        # logs to W&B and Tensorboard do not overlap
        state_copy = deepcopy(self.state)
        state_copy.global_step = self._get_general_global_step()

        self.control = self.callback_handler.on_log(self.args, state_copy, self.control, logs)
        output = {**logs, **{"step": state_copy.global_step}}
        self.state.log_history.append(output)

    def wipe_memory(self):
        gc.collect()
        torch.cuda.empty_cache()


# Mock to inform HF Trainer that the dataset is sized, and can be obtained via the data loader
# This is needed because we are decoupling dataloading from the trainer
class DatasetMock(Dataset, Sized):
    def __init__(self, nsteps=1):
        self.nsteps = nsteps

    def __len__(self):
        return self.nsteps


class T4RecTrainerCallback(TrainerCallback):
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


class HFWrapper(torch.nn.Module):
    """
    Prepare the signature of the forward method
    as required by HF Trainer
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, *args, **kwargs):
        inputs = kwargs
        return self._model(inputs, *args)
