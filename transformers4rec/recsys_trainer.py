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
import gc
import math
from collections.abc import Sized
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import (
    AdamW,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    is_torch_tpu_available,
)
from transformers.integrations import is_fairscale_available
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import DistributedTensorGatherer, nested_concat
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging

from .evaluation.recsys_metrics import EvalMetrics

# if is_fairscale_available():
#    from fairscale.optim import OSS

logger = logging.get_logger(__name__)


class DatasetType(Enum):
    train = "train"
    eval = "eval"


# Mock to inform HF Trainer that the dataset is sized, and can be obtained via the data loader
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


class RecSysTrainer(Trainer):
    """
    An :class:`~transformers.Trainer` specialized for sequential recommendation 
    including (session-based and session-aware recommendation)
    """

    def __init__(self, model_args, data_args, *args, **kwargs):
        self.past_global_steps = 0
        self.model_args = model_args
        self.data_args = data_args

        self.create_metrics()

        recsys_callback = RecSysTrainerCallback(self)

        mock_dataset = DatasetMock()
        super(RecSysTrainer, self).__init__(
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            callbacks=[recsys_callback],
            *args,
            **kwargs,
        )

    def _increment_past_global_steps(self, current_global_step: int):
        self.past_global_steps += current_global_step

    def _get_general_global_step(self) -> int:
        general_global_step = self.past_global_steps
        if self.model.training:
            general_global_step += self.state.global_step

        return general_global_step

    def create_metrics(self):
        """
        Instantiates streaming metrics (updated each step during the evaluation loop)
        """
        self.streaming_metrics = {}
        for dataset_type in DatasetType:
            # TODO: Make the metric top-k a hyperparameter
            self.streaming_metrics[dataset_type.value] = EvalMetrics(ks=[10, 20, 1000])

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        return self.eval_dataloader

    def set_train_dataloader(self, dataloader: DataLoader):
        self.train_dataloader = dataloader

    def set_eval_dataloader(self, dataloader: DataLoader):
        self.eval_dataloader = dataloader

    @property
    def log_attention_weights_callback(self) -> Callable:
        return self.__log_attention_weights_callback

    @log_attention_weights_callback.setter
    def log_attention_weights_callback(self, var: Callable):
        self.__log_attention_weights_callback = var

    @property
    def log_predictions_callback(self) -> Callable:
        return self.__log_predictions_callback

    @log_predictions_callback.setter
    def log_predictions_callback(self, var: Callable):
        self.__log_predictions_callback = var

    def reset_lr_scheduler(self) -> None:
        """ 
        Resets the LR scheduler of the previous :obj:`Trainer.train()` call, 
        so that a new LR scheduler one is created by the next :obj:`Trainer.train()` call.
        This is important for LR schedules like `get_linear_schedule_with_warmup()` which decays LR to 0 in the end of the train
        """
        self.lr_scheduler = None

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
        return len(dataloader) * dataloader.batch_size

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Overriding :obj:`Trainer.create_optimizer_and_scheduler()` to provide
        flexibility in the optimizers and learning rate schedule choices, as hyperparams
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            # if self.sharded_dpp:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=AdamW,
            #         lr=self.args.learning_rate,
            #         betas=(self.args.adam_beta1, self.args.adam_beta2),
            #         eps=self.args.adam_epsilon,
            #     )
            # else:
            #     self.optimizer = AdamW(
            #         optimizer_grouped_parameters,
            #         lr=self.args.learning_rate,
            #         betas=(self.args.adam_beta1, self.args.adam_beta2),
            #         eps=self.args.adam_epsilon,
            #     )

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        if self.lr_scheduler is None:
            if self.args.learning_rate_schedule == "constant_with_warmup":
                self.lr_scheduler = get_constant_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.args.learning_rate_warmup_steps,
                )
            elif self.args.learning_rate_schedule == "linear_with_warmup":
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.args.learning_rate_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            elif self.args.learning_rate_schedule == "cosine_with_warmup":
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.args.learning_rate_warmup_steps,
                    num_training_steps=num_training_steps,
                    num_cycles=self.args.learning_rate_num_cosine_cycles_by_epoch
                    * self.args.num_train_epochs,
                )
            else:
                raise ValueError(
                    "Invalid value for --learning_rate_schedule.  Valid values: constant_with_warmup | linear_with_warmup | cosine_with_warmup"
                )

    def log(self, logs: Dict[str, float]) -> None:
        """
        Overriding :obj:`Trainer.log()` to ensure that the global step
        used for logging to W&B and Tensorboard is incremental across multiple :obj:`Trainer.train()` methods
        so that the logged values do not overlap
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch

        # Incremental global steps across train() calls so that logs to W&B and Tensorboard do not overlap
        state_copy = deepcopy(self.state)
        state_copy.global_step = self._get_general_global_step()

        self.control = self.callback_handler.on_log(
            self.args, state_copy, self.control, logs
        )
        output = {**logs, **{"step": state_copy.global_step}}
        self.state.log_history.append(output)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Overriding :obj:`Trainer.prediction_loop()` (shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`) 
        to provide more flexibility to work with streaming metrics (computed at each eval batch) and
        to log with the outputs of the model (e.g. prediction scores, prediction metadata, attention weights)
        """

        # Ensures that metrics will be computed, even if self.compute_metrics function is not defined (because here we use streaming metrics)
        prediction_loss_only = False

        # Reseting streaming metrics for the dataset (Train, Valid or Test
        streaming_metrics_ds = self.streaming_metrics[metric_key_prefix]
        streaming_metrics_ds.reset()

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        # if self.args.n_gpu > 1 and not self.args.model_parallel:
        #    model = torch.nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size

        # num_examples = self.num_examples(dataloader)

        logger.info("***** Running %s *****", description)
        logger.info("  Batch size = %d", batch_size)

        preds_item_ids_scores_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        losses = []

        world_size = 1
        # if is_torch_tpu_available():
        #    world_size = xm.xrt_world_size()
        # elif self.args.local_rank != -1:
        #    world_size = torch.distributed.get_world_size()

        PADDING_INDEX = -100

        if not prediction_loss_only:

            if (
                metric_key_prefix == DatasetType.train.value
                and self.args.eval_steps_on_train_set
            ):
                num_examples = self.args.eval_steps_on_train_set * batch_size
            else:
                num_examples = self.num_examples(dataloader)

            logger.info("  Num sessions (examples) = %d", num_examples)

            if (
                not self.model_args.eval_on_last_item_seq_only
                and self.data_args.avg_session_length
            ):
                num_examples *= self.data_args.avg_session_length
                logger.info(
                    "  Num interactions (estimated by avg session length) = %d",
                    num_examples,
                )

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

        # if is_torch_tpu_available():
        #    dataloader = pl.ParallelLoader(
        #        dataloader, [self.args.device]
        #    ).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):

            # Limits the number of evaluation steps on train set (which is usually larger)
            if (
                metric_key_prefix == DatasetType.train.value
                and self.args.eval_steps_on_train_set > 0
                and step + 1 > self.args.eval_steps_on_train_set
            ):
                break

            loss, preds, labels, outputs = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            losses.append(loss.item())

            if step % self.args.compute_metrics_each_n_steps == 0:

                # Updates metrics and returns detailed metrics if log_predictions=True
                metrics_results_detailed = None
                if streaming_metrics_ds is not None:
                    metrics_results_detailed = streaming_metrics_ds.update(
                        preds,
                        labels,
                        return_individual_metrics=self.args.log_predictions,
                    )

                self._maybe_log_attention_weights(
                    inputs, outputs["model_outputs"], metric_key_prefix, step
                )

            if labels is not None:
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )

            if preds is not None and self.args.predict_top_k > 0:
                # preds_sorted_item_scores, preds_sorted_item_ids = torch.sort(preds, axis=1, descending=True)

                # if self.args.predict_top_k > 0:
                # preds_sorted_item_scores = preds_sorted_item_scores[:, :self.args.predict_top_k]
                # preds_sorted_item_ids = preds_sorted_item_ids[:, :self.args.predict_top_k]

                preds_sorted_item_scores, preds_sorted_item_ids = torch.topk(
                    preds, k=self.args.predict_top_k, dim=-1
                )

                self._maybe_log_predictions(
                    labels,
                    preds_sorted_item_ids,
                    preds_sorted_item_scores,
                    outputs["pred_metadata"],
                    metrics_results_detailed,
                    metric_key_prefix,
                )

                # The output predictions will be a tuple with the ranked top-n item ids, and item recommendation scores
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

                # del preds_item_ids_scores

            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
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
                losses_host, preds_item_ids_scores_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if not prediction_loss_only:
            preds_item_ids_scores_gatherer.add_arrays(
                self._gather_and_numpify(
                    preds_item_ids_scores_host, "preds_item_ids_scores"
                )
            )
            labels_gatherer.add_arrays(
                self._gather_and_numpify(labels_host, "eval_label_ids")
            )

        preds_item_ids_scores = (
            preds_item_ids_scores_gatherer.finalize()
            if not prediction_loss_only
            else None
        )
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        # Truncating labels and predictions (because the last batch is usually not complete)
        valid_preds_mask = label_ids != PADDING_INDEX
        label_ids = label_ids[valid_preds_mask]
        if isinstance(preds_item_ids_scores, tuple):
            preds_item_ids_scores = tuple(
                [
                    pred_section[valid_preds_mask]
                    for pred_section in preds_item_ids_scores
                ]
            )
        else:
            preds_item_ids_scores = preds_item_ids_scores[valid_preds_mask]

        # if self.compute_metrics is not None and preds is not None and label_ids is not None:
        #    metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        # else:
        #    metrics = {}

        metrics = {}

        # Computing the metrics results as the average of all steps
        if streaming_metrics_ds is not None:
            streaming_metrics_results = streaming_metrics_ds.result()
            metrics = {**metrics, **streaming_metrics_results}

        metrics[f"{metric_key_prefix}_loss"] = np.mean(losses)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(
            predictions=preds_item_ids_scores, label_ids=label_ids, metrics=metrics
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[float],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, Any]],
    ]:
        """
        Overriding :obj:`Trainer.prediction_step()` to provide more flexibility to unpack results from the model,
        like returning labels that are not exactly one input feature
        """

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

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

        other_outputs = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in outputs.items()
            if k not in ignore_keys + ["loss", "predictions", "labels"]
        }

        return (loss, predictions, labels, other_outputs)

    def _maybe_log_attention_weights(
        self,
        inputs: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        metric_key_prefix: str,
        prediction_step: int,
    ):
        """
        If --log_attention_weights is enabled, calls a callback function to log attention weights along with the input features
        """

        # Logs the attention weights
        if self.args.log_attention_weights and isinstance(
            self.model.model, PreTrainedModel
        ):

            if self.log_attention_weights_callback is not None:

                step_attention_weights = model_outputs[-1]
                assert (
                    len(step_attention_weights[0].shape) == 4
                ), "Attention weights tensor should be rank 4, with shape (batch_size, heads, seqlen, seqlen)"

                layers_step_attention_weights_cpu = list(
                    [layer_att.cpu().numpy() for layer_att in step_attention_weights]
                )

                # Converting torch Tensors to NumPy and callback predictions logging function
                inputs_cpu = {k: v.cpu().numpy() for k, v in inputs.items()}

                log_step = self._get_general_global_step() + prediction_step
                self.log_attention_weights_callback(
                    inputs=inputs_cpu,
                    att_weights=layers_step_attention_weights_cpu,
                    description="attention_{}_step_{:06}".format(
                        metric_key_prefix, log_step
                    ),
                )

    def _maybe_log_predictions(
        self,
        labels: torch.Tensor,
        pred_item_ids: torch.Tensor,
        pred_item_scores: torch.Tensor,
        pred_metadata: Dict[str, torch.Tensor],
        metrics: Dict[str, np.ndarray],
        metric_key_prefix: str,
    ):
        """
        If --log_predictions is enabled, calls a callback function to log predicted item ids, scores, metadata and metrics
        """

        if self.args.log_predictions and self.log_predictions_callback is not None:

            # Converting torch Tensors to NumPy and callback predictions logging function
            preds_metadata = {k: v.cpu().numpy() for k, v in pred_metadata.items()}

            self.log_predictions_callback(
                labels=labels.cpu().numpy(),
                pred_item_ids=pred_item_ids.cpu().numpy(),
                pred_item_scores=pred_item_scores.cpu()
                .numpy()
                .astype(np.float32),  # Because it is float16 when --fp16
                preds_metadata=preds_metadata,
                metrics=metrics,
                dataset_type=metric_key_prefix,
            )

    def wipe_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
