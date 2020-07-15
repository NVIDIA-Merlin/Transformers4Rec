"""
Extend Huggingface's Trainer Class to make it work with custom dataloader
for sequential dataset packed with different format in multiple sequences 
(e.g., item-id-seq, elapsed-time-seq) in parquet file format
"""

import logging
import math
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from transformers.data.data_collator import DataCollator, default_data_collator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput, is_wandb_available
from transformers.training_args import TrainingArguments, is_torch_tpu_available
from transformers import Trainer

from recsys_metrics import EvalPredictionTensor

logger = logging.getLogger(__name__)

softmax = nn.Softmax(dim=-1)

class RecSysTrainer(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """
    def __init__(self, *args, **kwargs):
        
        if 'f_feature_extract' not in kwargs: 
            self.f_feature_extract = lambda x: x
        else:
            self.f_feature_extract = kwargs.pop('f_feature_extract')

        if 'fast_test' not in kwargs:
            self.fast_test = False
        else:
            self.fast_test = kwargs.pop('fast_test')

        super(RecSysTrainer, self).__init__(*args, **kwargs)

    def get_rec_train_dataloader(self) -> DataLoader:
        if self.train_dataloader is not None:
            return self.train_dataloader
        return self.get_train_dataloader()            
        
    def get_rec_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if self.eval_dataloader is not None:
            return self.eval_dataloader
        return self.get_eval_dataloader(eval_dataset)

    def get_rec_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        if self.test_dataloader is not None:
            return self.test_dataloader
        return self.get_test_dataloader(test_dataset)

    def set_rec_train_dataloader(self, dataloader):
        self.train_dataloader = dataloader
        
    def set_rec_eval_dataloader(self, dataloader):
        self.eval_dataloader = dataloader

    def set_rec_test_dataloader(self, dataloader):
        self.test_dataloader = dataloader

    def num_examples(self, dataloader):
        return len(dataloader)

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        # NOTE: RecSys
        train_dataloader = self.get_rec_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        tr_acc = 0.0
        logging_loss = 0.0
        logging_acc = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        
        # NOTE: RecSys
        with train_dataloader:
            for epoch in train_iterator:
                if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)

                if is_torch_tpu_available():
                    parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                        self.args.device
                    )
                    epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
                else:
                    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

                for step, inputs in enumerate(epoch_iterator):
                    
                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    step_loss, step_acc = self._training_step(model, inputs, optimizer)
                    tr_loss += step_loss
                    tr_acc += step_acc

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if is_torch_tpu_available():
                            xm.optimizer_step(optimizer)
                        else:
                            optimizer.step()

                        scheduler.step()
                        model.zero_grad()
                        self.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                        if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                        ):
                            logs: Dict[str, float] = {}
                            logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                            logs["accuracy"] = (tr_acc - logging_acc) / self.args.logging_steps
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss = tr_loss
                            logging_acc = tr_acc

                            self._log(logs)

                            if self.args.evaluate_during_training:
                                self.evaluate()

                        if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                            # In all cases (even distributed/parallel), self.model is always a reference
                            # to the model we want to save.
                            if hasattr(model, "module"):
                                assert model.module is self.model
                            else:
                                assert model is self.model
                            # Save model checkpoint
                            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                            self.save_model(output_dir)

                            if self.is_world_master():
                                self._rotate_checkpoints()

                            if is_torch_tpu_available():
                                xm.rendezvous("saving_optimizer_states")
                                xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            elif self.is_world_master():
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                    if (self.args.max_steps > 0 and self.global_step > self.args.max_steps) or (self.fast_test and step > 4):
                        epoch_iterator.close()
                        break
                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    train_iterator.close()
                    break
                if self.args.tpu_metrics_debug:
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, tr_acc / self.global_step)        

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        _inputs = {}
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)
        
        # NOTE: RecSys
        outputs = model(*self.f_feature_extract(inputs))
        
        acc = outputs[0] # accuracy
        loss = outputs[1]  # model outputs are always tuple in transformers (see doc)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            acc = acc.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item(), acc.item()

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """

        # NOTE: RecSys
        eval_dataloader = self.get_rec_eval_dataloader()

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        cnt = 0
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():

                #NOTE: RecSys
                _inputs = self.f_feature_extract(inputs)
                labels = _inputs[0][:, 1:]
                
                outputs = model(*_inputs)

                step_eval_loss, logits = outputs[:2]
                eval_losses += [step_eval_loss.mean().item()]
            
            if not prediction_loss_only:
                # _preds.size(): N_BATCH x SEQLEN x ITEM_SIZE (=300000)
                _preds = logits.detach().unsqueeze(0)
                _preds = softmax(_preds)

                if preds is None:
                    preds = _preds
                else:
                    preds = torch.cat((preds, _preds), dim=0)
                if label_ids is None:
                    label_ids = labels.detach()
                else:
                    label_ids = torch.cat((label_ids, labels.detach()), dim=0)

            if self.fast_test and cnt > 4:
                break
            cnt += 1 

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            preds = preds.transpose(1,2).reshape(-1, preds.size(-3), preds.size(-1))
            metrics = self.compute_metrics(EvalPredictionTensor(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)