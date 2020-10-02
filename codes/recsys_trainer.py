"""
Extend Huggingface's Trainer Class to make it work with custom dataloader
for sequential dataset packed with different format in multiple sequences 
(e.g., item-id-seq, elapsed-time-seq) in parquet file format
"""

import logging
import os

from typing import Dict, List, Optional, NamedTuple, Callable
from enum import Enum

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
import wandb

from recsys_metrics import EvalMetrics

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, is_wandb_available
from transformers.training_args import is_torch_tpu_available
from transformers import Trainer, get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)

softmax = nn.Softmax(dim=-1)


class TrainOutputAcc(NamedTuple):
    global_step: int
    training_loss: float
    training_acc: float
    
class PredictionOutput(NamedTuple):
    metrics_all: Optional[Dict[str, float]]
    metrics_neg: Optional[Dict[str, float]]


class DatasetType(Enum):
    train = "Train"
    valid = "Valid"
    test = "Test"


class RecSysTrainer(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """
    def __init__(self, *args, **kwargs):
        self.global_step = 0

        if 'fast_test' not in kwargs:
            self.fast_test = False
        else:
            self.fast_test = kwargs.pop('fast_test')

        if 'log_predictions' not in kwargs:
            self.log_predictions = False
        else:
            self.log_predictions = kwargs.pop('log_predictions')

        self.create_metrics()
        
        super(RecSysTrainer, self).__init__(*args, **kwargs)


    def create_metrics(self):
        self.streaming_metrics_all = {}
        self.streaming_metrics_neg = {}
        for dataset_type in DatasetType:
            self.streaming_metrics_all[dataset_type] = EvalMetrics(ks=[5,10,100,1000])
            self.streaming_metrics_neg[dataset_type] = EvalMetrics(ks=[5,10])

    def get_rec_train_dataloader(self) -> DataLoader:
        if self.train_dataloader is not None:
            return self.train_dataloader
        return self.get_train_dataloader()            
        
    def get_rec_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def get_rec_test_dataloader(self) -> DataLoader:
        return self.test_dataloader

    def set_rec_train_dataloader(self, dataloader):
        self.train_dataloader = dataloader
        
    def set_rec_eval_dataloader(self, dataloader):
        self.eval_dataloader = dataloader

    def set_rec_test_dataloader(self, dataloader):
        self.test_dataloader = dataloader

    def num_examples(self, dataloader):
        return len(dataloader)

    def update_wandb_args(self, args):
        if is_wandb_available:
            wandb.config.update(args)

    def train(self, model_path: Optional[str] = None, log_attention_weights_fn: Callable = None):
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

        optimizer, _ = self.get_optimizers(num_training_steps=t_total)

        
        if self.args.learning_rate_schedule == 'constant_with_warmup':
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps = self.args.learning_rate_warmup_steps)
        elif self.args.learning_rate_schedule == 'linear_with_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = self.args.learning_rate_warmup_steps, num_training_steps=t_total)
        elif self.args.learning_rate_schedule == 'cosine_with_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = self.args.learning_rate_warmup_steps, num_training_steps=t_total,
                                                                   num_cycles= self.args.learning_rate_num_cosine_cycles)
        else:
            raise ValueError('Invalid value for --learning_rate_schedule.  Valid values: constant_with_warmup | linear_with_warmup | cosine_with_warmup')


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
        logger.info("  Num samples by epoch = %d", self.num_examples(train_dataloader) * self.args.train_batch_size)
        logger.info("  Num steps by epoch = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

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
                    epoch_iterator = tqdm(parallel_loader, desc="In-Epoch Iteration", disable=not self.is_local_master())
                else:
                    epoch_iterator = tqdm(train_dataloader, desc="In-Epoch Iteration", disable=not self.is_local_master())

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
                            logs["train_accuracy"] = (tr_acc - logging_acc) / self.args.logging_steps
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss = tr_loss
                            logging_acc = tr_acc

                            self._log(logs)

                            #if self.args.evaluate_during_training:
                            #    self.evaluate()

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

                    if (self.args.max_steps > 0 and self.global_step > self.args.max_steps) or (self.fast_test and step > 2):
                        epoch_iterator.close()
                        break
                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    train_iterator.close()
                    break
                if self.args.tpu_metrics_debug:
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                
                if self.args.validate_every > 0 and (epoch + 1) % self.args.validate_every == 0:
                    self._run_validation(log_attention_weights_fn=log_attention_weights_fn)

            # Compute metrics on Train set and Eval Set after all training epochs (for each day)
            self._run_validation(log_attention_weights_fn=log_attention_weights_fn)
            
        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutputAcc(self.global_step, tr_loss / self.global_step, tr_acc / self.global_step)        

    def _run_validation(self, log_attention_weights_fn: Callable = None):
        train_output_all, train_output_neg = self.evaluate(self.get_rec_train_dataloader(), DatasetType.train, log_attention_weights_fn = log_attention_weights_fn)
        valid_output_all, valid_output_neg = self.evaluate(self.get_rec_eval_dataloader(), DatasetType.valid, log_attention_weights_fn = log_attention_weights_fn)

        output_eval_file = os.path.join(self.args.output_dir, "valid_train_results.txt")
        if self.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info(f"*** Train results (all) (epoch: {self.epoch})***")
                writer.write(f"*** Train results (all) (epoch: {self.epoch})***")
                for key in sorted(train_output_all.keys()):
                    logger.info("  %s = %s", key, str(train_output_all[key]))
                    writer.write("%s = %s\n" % (key, str(train_output_all[key])))

                logger.info(f"*** Train results (neg) (epoch: {self.epoch})***")
                writer.write(f"*** Train results (neg) (epoch: {self.epoch})***")
                for key in sorted(train_output_neg.keys()):
                    logger.info("  %s = %s", key, str(train_output_neg[key]))
                    writer.write("%s = %s\n" % (key, str(train_output_neg[key])))

                logger.info(f"*** Validation results (all) (epoch: {self.epoch})***")
                writer.write(f"*** Validation results (all) (epoch: {self.epoch})***")
                for key in sorted(valid_output_all.keys()):
                    logger.info("  %s = %s", key, str(valid_output_all[key]))
                    writer.write("%s = %s\n" % (key, str(valid_output_all[key])))

                logger.info(f"*** Validation results (neg) (epoch: {self.epoch})***")
                writer.write(f"*** Validation results (neg) (epoch: {self.epoch})***")
                for key in sorted(valid_output_neg.keys()):
                    logger.info("  %s = %s", key, str(valid_output_neg[key]))
                    writer.write("%s = %s\n" % (key, str(valid_output_neg[key])))

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        _inputs = {}
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)
        
        # NOTE: RecSys
        outputs = model(inputs)
        
        acc = outputs[0] # accuracy
        loss = outputs[2]  # model outputs are always tuple in transformers (see doc)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            acc = acc.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps


        '''

        #l1_reg_factor = torch.tensor(1e-5)
        l2_reg_factor = torch.tensor(1e-3)    

        #L1_reg = torch.tensor(0., requires_grad=True)
        L2_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'embedding_tables' in name:
                #L1_reg = L1_reg + torch.norm(param, 1)
                L2_reg = L2_reg + torch.norm(param, 2)

        #L1_reg = L1_reg * l1_reg_factor
        L2_reg = L2_reg * l2_reg_factor

        self._log({#'l1_reg_loss': L1_reg,
                'l2_reg_loss': L2_reg.item(),
                'loss_before_reg': loss.item()})

        loss = loss + L2_reg

        '''

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item(), acc.item()

    def evaluate(
        self, eval_dataloader: Optional[DataLoader] = None, dataset_type: Optional[DatasetType] = DatasetType.valid,
        prediction_loss_only: Optional[bool] = None, log_attention_weights_fn: Callable = None
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataloader: (Optional) Pass a data loader if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """

        # NOTE: RecSys

        if eval_dataloader is None:
            eval_dataloader = self.get_rec_eval_dataloader()

        output = self._prediction_loop(eval_dataloader, dataset_type, prediction_loss_only=prediction_loss_only,
                                        log_attention_weights_fn=log_attention_weights_fn)

        self._log(output.metrics_all)
        self._log(output.metrics_neg)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics_all, output.metrics_neg

    def predict(self, test_dataloader: Optional[DataLoader] = None, dataset_type: Optional[DatasetType] = DatasetType.test,
                log_predictions_fn: Callable = None, log_attention_weights_fn: Callable = None) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        if test_dataloader is None:
            test_dataloader = self.get_rec_test_dataloader()

        output = self._prediction_loop(test_dataloader, dataset_type=dataset_type, 
                        log_predictions_fn=log_predictions_fn, log_attention_weights_fn=log_attention_weights_fn)

        self._log(output.metrics_neg)
        self._log(output.metrics_all)

        return output

    def _prediction_loop(
        self, dataloader: DataLoader, dataset_type: DatasetType, prediction_loss_only: Optional[bool] = None, 
        log_predictions_fn: Callable = None, log_attention_weights_fn: Callable = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        description = dataset_type.value
        streaming_metrics_all_ds = self.streaming_metrics_all[dataset_type]
        streaming_metrics_neg_ds = self.streaming_metrics_neg[dataset_type]

        # Reseting streaming metrics each day
        streaming_metrics_all_ds.reset()
        streaming_metrics_neg_ds.reset()


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
        logger.info("  Num steps = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        eval_losses_neg: List[float] = []
        eval_losses_ce: List[float] = []
        eval_accs: List[float] = []
        eval_accs_neg: List[float] = []
        step = 0
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():

                #NOTE: RecSys
                outputs = model(inputs)
                
                step_eval_acc, step_eval_acc_neg, step_eval_loss, step_eval_loss_neg, step_eval_loss_ce, preds_neg, labels_neg, preds_all, labels_all, preds_metadata = outputs[:10]
                
                step_attention_weights = None
                if self.args.log_attention_weights:
                    step_attention_weights = outputs[11]


                eval_accs += [step_eval_acc.mean().item()]                
                eval_losses += [step_eval_loss.mean().item()]                
                eval_losses_ce += [step_eval_loss_ce.mean().item()]
                if step_eval_acc_neg is not None:
                    eval_accs_neg += [step_eval_acc_neg.mean().item()]
                    eval_losses_neg += [step_eval_loss_neg.mean().item()]


                if not prediction_loss_only:

                     # preds.size(): N_BATCH x SEQLEN x (POS_Sample + NEG_Sample) (=51)
                    # labels.size(): ...  x 1 [51]
                    
                    if step % self.args.compute_metrics_each_n_steps == 0:

                        #Updates metrics and returns detailed metrics if log_predictions=True
                        metrics_results_detailed_all = None
                        metrics_results_detailed_neg = None
                        if streaming_metrics_all_ds is not None:
                            metrics_results_detailed_all = streaming_metrics_all_ds.update(preds_all, labels_all, return_individual_metrics=self.log_predictions)
                        if streaming_metrics_neg_ds is not None:
                            if preds_neg is not None:
                                metrics_results_detailed_neg = streaming_metrics_neg_ds.update(preds_neg, labels_neg, return_individual_metrics=self.log_predictions)


                        if self.args.log_attention_weights and \
                            isinstance(self.model.model, PreTrainedModel): #Checks if its a transformer                        

                            if log_attention_weights_fn is not None:
                                
                                step_attention_weights_cpu = list([layer_att.cpu().numpy() for layer_att in step_attention_weights])

                                #Converting torch Tensors to NumPy and callback predictions logging function
                                inputs_cpu = {k: v.cpu().numpy() for k, v in inputs.items()}

                                log_attention_weights_fn(inputs=inputs_cpu, 
                                                        att_weights=step_attention_weights_cpu, 
                                                        description='attention_{}_step_{:06}'.format(description, self.global_step))
                                               

                        if self.log_predictions:
                            #Converting torch Tensors to NumPy and callback predictions logging function
                            preds_metadata = {k: v.cpu().numpy() for k, v in preds_metadata.items()}

                            if log_predictions_fn is not None:                            
                            
                                preds_neg_values = None
                                labels_neg_values = None
                                if preds_neg is not None:
                                    preds_neg_values = preds_neg.cpu().numpy()
                                    labels_neg_values = labels_neg.cpu().numpy()
                                log_predictions_fn(preds_neg_values, labels_neg_values, 
                                                    metrics_results_detailed_neg, metrics_results_detailed_all, 
                                                    preds_metadata)
                        
            if self.fast_test and step > 4:
                break
            step += 1 

        if streaming_metrics_all_ds is not None:
            metrics_all = streaming_metrics_all_ds.result()
        else:
            metrics_all = {}

        if streaming_metrics_neg_ds is not None:
            metrics_neg = streaming_metrics_neg_ds.result()
        else:
            metrics_neg = {}

        if len(eval_losses) > 0:
            metrics_all[f"{description}_loss"] = np.mean(eval_losses)
        if len(eval_losses_ce) > 0:
            metrics_all[f"{description}_loss_xe"] = np.mean(eval_losses_ce)
        if len(eval_losses_neg) > 0:
            metrics_neg[f"{description}_loss_xe_neg"] = np.mean(eval_losses_neg)
        if len(eval_accs) > 0:
            metrics_all[f"{description}_accuracy"] = np.mean(eval_accs)
        if len(eval_accs_neg) > 0:
            metrics_all[f"{description}_accuracy_neg"] = np.mean(eval_accs_neg)

        # Prefix all keys with eval_
        for key in list(metrics_all.keys()):
            if not key.startswith(f"{description}_"):
                metrics_all[f"{description}_{key}_all"] = metrics_all.pop(key)

        for key in list(metrics_neg.keys()):
            if not key.startswith(f"{description}_"):
                metrics_neg[f"{description}_{key}_neg"] = metrics_neg.pop(key)
        
        return PredictionOutput(metrics_all=metrics_all, metrics_neg=metrics_neg)
