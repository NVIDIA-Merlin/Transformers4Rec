"""
Example arguments for command line: 
    CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false python recsys_main.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset_path/ --start_date 2019-10-01 --end_date 2019-10-15 --data_loader_engine nvtabular --per_device_train_batch_size 320 --per_device_eval_batch_size 512 --model_type gpt2 --loss_type cross_entropy --logging_steps 10 --d_model 256 --n_layer 2 --n_head 8 --dropout 0.1 --learning_rate 0.001 --similarity_type concat_mlp --num_train_epochs 1 --all_rescale_factor 1 --neg_rescale_factor 0 --feature_config ../datasets/ecommerce-large/config/features/session_based_features_pid.yaml --inp_merge mlp --tf_out_activation tanh --experiments_group local_test --weight_decay 1.3e-05 --learning_rate_schedule constant_with_warmup --learning_rate_warmup_steps 0 --learning_rate_num_cosine_cycles 1.25 --dataloader_drop_last --compute_metrics_each_n_steps 1 --hidden_act gelu_new --save_steps 0 --eval_on_last_item_seq_only --fp16 --overwrite_output_dir --session_seq_length_max 20 --predict_top_k 1000 --eval_accumulation_steps 10
"""

import os
import sys
import math
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from collections import Counter, namedtuple
import pyarrow
import pyarrow.parquet as pq
import wandb

from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from recsys_models import get_recsys_model
from recsys_meta_model import RecSysMetaModel
from recsys_trainer import RecSysTrainer, DatasetType, DatasetMock
from recsys_utils import safe_json
from recsys_args import DataArguments, ModelArguments, TrainingArguments
from recsys_data import (
    fetch_data_loaders,
    get_avail_data_dates    
)


import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import is_main_process
from transformers.integrations import WandbCallback

try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger(__name__)

from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import dllogger as DLLogger
DLLOGGER_FILENAME = 'log.json'

# this code use Version 3
assert sys.version_info.major > 2

PRED_LOG_PARQUET_FILE_PATTERN = 'pred_logs/preds_date_{}.parquet'
ATTENTION_LOG_FOLDER = 'attention_weights'


class AllHparams:
    """
    Used to aggregate all arguments in a single object for logging
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def to_sanitized_dict(self):
        result = {k:v for k,v in self.__dict__.items() if safe_json(v)}
        return result



def get_label_feature_name(feature_map: Dict[str, Any]) -> str:
    """
        Analyses the feature map config and returns the name of the label feature (e.g. item_id)
    """
    label_feature_config = list([k for k,v in feature_map.items() if 'is_label' in v and v['is_label']])

    if len(label_feature_config) == 0:
        raise ValueError('One feature have be configured as label (is_label = True)')
    if len(label_feature_config) > 1:
        raise ValueError('Only one feature can be selected as label (is_label = True)')
    label_name = label_feature_config[0]
    return label_name

def main():
  
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    #Ensuring to set W&B run name to null, so that a nice run name is generated
    training_args.run_name = None

    all_hparams = {**asdict(data_args), **asdict(model_args), **asdict(training_args)}
    all_hparams = {k:v for k,v in all_hparams.items() if safe_json(v)}

    #Loading features config file
    with open(data_args.feature_config) as yaml_file:
        feature_map = yaml.load(yaml_file, Loader=yaml.FullLoader)

    label_name = get_label_feature_name(feature_map)
    #training_args.label_names = [label_name]
    target_size = feature_map[label_name]['cardinality']

    #Enables profiling with DLProf and Nsight Systems (slows down training)
    if training_args.pyprof:
        logger.info("Enabling PyProf for profiling, to inspect with DLProf and Nsight Sytems. This will slow down training and should be used only for debug purposes.", training_args)        
        import pyprof
        pyprof.init(enable_function_stack=True)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training, Model and Data parameters {all_hparams}")

    DLLogger.init(backends=[
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, os.path.join(training_args.output_dir, DLLOGGER_FILENAME)),
    ])
    
    DLLogger.log(step="PARAMETER", 
                 data=all_hparams, 
                 verbosity=Verbosity.DEFAULT)
    DLLogger.flush()

    set_seed(training_args.seed)

    seq_model, config = get_recsys_model(model_args, data_args, training_args, target_size)
    rec_model = RecSysMetaModel(seq_model, config, model_args, data_args, feature_map)

    #if training_args.model_parallel:
    #    rec_model = rec_model.to(training_args.device)

    trainer = RecSysTrainer(
        model=rec_model,        
        args=training_args
    )

    
    #Creating an object with all hparams and a method to get sanitized values (like DataClass), because the setup code for WandbCallback requires a DataClass and not a dict
    all_hparams = AllHparams(**all_hparams)
    #Enforcing init of W&B  before begin_train callback (where it is originally initiated)
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            callback.setup(all_hparams, trainer.state, trainer.model, reinit=False)

            #Saving Weights & Biases run name to DLLogger
            wandb_run_name = wandb.run.name
            DLLogger.log(step="PARAMETER", 
                        data={'wandb_run_name': wandb_run_name}, 
                        verbosity=Verbosity.DEFAULT)   

            break

    set_log_attention_weights_callback(trainer, training_args)
    

    data_dates = get_avail_data_dates(data_args)
    results_dates = {}

    for date_idx in range(1, len(data_dates)):
        train_date, eval_date = data_dates[date_idx - 1], data_dates[date_idx]
        train_date_str, eval_date_str = train_date.strftime("%Y-%m-%d"), eval_date.strftime("%Y-%m-%d")

        # Training
        if training_args.do_train:
            logger.info("************* Training (date:{}) *************".format(train_date_str))

            train_loader, eval_loader \
                 = fetch_data_loaders(data_args, training_args, feature_map, train_date, eval_date)


            trainer.set_train_dataloader(train_loader)
            trainer.set_eval_dataloader(eval_loader)

            model_path = (
                model_args.model_name_or_path
                if model_args.model_name_or_path is not None \
                    and os.path.isdir(model_args.model_name_or_path)
                else None
            )

            trainer.reset_lr_scheduler()
            trainer.train(model_path=model_path)


        # Evaluation
        if training_args.do_eval:
            set_log_predictions_callback(trainer, training_args, eval_date)

            logger.info("************* Evaluation *************")

            # Loading again the data loaders, because some data loaders (e.g. NVTabular do not reset after they are not totally iterated over)
            train_loader, eval_loader \
                 = fetch_data_loaders(data_args, training_args, feature_map, train_date, eval_date, shuffle_train_dataloader=False)

            logger.info(f'Evaluating on train set ({train_date_str})....')
            trainer.set_train_dataloader(train_loader)
            #Defining temporarily the the train data loader for evaluation
            trainer.set_eval_dataloader(train_loader)

            train_metrics = trainer.evaluate(metric_key_prefix=DatasetType.train.value)

            log_metric_results(training_args.output_dir, train_metrics, prefix=DatasetType.train.value, date=train_date_str)
            
            
            logger.info(f'Evaluating on test set ({eval_date_str})....')
            trainer.set_eval_dataloader(eval_loader)
            eval_metrics = trainer.evaluate(metric_key_prefix=DatasetType.eval.value)

            log_metric_results(training_args.output_dir, eval_metrics, prefix=DatasetType.eval.value, date=eval_date_str)


            results_dates[eval_date] = {**eval_metrics, **train_metrics}

    logger.info("Training and evaluation loops are finished")
    
     
    if trainer.is_world_process_zero():
        
        logger.info("Saving model...")   
        trainer.save_model()
        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
      
        if training_args.do_eval:    
            logger.info("Computing and loging AOD metrics")   
            results_df= pd.DataFrame.from_dict(results_dates, orient='index')
            results_df.reset_index().to_csv(os.path.join(training_args.output_dir, "eval_train_results_dates.csv"), index=False) 

            # Computing Average Over Days (AOD) metrics
            results_avg_days = dict(results_df.mean())
            # Logging to W&B
            trainer.log({f"{k}_AOD":v for k, v in results_avg_days.items()})

            log_aod_metric_results(training_args.output_dir, results_df, results_avg_days)              



def log_metric_results(output_dir, metrics, prefix, date):
    """
    Logs to a text file metric results for each day, in a human-readable format
    """
    output_file = os.path.join(output_dir, f"{prefix}_results_dates.txt")                    
    with open(output_file, "a") as writer:
        logger.info(f"***** {prefix} results ({date})*****")
        writer.write(f"\n***** {prefix} results ({date})*****\n")
        for key in sorted(metrics.keys()):
            logger.info("  %s = %s", key, str(metrics[key]))
            writer.write("%s = %s\n" % (key, str(metrics[key])))

    DLLogger.log(step=(date,), data=metrics, verbosity=Verbosity.VERBOSE)
    DLLogger.flush()

def log_aod_metric_results(output_dir, results_df, results_avg_days):
    """
    Logs to a text file the final metric results (average over days), in a human-readable format
    """
    output_eval_file = os.path.join(output_dir, "eval_results_avg_over_days.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results (avg over dates) *****")
        writer.write(f"\n***** Eval results (avg over dates) *****\n")
        for key in sorted(results_avg_days.keys()):
            logger.info("  %s = %s", key, str(results_avg_days[key]))
            writer.write("%s = %s\n" % (key, str(results_avg_days[key])))
       

    #Logging AOD metrics with DLLogger
    DLLogger.log(step=(), data=results_avg_days, verbosity=Verbosity.VERBOSE)
    DLLogger.flush()

    return results_avg_days

def set_log_attention_weights_callback(trainer, training_args):
    """
    Sets a callback in the :obj:`RecSysTrainer` to log the attention weights of Transformer models
    """
    log_attention_weights_callback = None
    if training_args.log_attention_weights:
        attention_output_path = os.path.join(training_args.output_dir, ATTENTION_LOG_FOLDER)
        logger.info('Will output attention weights (and inputs) logs to {}'.format(attention_output_path))
        att_weights_logger = AttentionWeightsLogger(attention_output_path)

        trainer.log_attention_weights_callback = att_weights_logger.log 

class AttentionWeightsLogger:
    """
    Manages the logging of Transformers' attention weights
    """
    def __init__(self, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

    def log(self, inputs, att_weights, description):
        filename = os.path.join(self.output_path, description+'.pickle')

        data = (inputs, att_weights)
        with open(filename, 'wb') as ouf:
            pickle.dump(data, ouf)
            ouf.close()


def set_log_predictions_callback(trainer, training_args, eval_date):
    """
    Sets a callback in the :obj:`RecSysTrainer` to log the predictions of the model, for each day
    """
    if training_args.log_predictions:
        output_preds_logs_path = os.path.join(training_args.output_dir, PRED_LOG_PARQUET_FILE_PATTERN.format(eval_date.strftime("%Y-%m-%d")))
        logger.info('Will output prediction logs to {}'.format(output_preds_logs_path))
        prediction_logger = PredictionLogger(output_preds_logs_path)
        trainer.log_predictions_callback = prediction_logger.log_predictions

class PredictionLogger:
    """
    Manages the logging of model predictions during evaluation
    """
    def __init__(self, output_parquet_path):
        self.output_parquet_path = output_parquet_path
        self.pq_writer = None

    def _create_pq_writer_if_needed(self, new_rows_df):
        if not self.pq_writer:
            new_rows_pa = pyarrow.Table.from_pandas(new_rows_df)
            # Creating parent folder recursively
            parent_folder = os.path.dirname(os.path.abspath(self.output_parquet_path))
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
            # Creating parquet file
            self.pq_writer = pq.ParquetWriter(self.output_parquet_path, new_rows_pa.schema)

    def _append_new_rows_to_parquet(self, new_rows_df):
        new_rows_pa = pyarrow.Table.from_pandas(new_rows_df)
        self.pq_writer.write_table(new_rows_pa)


    def log_predictions(self, labels, pred_item_ids, pred_item_scores, preds_metadata, metrics, dataset_type):
        num_predictions = preds_metadata[list(preds_metadata.keys())[0]].shape[0]
        new_rows = []
        for idx in range(num_predictions):
            row = {}
            row['dataset_type'] = dataset_type

            if metrics is not None:
                # Adding metrics all detailed results
                for metric in metrics:
                    row['metric_'+metric] = metrics[metric][idx]

            if labels is not None:
                row['relevant_item_ids'] = [labels[idx]]
                row['rec_item_ids'] = pred_item_ids[idx]
                row['rec_item_scores'] = pred_item_scores[idx]

            # Adding metadata features
            for feat_name in preds_metadata:
                row['metadata_'+feat_name] = [preds_metadata[feat_name][idx]]

            new_rows.append(row)

        new_rows_df = pd.DataFrame(new_rows)

        self._create_pq_writer_if_needed(new_rows_df)
        self._append_new_rows_to_parquet(new_rows_df)

    def close(self):
        if self.pq_writer:
            self.pq_writer.close()


if __name__ == "__main__":
    main()      