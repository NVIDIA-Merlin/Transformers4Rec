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
import random

from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from recsys_utils import safe_json
from recsys_args import DataArguments, ModelArguments, TrainingArguments
from recsys_data import (
    fetch_data_loader,
    get_avail_data_dates    
)

from baselines.knn.vsknn import VMContextKNN

from transformers import HfArgumentParser


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

def get_parquet_files_names(data_path, date, is_train, eval_on_test_set=False):
    date_format="%Y-%m-%d"

    if is_train:
        parquet_paths = ["session_start_date={}-train.parquet".format(date.strftime(date_format))]
    else:
        if eval_on_test_set:
            eval_dataset_type = 'test'
        else:
            eval_dataset_type = 'valid'
        parquet_paths = ["session_start_date={}-{}.parquet".format(date.strftime(date_format), eval_dataset_type)]

    parquet_paths = [os.path.join(data_path, parquet_file) for parquet_file in parquet_paths]

    ##If paths are folders, list the parquet file within the folders
    #parquet_paths = get_filenames(parquet_paths, files_filter_pattern="*.parquet"
    

    return parquet_paths


def get_dataloaders(data_args, training_args, train_data_paths, eval_data_paths, feature_map):
    train_loader = fetch_data_loader(data_args, training_args, feature_map, train_data_paths, is_train_set=True)
    eval_loader = fetch_data_loader(data_args, training_args, feature_map, eval_data_paths, is_train_set=False)

    return train_loader, eval_loader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)

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

    
    DLLogger.init(backends=[
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, os.path.join(training_args.output_dir, DLLOGGER_FILENAME)),
    ])
    
    DLLogger.log(step="PARAMETER", 
                 data=all_hparams, 
                 verbosity=Verbosity.DEFAULT)
    DLLogger.flush()

    set_seed(training_args.seed)

    
    '''
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
    '''
    

    algorithm = VMContextKNN(k=500, sample_size=5000,  weighting='quadratic', weighting_score='quadratic', idf_weighting=5,
            sampling='recent', similarity='cosine',
            dwelling_time=False, last_n_days=None, last_n_clicks=None, remind=True, push_reminders=False, add_reminders=False, 
            extend=False, weighting_time=False, normalize=True, idf_weighting_session=False, 
            session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' )


    data_dates = get_avail_data_dates(data_args)
    results_dates = {}

    for date_idx in range(1, len(data_dates)):
        train_date, eval_date = data_dates[date_idx - 1], data_dates[date_idx]
        train_date_str, eval_date_str = train_date.strftime("%Y-%m-%d"), eval_date.strftime("%Y-%m-%d")

        train_data_paths = get_parquet_files_names(data_args.data_path, train_date, is_train=True)
        eval_data_paths = get_parquet_files_names(data_args.data_path, eval_date, is_train=False, 
                                                  eval_on_test_set=training_args.eval_on_test_set)
 
                
        # Training
        if training_args.do_train:
            logger.info("************* Training (date:{}) *************".format(train_date_str))

            train_df = prepare_data(train_data_paths, user_id_fname='user_idx', item_id_fname='sess_pid_seq', timestamp_fname='sess_etime_seq')
            algorithm.fit(train_df)

            eval_df = prepare_data(eval_data_paths, user_id_fname='user_idx', item_id_fname='sess_pid_seq', timestamp_fname='sess_etime_seq')            

            #Adapt the evaluate_sessions() code from Transf4Rec to loop over items and call predict_next()
            #algorithm.predict_next(self, session_id, input_item_id, predict_for_item_ids, timestamp=0, skip=False, mode_type='view')

            #Include CPU-based metric using EvalMetrics(use_cpu=True)
            


        # Evaluation
        if training_args.do_eval:            

            logger.info("************* Evaluation *************")

            # Loading again the data loaders, because some data loaders (e.g. NVTabular do not reset after they are not totally iterated over)
            train_loader, eval_loader = get_dataloaders(data_args, training_args, train_data_paths, eval_data_paths, feature_map)


            
            logger.info(f'Evaluating on test set ({eval_date_str})....')

            results_dates[eval_date] = eval_metrics

    logger.info("Training and evaluation loops are finished")
    
     
    if trainer.is_world_process_zero():
        
        if training_args.do_eval:    
            logger.info("Computing and loging AOD metrics")   
            results_df= pd.DataFrame.from_dict(results_dates, orient='index')
            results_df.reset_index().to_csv(os.path.join(training_args.output_dir, "eval_train_results_dates.csv"), index=False) 

            # Computing Average Over Days (AOD) metrics
            results_avg_days = dict(results_df.mean())
            # Logging to W&B
            trainer.log({f"{k}_AOD":v for k, v in results_avg_days.items()})

            log_aod_metric_results(training_args.output_dir, results_df, results_avg_days)    


def prepare_data(train_data_paths, user_id_fname, item_id_fname, timestamp_fname):
    dataframes = []
    for parquet_file in train_data_paths:
        df = pd.read_parquet(parquet_file, columns=[user_id_fname, item_id_fname, timestamp_fname])
        dataframes.append(df)
    
    concat_df = pd.concat(dataframes)

    #Generating contiguous item ids
    concat_df['SessionId'] = range(0, len(concat_df))

    # Converts from the representation of one row per session to one row by interaction
    data_df = explode_multiple_cols(concat_df, [item_id_fname, timestamp_fname])    

    data_df = data_df.rename({user_id_fname: 'UserId',
                    item_id_fname: 'ItemId',
                    timestamp_fname: 'Time'}, axis=1)

    return data_df
        
def explode_multiple_cols(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]    

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


if __name__ == "__main__":
    main()      