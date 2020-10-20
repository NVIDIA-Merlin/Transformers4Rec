"""
How torun : 
    CUDA_VISIBLE_DEVICES=0 python main_runner.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset/sessions_with_neg_samples_example/ --per_device_train_batch_size 128 --model_type xlnet
"""
import os
import sys
import math
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from collections import Counter
import pyarrow
import pyarrow.parquet as pq
import wandb

import yaml
from transformers import (
    HfArgumentParser,
    set_seed,
)

from recsys_models import get_recsys_model
from recsys_meta_model import RecSysMetaModel
from recsys_trainer import RecSysTrainer

from recsys_args import DataArguments, ModelArguments, TrainingArguments
from recsys_data import (
    fetch_data_loaders,
    get_avail_data_dates
)

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

def main():

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

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
    logger.info("Training/evaluation parameters %s", training_args)

    DLLogger.init(backends=[
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, os.path.join(training_args.output_dir, DLLOGGER_FILENAME)),
    ])

    hparams = {**asdict(data_args), **asdict(model_args), **asdict(training_args)}
    DLLogger.log(step="PARAMETER", 
                 data=hparams, 
                 verbosity=Verbosity.DEFAULT)
    DLLogger.flush()

    set_seed(training_args.seed)

    with open(data_args.feature_config) as yaml_file:
        feature_map = yaml.load(yaml_file, Loader=yaml.FullLoader)
    target_size = feature_map['sess_pid_seq']['cardinality']

    seq_model, config = get_recsys_model(model_args, data_args, training_args, target_size)
    rec_model = RecSysMetaModel(seq_model, config, model_args, data_args, feature_map)

    

    trainer = RecSysTrainer(
        model=rec_model,
        args=training_args,
        fast_test=training_args.fast_test,
        log_predictions=training_args.log_predictions
    )

    #Saving Weights & Biases run name
    wandb.run.save()
    wandb_run_name = wandb.run.name
    DLLogger.log(step="PARAMETER", 
                 data={'wandb_run_name': wandb_run_name}, 
                 verbosity=Verbosity.DEFAULT)                 

    trainer.update_wandb_args(model_args)
    trainer.update_wandb_args(data_args)

    data_dates = get_avail_data_dates(data_args)
    results_dates_all = {}
    results_dates_neg = {}

    att_weights_fn = None
    if training_args.log_attention_weights:
        attention_output_path = os.path.join(training_args.output_dir, ATTENTION_LOG_FOLDER)
        logger.info('Will output attention weights (and inputs) logs to {}'.format(attention_output_path))
        att_weights_logger = AttentionWeightsLogger(attention_output_path)
        att_weights_fn = att_weights_logger.log 

    for date_idx in range(1, len(data_dates)):
        train_date, eval_date, test_date = data_dates[date_idx - 1], data_dates[date_idx -1], data_dates[date_idx]

        train_loader, eval_loader, test_loader \
            = fetch_data_loaders(data_args, training_args, feature_map, train_date, eval_date, test_date)

        trainer.set_rec_train_dataloader(train_loader)
        trainer.set_rec_eval_dataloader(eval_loader)
        trainer.set_rec_test_dataloader(test_loader)

        # Training
        if training_args.do_train:
            logger.info("*** Train (date:{})***".format(train_date))

            model_path = (
                model_args.model_name_or_path
                if model_args.model_name_or_path is not None \
                    and os.path.isdir(model_args.model_name_or_path)
                else None
            )
            trainer.train(model_path=model_path, 
                          day_index=date_idx,
                         #log_attention_weights_fn=att_weights_fn
                         )

        # Evaluation (on testset)
        if training_args.do_eval:
            logger.info("*** Evaluate (date:{})***".format(test_date))

            # To log predictions in a parquet file for each day
            if training_args.log_predictions:
                output_preds_logs_path = os.path.join(training_args.output_dir, PRED_LOG_PARQUET_FILE_PATTERN.format(test_date.strftime("%Y-%m-%d")))
                logger.info('Will output prediction logs to {}'.format(output_preds_logs_path))
                prediction_logger = PredictionLogger(output_preds_logs_path)


            #if training_args.log_attention_weights:
            #    attention_output_path = os.path.join(training_args.output_dir, ATTENTION_LOG_FOLDER)
            #    logger.info('Will output attention weights (and inputs) logs to {}'.format(attention_output_path))
            #    att_weights_logger = AttentionWeightsLogger(attention_output_path)

            try:
                log_predictions_fn = prediction_logger.log_predictions if training_args.log_predictions else None
                #att_weights_fn = att_weights_logger.log if training_args.log_attention_weights else None
                
                eval_output = trainer.predict(log_predictions_fn=log_predictions_fn, log_attention_weights_fn=att_weights_fn)
                eval_metrics_all = eval_output.metrics_all
                eval_metrics_neg = eval_output.metrics_neg

                output_eval_file = os.path.join(training_args.output_dir, "eval_results_dates.txt")
                if trainer.is_world_master():
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results (all) (date:{})*****".format(test_date))
                        writer.write("***** Eval results (all) (date:{})*****".format(test_date))
                        for key in sorted(eval_metrics_all.keys()):
                            logger.info("  %s = %s", key, str(eval_metrics_all[key]))
                            writer.write("%s = %s\n" % (key, str(eval_metrics_all[key])))
                        
                        logger.info("***** Eval results (neg) (date:{})*****".format(test_date))
                        writer.write("***** Eval results (neg) (date:{})*****".format(test_date))
                        for key in sorted(eval_metrics_neg.keys()):
                            logger.info("  %s = %s", key, str(eval_metrics_neg[key]))
                            writer.write("%s = %s\n" % (key, str(eval_metrics_neg[key])))

                results_dates_all[test_date] = eval_metrics_all
                results_dates_neg[test_date] = eval_metrics_neg

                DLLogger.log(step=(test_date.strftime("%Y-%m-%d"),), data={**eval_metrics_all, **eval_metrics_neg}, verbosity=Verbosity.VERBOSE)
                DLLogger.flush()

            finally:
                if training_args.log_predictions:
                    prediction_logger.close()
        
    logger.info("train and eval for all dates are done")
    trainer.save_model()

    if training_args.do_eval and trainer.is_world_master():
        
        eval_df_all = pd.DataFrame.from_dict(results_dates_all, orient='index')
        eval_df_neg = pd.DataFrame.from_dict(results_dates_neg, orient='index')
        np.save(os.path.join(training_args.output_dir, "eval_results_dates_all.npy"), eval_df_all)
        np.save(os.path.join(training_args.output_dir, "eval_results_dates_neg.npy"), eval_df_neg)
        eval_avg_days_all = dict(eval_df_all.mean())
        eval_avg_days_neg = dict(eval_df_neg.mean())
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_avg.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results (all) (avg over dates)*****")
            for key in sorted(eval_avg_days_all.keys()):
                logger.info("  %s = %s", key, str(eval_avg_days_all[key]))
                writer.write("%s = %s\n" % (key, str(eval_avg_days_all[key])))
            trainer._log({f"AOD_all_{k}":v for k, v in eval_avg_days_all.items()})
            
            logger.info("***** Eval results (neg) (avg over dates)*****")
            for key in sorted(eval_avg_days_neg.keys()):
                logger.info("  %s = %s", key, str(eval_avg_days_neg[key]))
                writer.write("%s = %s\n" % (key, str(eval_avg_days_neg[key])))
            trainer._log({f"AOD_neg_{k}":v for k, v in eval_avg_days_neg.items()})

        #Logging with DLLogger for AutoBench
        eval_avg_metrics = {**eval_avg_days_all, **eval_avg_days_neg}
        DLLogger.log(step=(), data=eval_avg_metrics, verbosity=Verbosity.VERBOSE)
        DLLogger.flush()
                
    return results_dates_all


class AttentionWeightsLogger:

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



class PredictionLogger:

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

    def log_predictions(self, pred_scores_neg, labels_neg, metrics_neg, metrics_all, preds_metadata):
        num_predictions = preds_metadata[list(preds_metadata.keys())[0]].shape[0]
        new_rows = []
        for idx in range(num_predictions):
            row = {}

            if metrics_all is not None:
                # Adding metrics all detailed results
                for metric in metrics_all:
                    row['metric_all_'+metric] = metrics_all[metric][idx]

            if metrics_neg is not None:
                # Adding metrics neg detailed results
                for metric in metrics_neg:
                    row['metric_neg_'+metric] = metrics_neg[metric][idx]

            if labels_neg is not None:
                row['relevant_item_ids'] = [labels_neg[idx]]
                row['rec_item_scores'] = pred_scores_neg[idx]

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