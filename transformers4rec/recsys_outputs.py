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

import logging
import os
from dataclasses import asdict

import dllogger as DLLogger
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import wandb
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
from transformers.integrations import WandbCallback

from .recsys_utils import safe_json

try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger(__name__)

DLLOGGER_FILENAME = "log.json"
PRED_LOG_PARQUET_FILE_PATTERN = "pred_logs/preds_{:04}.parquet"
ATTENTION_LOG_FOLDER = "attention_weights"


def creates_output_dir(training_args):
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


def config_dllogger(output_dir):
    DLLogger.init(
        backends=[
            StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(
                Verbosity.VERBOSE, os.path.join(output_dir, DLLOGGER_FILENAME),
            ),
        ]
    )


def log_parameters(trainer, data_args, model_args, training_args):
    all_hparams = {
        **asdict(data_args),
        **asdict(model_args),
        **asdict(training_args),
    }
    all_hparams = {k: v for k, v in all_hparams.items() if safe_json(v)}

    logger.info(f"Training, Model and Data parameters {all_hparams}")

    DLLogger.log(step="PARAMETER", data=all_hparams, verbosity=Verbosity.DEFAULT)

    # Creating an object with all hparams and a method to get sanitized values (like DataClass),
    # because the setup code for WandbCallback requires a DataClass and not a dict
    all_hparams_aggregated = AllHparams(**all_hparams)
    # Enforcing init of W&B  before begin_train callback (where it is originally initiated)
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            callback.setup(
                all_hparams_aggregated, trainer.state, trainer.model, reinit=False
            )

            # Saving Weights & Biases run name to DLLogger
            wandb_run_name = wandb.run.name
            DLLogger.log(
                step="PARAMETER",
                data={"wandb_run_name": wandb_run_name},
                verbosity=Verbosity.DEFAULT,
            )

            break

    DLLogger.flush()


def log_metric_results(output_dir, metrics, prefix, time_index):
    """
    Logs to a text file metric results for each time window, in a human-readable format
    """
    output_file = os.path.join(output_dir, f"{prefix}_results_over_time.txt")
    with open(output_file, "a") as writer:
        logger.info(f"***** {prefix} results (time index): {time_index})*****")
        writer.write(f"\n***** {prefix} results (time index): {time_index})*****\n")
        for key in sorted(metrics.keys()):
            logger.info("  %s = %s", key, str(metrics[key]))
            writer.write("%s = %s\n" % (key, str(metrics[key])))

    DLLogger.log(step=(time_index,), data=metrics, verbosity=Verbosity.VERBOSE)
    DLLogger.flush()


def log_aot_metric_results(output_dir, results_avg_time):
    """
    Logs to a text file the final metric results (average over time), in a human-readable format
    """
    output_eval_file = os.path.join(output_dir, "eval_results_avg_over_time.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results (avg over time) *****")
        writer.write(f"\n***** Eval results (avg over time) *****\n")
        for key in sorted(results_avg_time.keys()):
            logger.info("  %s = %s", key, str(results_avg_time[key]))
            writer.write("%s = %s\n" % (key, str(results_avg_time[key])))

    # Logging AOT metrics with DLLogger
    DLLogger.log(step=(), data=results_avg_time, verbosity=Verbosity.VERBOSE)
    DLLogger.flush()

    return results_avg_time


def set_log_attention_weights_callback(trainer, training_args):
    """
    Sets a callback in the :obj:`RecSysTrainer` to log the attention weights of Transformer models
    """
    trainer.log_attention_weights_callback = None
    if training_args.log_attention_weights:
        attention_output_path = os.path.join(
            training_args.output_dir, ATTENTION_LOG_FOLDER
        )
        logger.info(
            f"Will output attention weights (and inputs) logs to {attention_output_path}"
        )
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
        filename = os.path.join(self.output_path, description + ".pickle")

        data = (inputs, att_weights)
        with open(filename, "wb") as ouf:
            pickle.dump(data, ouf)
            ouf.close()


def set_log_predictions_callback(trainer, training_args, time_index_eval):
    """
    Sets a callback in the :obj:`RecSysTrainer` to log the predictions of the model, for each time window
    """
    if training_args.log_predictions:
        output_preds_logs_path = os.path.join(
            training_args.output_dir,
            PRED_LOG_PARQUET_FILE_PATTERN.format(time_index_eval),
        )
        logger.info("Will output prediction logs to {}".format(output_preds_logs_path))
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
            self.pq_writer = pq.ParquetWriter(
                self.output_parquet_path, new_rows_pa.schema
            )

    def _append_new_rows_to_parquet(self, new_rows_df):
        new_rows_pa = pyarrow.Table.from_pandas(new_rows_df)
        self.pq_writer.write_table(new_rows_pa)

    def log_predictions(
        self,
        labels,
        pred_item_ids,
        pred_item_scores,
        preds_metadata,
        metrics,
        dataset_type,
    ):
        num_predictions = preds_metadata[list(preds_metadata.keys())[0]].shape[0]
        new_rows = []
        for idx in range(num_predictions):
            row = {}
            row["dataset_type"] = dataset_type

            if metrics is not None:
                # Adding metrics all detailed results
                for metric in metrics:
                    row["metric_" + metric] = metrics[metric][idx]

            if labels is not None:
                row["relevant_item_ids"] = [labels[idx]]
                row["rec_item_ids"] = pred_item_ids[idx]
                row["rec_item_scores"] = pred_item_scores[idx]

            # Adding metadata features
            for feat_name in preds_metadata:
                row["metadata_" + feat_name] = [preds_metadata[feat_name][idx]]

            new_rows.append(row)

        new_rows_df = pd.DataFrame(new_rows)

        self._create_pq_writer_if_needed(new_rows_df)
        self._append_new_rows_to_parquet(new_rows_df)

    def close(self):
        if self.pq_writer:
            self.pq_writer.close()


class AllHparams:
    """
    Used to aggregate all arguments in a single object for logging
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def to_sanitized_dict(self):
        result = {k: v for k, v in self.__dict__.items() if safe_json(v)}
        return result
