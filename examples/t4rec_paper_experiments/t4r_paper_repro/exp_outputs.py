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
import wandb
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
from transformers.integrations import WandbCallback

logger = logging.getLogger(__name__)

DLLOGGER_FILENAME = "log.json"


def creates_output_dir(training_args):
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)


def config_dllogger(output_dir):
    DLLogger.init(
        backends=[
            StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(
                Verbosity.VERBOSE,
                os.path.join(output_dir, DLLOGGER_FILENAME),
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
            callback.setup(all_hparams_aggregated, trainer.state, trainer.model, reinit=False)

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
        writer.write("\n***** Eval results (avg over time) *****\n")
        for key in sorted(results_avg_time.keys()):
            logger.info("  %s = %s", key, str(results_avg_time[key]))
            writer.write("%s = %s\n" % (key, str(results_avg_time[key])))

    # Logging AOT metrics with DLLogger
    DLLogger.log(step=(), data=results_avg_time, verbosity=Verbosity.VERBOSE)
    DLLogger.flush()

    return results_avg_time


class AllHparams:
    """
    Used to aggregate all arguments in a single object for logging
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def to_sanitized_dict(self):
        result = {k: v for k, v in self.__dict__.items() if safe_json(v)}
        return result


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False
