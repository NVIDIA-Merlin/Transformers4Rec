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
import sys

import pandas as pd
import transformers
import yaml
from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import is_main_process

from .recsys_args import DataArguments, ModelArguments, TrainingArguments
from .recsys_data import fetch_data_loader, get_items_sorted_freq
from .recsys_meta_model import RecSysMetaModel
from .recsys_models import get_recsys_model
from .recsys_outputs import (
    AttentionWeightsLogger,
    PredictionLogger,
    config_dllogger,
    creates_output_dir,
    log_aot_metric_results,
    log_metric_results,
    log_parameters,
    set_log_attention_weights_callback,
    set_log_predictions_callback,
)
from .recsys_trainer import DatasetMock, DatasetType, RecSysTrainer
from .recsys_utils import (
    get_label_feature_name,
    get_parquet_files_names,
    get_timestamp_feature_name,
)

logger = logging.getLogger(__name__)


# this code use Version 3
assert sys.version_info.major > 2


def main():

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (data_args, model_args, training_args,) = parser.parse_args_into_dataclasses()

    # Ensuring to set W&B run name to null, so that a nice run name is generated
    training_args.run_name = None

    # Loading features config file
    with open(data_args.feature_config) as yaml_file:
        feature_map = yaml.load(yaml_file, Loader=yaml.FullLoader)

    label_name = get_label_feature_name(feature_map)
    target_size = feature_map[label_name]["cardinality"]

    config_pyprof(training_args)

    creates_output_dir(training_args)

    config_logging(training_args)

    set_seed(training_args.seed)

    # Instantiates the model defined in --model_type
    seq_model, config = get_recsys_model(
        model_args, data_args, training_args, target_size
    )
    # Instantiate the RecSys Meta Model
    rec_model = RecSysMetaModel(seq_model, config, model_args, data_args, feature_map)

    # if training_args.model_parallel:
    #    rec_model = rec_model.to(training_args.device)

    # Instantiate the RecSysTrainer, which manages training and evaluation
    trainer = RecSysTrainer(
        model=rec_model, args=training_args, model_args=model_args, data_args=data_args,
    )

    log_parameters(trainer, data_args, model_args, training_args)

    set_log_attention_weights_callback(trainer, training_args)

    results_over_time = {}

    for time_index in range(
        data_args.start_time_window_index, data_args.final_time_window_index
    ):
        if data_args.no_incremental_training:
            if data_args.training_time_window_size > 0:
                time_index_eval = time_index + 1
                time_indices_train = list(
                    range(
                        max(
                            time_index_eval - data_args.training_time_window_size,
                            data_args.start_time_window_index,
                        ),
                        time_index_eval,
                    )
                )
            else:
                time_indices_train = list(range(1, time_index + 1))
                time_index_eval = time_index + 1
        else:
            time_indices_train = [time_index]
            time_index_eval = time_index + 1

        if (
            model_args.negative_sampling
            and model_args.neg_sampling_extra_samples_per_batch > 0
        ):
            items_sorted_freq_series = get_items_sorted_freq(
                train_data_paths, item_id_feature_name=label_name
            )
            trainer.model.set_items_freq_for_sampling(items_sorted_freq_series)

        train_data_paths = get_parquet_files_names(
            data_args, time_indices_train, is_train=True
        )
        eval_data_paths = get_parquet_files_names(
            data_args,
            [time_index_eval],
            is_train=False,
            eval_on_test_set=training_args.eval_on_test_set,
        )

        # Training
        if training_args.do_train:
            logger.info(
                f"************* Training (time indices:{time_indices_train[0]}-{time_indices_train[-1]}) *************"
            )

            train_loader, eval_loader = get_dataloaders(
                data_args,
                training_args,
                train_data_paths,
                eval_data_paths,
                feature_map,
            )

            trainer.set_train_dataloader(train_loader)
            trainer.set_eval_dataloader(eval_loader)

            model_path = (
                model_args.model_name_or_path
                if model_args.model_name_or_path is not None
                and os.path.isdir(model_args.model_name_or_path)
                else None
            )

            trainer.reset_lr_scheduler()
            trainer.train(model_path=model_path)

        # Evaluation
        if training_args.do_eval:
            set_log_predictions_callback(trainer, training_args, time_index_eval)

            logger.info(f"************* Evaluation *************")

            # Loading again the data loaders, because some data loaders (e.g. NVTabular do not reset after they are not totally iterated over)
            train_loader, eval_loader = get_dataloaders(
                data_args,
                training_args,
                train_data_paths,
                eval_data_paths,
                feature_map,
            )

            logger.info(
                f"Evaluating on train set (time index:{time_indices_train})...."
            )
            trainer.set_train_dataloader(train_loader)
            # Defining temporarily the the train data loader for evaluation
            trainer.set_eval_dataloader(train_loader)

            train_metrics = trainer.evaluate(metric_key_prefix=DatasetType.train.value)
            trainer.wipe_memory()
            log_metric_results(
                training_args.output_dir,
                train_metrics,
                prefix=DatasetType.train.value,
                time_index=time_index_eval,
            )

            logger.info(f"Evaluating on test set (time index:{time_index_eval})....")
            trainer.set_eval_dataloader(eval_loader)
            eval_metrics = trainer.evaluate(metric_key_prefix=DatasetType.eval.value)
            trainer.wipe_memory()

            log_metric_results(
                training_args.output_dir,
                eval_metrics,
                prefix=DatasetType.eval.value,
                time_index=time_index_eval,
            )

            results_over_time[time_index_eval] = {
                **eval_metrics,
                **train_metrics,
            }

    logger.info("Training and evaluation loops are finished")

    if trainer.is_world_process_zero():

        logger.info("Saving model...")
        trainer.save_model()
        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

        if training_args.do_eval:
            logger.info("Computing and loging AOT metrics")
            results_df = pd.DataFrame.from_dict(results_over_time, orient="index")
            results_df.reset_index().to_csv(
                os.path.join(training_args.output_dir, "eval_train_results.csv"),
                index=False,
            )

            # Computing Average Over Time (AOT) metrics
            results_avg_time = dict(results_df.mean())
            results_avg_time = {f"{k}_AOT": v for k, v in results_avg_time.items()}
            # Logging to W&B
            trainer.log(results_avg_time)

            log_aot_metric_results(training_args.output_dir, results_avg_time)


def get_dataloaders(
    data_args, training_args, train_data_paths, eval_data_paths, feature_map
):
    train_loader = fetch_data_loader(
        data_args, training_args, feature_map, train_data_paths, is_train_set=True,
    )
    eval_loader = fetch_data_loader(
        data_args, training_args, feature_map, eval_data_paths, is_train_set=False,
    )

    return train_loader, eval_loader


def config_pyprof(training_args):
    # Enables profiling with DLProf and Nsight Systems (slows down training)
    if training_args.pyprof:
        logger.info(
            "Enabling PyProf for profiling, to inspect with DLProf and Nsight Sytems. This will slow down "
            "training and should be used only for debug purposes.",
        )
        import pyprof

        pyprof.init(enable_function_stack=True)


def config_logging(training_args):
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

    config_dllogger(training_args.output_dir)


if __name__ == "__main__":
    main()

