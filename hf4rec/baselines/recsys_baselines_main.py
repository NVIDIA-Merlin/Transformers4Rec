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
"""
Example arguments for command line: 
    CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false python recsys_main.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset_path/ --start_date 2019-10-01 --end_date 2019-10-15 --data_loader_engine nvtabular --per_device_train_batch_size 320 --per_device_eval_batch_size 512 --model_type gpt2 --loss_type cross_entropy --logging_steps 10 --d_model 256 --n_layer 2 --n_head 8 --dropout 0.1 --learning_rate 0.001 --similarity_type concat_mlp --num_train_epochs 1 --all_rescale_factor 1 --neg_rescale_factor 0 --feature_config ../datasets/ecommerce-large/config/features/session_based_features_pid.yaml --inp_merge mlp --tf_out_activation tanh --experiments_group local_test --weight_decay 1.3e-05 --learning_rate_schedule constant_with_warmup --learning_rate_warmup_steps 0 --learning_rate_num_cosine_cycles 1.25 --dataloader_drop_last --compute_metrics_each_n_steps 1 --hidden_act gelu_new --save_steps 0 --eval_on_last_item_seq_only --fp16 --overwrite_output_dir --session_seq_length_max 20 --predict_top_k 1000 --eval_accumulation_steps 10
"""
import importlib
import inspect
import logging
import math
import multiprocessing
import os
import random
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import wandb
import yaml
from joblib import Parallel, delayed
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from tqdm import tqdm
from transformers import HfArgumentParser

from ..evaluation.recsys_metrics import EvalMetrics
from ..recsys_args import DataArguments, ModelArguments, TrainingArguments
from ..recsys_data import fetch_data_loader
from ..recsys_utils import (
    get_label_feature_name,
    get_object_size,
    get_parquet_files_names,
    get_timestamp_feature_name,
    safe_json,
)

try:
    import cPickle as pickle
except:
    import pickle


logger = logging.getLogger(__name__)

import dllogger as DLLogger
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity

DLLOGGER_FILENAME = "log.json"

# this code use Version 3
assert sys.version_info.major > 2

SESSION_FNAME = "SessionId"
ITEM_FNAME = "ItemId"
TIMESTAMP_FNAME = "Time"


def get_algorithm_class(alg_name):
    if alg_name == "vsknn":
        from .knn.vsknn import VMContextKNN

        return VMContextKNN

    if alg_name == "vstan":
        from .knn.vstan import VSKNN_STAN

        return VSKNN_STAN

    if alg_name == "stan":
        from .knn.stan import STAN

        return STAN

    elif alg_name == "gru4rec":
        from .gru4rec.gru4rec import GRU4Rec

        return GRU4Rec

    elif alg_name == "narm":
        from .narm.narm import NARM

        return NARM
    else:
        raise ValueError(f"The '{alg_name}' algorithm is not supported")


class WandbLogger:
    def __init__(self):
        if WandbLogger.is_wandb_available():
            import wandb

            wandb.ensure_configured()
            if wandb.api.api_key is None:
                has_wandb = False
                logger.warning(
                    "W&B installed but not logged in. Run `wandb login` or set the WANDB_API_KEY env variable."
                )
                self._wandb = None
            else:
                self._wandb = wandb
        else:
            logger.warning(
                "WandbCallback requires wandb to be installed. Run `pip install wandb`."
            )

        self._initialized = False

    def setup(self, config_to_log, reinit, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.
        Environment:
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return

        logger.info(
            'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
        )

        self._wandb.init(
            project=os.getenv("WANDB_PROJECT", "huggingface"),
            config=config_to_log,
            # name="xpto",
            reinit=reinit,
        )

        self._initialized = True

    def log(self, logs, step=0):
        if self._wandb is None:
            return
        if not self._initialized:
            return

        self._wandb.log(logs, step=step)

    @classmethod
    def is_wandb_available(cls):
        # any value of WANDB_DISABLED disables wandb
        if os.getenv("WANDB_DISABLED", "").upper() in ["TRUE", "1"]:
            logger.warn(
                "Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
                "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
            )
            return False
        return importlib.util.find_spec("wandb") is not None


class AllHparams:
    """
    Used to aggregate all arguments in a single object for logging
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def to_sanitized_dict(self):
        result = {k: v for k, v in self.__dict__.items() if safe_json(v)}
        return result


def get_dataloaders(
    data_args, training_args, train_data_paths, eval_data_paths, feature_map
):
    train_loader = fetch_data_loader(
        data_args, training_args, feature_map, train_data_paths, is_train_set=True
    )
    eval_loader = fetch_data_loader(
        data_args, training_args, feature_map, eval_data_paths, is_train_set=False
    )

    return train_loader, eval_loader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def main():

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            data_args,
            model_args,
            training_args,
            remaining_args,
        ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        remaining_hparams = parse_remaining_args(remaining_args)

    # Ensuring to set W&B run name to null, so that a nice run name is generated
    training_args.run_name = None

    all_hparams = {
        **asdict(data_args),
        **asdict(model_args),
        **asdict(training_args),
        **remaining_hparams,
    }
    all_hparams = {k: v for k, v in all_hparams.items() if safe_json(v)}

    # Loading features config file
    with open(data_args.feature_config) as yaml_file:
        feature_map = yaml.load(yaml_file, Loader=yaml.FullLoader)

    item_id_fname = get_label_feature_name(feature_map)
    timestamp_fname = get_timestamp_feature_name(feature_map)

    # training_args.label_names = [label_name]
    target_size = feature_map[item_id_fname]["cardinality"]

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

    DLLogger.init(
        backends=[
            StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(
                Verbosity.VERBOSE,
                os.path.join(training_args.output_dir, DLLOGGER_FILENAME),
            ),
        ]
    )

    DLLogger.log(step="PARAMETER", data=all_hparams, verbosity=Verbosity.DEFAULT)
    DLLogger.flush()

    set_seed(training_args.seed)

    wandb_logger = WandbLogger()
    wandb_logger.setup(all_hparams, reinit=False)

    """
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
    """

    algorithm = get_algorithm(
        model_args.model_type, remaining_hparams, training_args.seed
    )

    results_times = {}

    start_session_id = 0

    train_interactions_cumulative_df = pd.DataFrame()

    items_to_predict_set = set()

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
            time_indices_train = time_index
            time_index_eval = time_index + 1

        # Training
        if training_args.do_train:
            logger.info(
                f"************* Training (time indices:{time_indices_train}) *************"
            )

            # If trains with all past available training data (since time index 1)
            if (
                data_args.no_incremental_training
                and data_args.training_time_window_size == 0
            ):
                # Loads only the last time index, because other past data is already available for concatenation
                time_indices_train = time_indices_train[-1]

            train_data_paths = get_parquet_files_names(
                data_args, time_indices_train, is_train=True
            )

            train_sessions_df, start_session_id = prepare_sessions_data(
                train_data_paths,
                item_id_fname=item_id_fname,
                timestamp_fname=timestamp_fname,
                start_session_id=start_session_id,
                session_seq_length_max=data_args.session_seq_length_max,
            )

            # train_sessions_df = train_sessions_df[:100]
            train_interactions_df = sessions_to_interactions_dataframe(
                train_sessions_df
            )

            # If trains with all past available training data (since time index 1)
            if (
                data_args.no_incremental_training
                and data_args.training_time_window_size == 0
            ):
                # Concatenating with data already preprocessed from past time units
                train_interactions_cumulative_df = pd.concat(
                    [train_interactions_cumulative_df, train_interactions_df]
                )
                train_interactions_df = train_interactions_cumulative_df

            algorithm.fit(train_interactions_df)

            # logger.info(f"The algorithm memory size: {get_object_size(algorithm)}")

            unique_items = train_interactions_df[ITEM_FNAME].unique()
            if data_args.no_incremental_training:
                items_to_predict_set = set(unique_items)
            else:
                items_to_predict_set.update(unique_items)

        # Evaluation
        if training_args.do_eval:
            logger.info(
                f"************* Evaluation (time index:{time_index_eval}) *************"
            )
            eval_data_paths = get_parquet_files_names(
                data_args,
                [time_index_eval],
                is_train=False,
                eval_on_test_set=training_args.eval_on_test_set,
            )

            eval_sessions_df, start_session_id = prepare_sessions_data(
                eval_data_paths,
                item_id_fname=item_id_fname,
                timestamp_fname=timestamp_fname,
                start_session_id=start_session_id,
                session_seq_length_max=data_args.session_seq_length_max,
            )

            eval_interactions_df = sessions_to_interactions_dataframe(eval_sessions_df)

            # Grouping the sessions again after preprocessing
            eval_sessions_df = (
                eval_interactions_df.groupby(SESSION_FNAME)
                .agg({ITEM_FNAME: list, TIMESTAMP_FNAME: list})
                .reset_index()
            )

            items_to_predict = np.array(list(items_to_predict_set))

            # eval_sessions_preproc_df = eval_sessions_preproc_df[:100]
            if not remaining_hparams["eval_baseline_cpu_parallel"]:
                # Sequential approach
                metrics = EvalMetrics(ks=[10, 20], use_cpu=True, use_torch=False)
                for idx, session_row in tqdm(
                    eval_sessions_df.iterrows(), total=len(eval_sessions_df),
                ):
                    evaluate_session(
                        algorithm,
                        metrics,
                        session_row,
                        items_to_predict,
                        total_items=target_size,
                        eval_on_last_item_seq_only=model_args.eval_on_last_item_seq_only,
                    )
                eval_metrics_results = metrics.result()

            else:
                # Parallel approach
                num_cores = multiprocessing.cpu_count()
                logger.info(f"Number of CPU cores: {num_cores}")

                n_workers = max(data_args.workers_count, 1)
                logger.info(f"Number of workers (--workers_count): {n_workers}")

                logger.info(
                    "Eval dataset - # sessions: {}".format(len(eval_sessions_df))
                )
                chunk_size = (len(eval_sessions_df) // n_workers) + 1
                logger.info(f"# sessions by worker: {chunk_size}")
                eval_sessions_df_chunks = split(eval_sessions_df, chunk_size=chunk_size)

                chunks_metrics_results = Parallel(
                    n_jobs=n_workers, batch_size=1, verbose=100,
                )(
                    delayed(evaluate_sessions_parallel)(
                        sessions_chunk_df,
                        algorithm,  # deepcopy(algorithm),
                        items_to_predict,
                        target_size,
                        model_args.eval_on_last_item_seq_only,
                        job_id,
                    )
                    for job_id, sessions_chunk_df in enumerate(eval_sessions_df_chunks)
                )

                # Averaging metrics by chunk
                chunks_metrics_df = pd.DataFrame(chunks_metrics_results)
                eval_metrics_results = chunks_metrics_df.mean(axis=0).to_dict()

            # Adding prefix to make them compatible with the Transformers metrics
            eval_metrics_results = {
                f"eval_{k}": v for k, v in eval_metrics_results.items()
            }
            results_times[time_index_eval] = eval_metrics_results

            # Logging metrics with DLLogger
            DLLogger.log(
                step=(time_index_eval),
                data=eval_metrics_results,
                verbosity=Verbosity.VERBOSE,
            )
            DLLogger.flush()

            logger.info(
                f"Eval metrics for time index {time_index_eval}: {eval_metrics_results}"
            )

            # Logging to W&B
            eval_metrics_results_wandb = {
                k.replace("eval_", "eval/"): v for k, v in eval_metrics_results.items()
            }
            wandb_logger.log(eval_metrics_results_wandb, step=time_index_eval)

    logger.info("Training and evaluation loops are finished")

    if training_args.do_eval:
        logger.info("Computing and loging AOT metrics")
        results_df = pd.DataFrame.from_dict(results_times, orient="index")
        results_df.reset_index().to_csv(
            os.path.join(training_args.output_dir, "eval_train_results.csv"),
            index=False,
        )

        # Computing Average Over Time (AOT) metrics
        results_avg_time = dict(results_df.mean())
        results_avg_time = {f"{k}_AOT": v for k, v in results_avg_time.items()}
        # Logging to W&B
        results_avg_time_wandb = {
            k.replace("eval_", "eval/"): v for k, v in results_avg_time.items()
        }
        wandb_logger.log(results_avg_time_wandb, step=time_index_eval)

        log_aot_metric_results(training_args.output_dir, results_avg_time)


def parse_remaining_args(remaining_args):
    def parse_buffer(args_buffer, hparams):
        if len(args_buffer) > 0:
            if len(args_buffer) == 1:
                hparams[args_buffer[0]] = True
            elif len(args_buffer) == 2:
                hparams[args_buffer[0]] = args_buffer[1]
            else:
                raise ValueError(
                    "Could not parse these arguments: {}".format(args_buffer)
                )

    hparams = {}
    args_buffer = []
    for arg in remaining_args:
        if arg.startswith("--"):
            parse_buffer(args_buffer, hparams)

            args_buffer = [arg.replace("--", "")]
        else:
            args_buffer.append(arg)

    parse_buffer(args_buffer, hparams)

    # Type casting the arguments from string
    hparams = {k: cast_str_argument(v) for k, v in hparams.items()}
    # Make it return None if the key does not exist
    hparams = defaultdict(lambda: None, hparams)

    return hparams


def cast_str_argument(arg_value):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_float(s):
        try:
            int(s)
            return False
        except ValueError:
            float(s)
            return True

    def is_bool(value):
        return value.lower() in ["false", "true"]

    if type(arg_value) is not str:
        return arg_value
    else:
        if is_bool(arg_value):
            return arg_value.lower().strip() == "true"
        elif is_number(arg_value):
            if is_float(arg_value):
                return float(arg_value)
            else:
                return int(arg_value)
        else:
            return str(arg_value)


def get_algorithm(model_type, remaining_hparams, seed):
    model_hparms = {
        k.replace(f"{model_type}-", ""): v
        for k, v in remaining_hparams.items()
        if k.startswith(f"{model_type}-")
    }

    alg_cls = get_algorithm_class(model_type)
    # Removing not existing model args in the class constructor
    # model_hparms = filter_kwargs(model_hparms, alg_cls)

    constructor_args = inspect.getfullargspec(alg_cls.__init__)
    # Sets the seed if the model class accepts it
    if "seed" in constructor_args.args:
        model_hparms["seed"] = seed

    logger.info(
        f"Instantiating the algorithm {model_type} with these arguments: {model_hparms}"
    )

    algorithm_obj = alg_cls(
        session_key=SESSION_FNAME,
        item_key=ITEM_FNAME,
        time_key=TIMESTAMP_FNAME,
        **model_hparms,
    )

    return algorithm_obj


def index_marks(nrows, chunk_size):
    return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)


def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)


def evaluate_sessions_parallel(
    sessions_chuck_df,
    algorithm,
    items_to_predict,
    target_size,
    eval_on_last_item_seq_only,
    job_id,
):
    # TODO: Make the metric top-k a hyperparameter
    metrics = EvalMetrics(ks=[10, 20], use_cpu=True, use_torch=False)

    for idx, session_row in sessions_chuck_df.iterrows():
        evaluate_session(
            algorithm,
            metrics,
            session_row,
            items_to_predict,
            target_size,
            eval_on_last_item_seq_only,
        )

    eval_metrics_results = metrics.result()
    return eval_metrics_results


def evaluate_session(
    algorithm,
    metrics,
    session_row,
    items_to_predict,
    total_items,
    eval_on_last_item_seq_only,
):
    last_valid_preds_all_items = None
    for pos, (item_id, ts) in enumerate(
        zip(session_row[ITEM_FNAME], session_row[TIMESTAMP_FNAME])
    ):
        if pos < len(session_row[ITEM_FNAME]) - 1:
            label_next_item = session_row[ITEM_FNAME][pos + 1]

            preds_series = algorithm.predict_next(
                session_id=session_row[SESSION_FNAME],
                input_item_id=item_id,
                predict_for_item_ids=items_to_predict,
                timestamp=ts,
            )

            if (
                not eval_on_last_item_seq_only
                or pos == len(session_row[ITEM_FNAME]) - 2
            ):
                # preds_series.sort_values(ascending=False, inplace=True)
                preds_all_items = np.zeros(total_items)

                # Checks if the algorithm was able to provide next-item recommendations after the given item.
                # For example, GRU4Rec is unable to recommend after seing an item which was not seen during train
                # In such cases, we use the last valid recommendation list for the session, if available.
                # Otherwise, if there was no valid recommendation list for the session, we assume a wrong prediction
                if preds_series is not None:
                    preds_all_items[preds_series.index] = preds_series.values
                    last_valid_preds_all_items = preds_all_items
                else:
                    logger.debug("Item not found during prediction: {}".format(item_id))
                    if last_valid_preds_all_items is not None:
                        preds_all_items = last_valid_preds_all_items
                    else:
                        # This ensures that the recommendation is considered wrong
                        label_next_item = -1

                metrics.update(
                    preds=preds_all_items.reshape(1, -1),
                    labels=np.array([label_next_item]),
                )


def dataframe_from_parquet_files(train_data_paths, cols_to_read):
    dataframes = []
    for parquet_file in train_data_paths:
        df = pd.read_parquet(parquet_file, columns=cols_to_read)
        dataframes.append(df)

    concat_df = pd.concat(dataframes)
    return concat_df


def prepare_sessions_data(
    train_data_paths,
    item_id_fname,
    timestamp_fname,
    start_session_id,
    session_seq_length_max,
):
    concat_df = dataframe_from_parquet_files(
        train_data_paths, cols_to_read=[item_id_fname, timestamp_fname]
    )

    # TODO: Ensure data is sorted by time

    # Generating contiguous session ids
    concat_df[SESSION_FNAME] = start_session_id + np.arange(len(concat_df))
    last_session_id = concat_df[SESSION_FNAME].max()

    concat_df = concat_df.rename(
        {item_id_fname: ITEM_FNAME, timestamp_fname: TIMESTAMP_FNAME}, axis=1
    )

    concat_df = concat_df[[SESSION_FNAME, ITEM_FNAME, TIMESTAMP_FNAME]]

    # Truncating long sessions

    concat_df[ITEM_FNAME] = concat_df[ITEM_FNAME].apply(
        lambda x: x[:session_seq_length_max]
    )
    concat_df[TIMESTAMP_FNAME] = concat_df[TIMESTAMP_FNAME].apply(
        lambda x: x[:session_seq_length_max]
    )

    return concat_df, last_session_id


def sessions_to_interactions_dataframe(sessions_df):
    # Converts from the representation of one row per session to one row by interaction
    interactions_df = explode_multiple_cols(sessions_df, [ITEM_FNAME, TIMESTAMP_FNAME])

    convert_timestamps(interactions_df)

    return interactions_df


def convert_timestamps(interactions_df):
    # TODO: This condition is TEMPORARY and only the YOOCHOOSE ecommerce dataset, for which
    # the timestamp column is datetime[ns] and not timestamp (in ms) as the other datasets.
    # P.s. The YOOCHOOSE preprocessed dataset was already fixed to output timestamps,
    # but will generate a new dataset later after finishing the paper experiments already started
    if is_datetime(interactions_df[TIMESTAMP_FNAME].dtype):
        interactions_df[TIMESTAMP_FNAME] = (
            interactions_df[TIMESTAMP_FNAME].astype("int64") // 1e6
        ).astype("int64")

    # Convert timestamps to seconds
    ts_lengh_series = interactions_df[TIMESTAMP_FNAME].astype(str).apply(len)

    ts_lenght_min = ts_lengh_series.min()
    ts_lenght_max = ts_lengh_series.max()

    # Timestamp in nanoseconds
    if ts_lenght_min >= 18 and ts_lenght_min < 21:
        interactions_df[TIMESTAMP_FNAME] = (
            interactions_df[TIMESTAMP_FNAME] / 1e9
        ).astype("int64")
    # Timestamp in microseconds
    if ts_lenght_min >= 15:
        interactions_df[TIMESTAMP_FNAME] = (
            interactions_df[TIMESTAMP_FNAME] / 1e6
        ).astype("int64")
    # Timestamp in miliseconds
    if ts_lenght_min >= 12:
        interactions_df[TIMESTAMP_FNAME] = (
            interactions_df[TIMESTAMP_FNAME] / 1e3
        ).astype("int64")
    # Timestamp in seconds
    elif ts_lenght_min >= 9:
        pass
    else:
        raise Exception(
            f"The timestamps have invalid length (min: {ts_lenght_min}, max: {ts_lenght_max})."
        )


def explode_multiple_cols(df, lst_cols, fill_value=""):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return (
            pd.DataFrame(
                {
                    col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                    for col in idx_cols
                }
            )
            .assign(**{col: np.concatenate(df[col].values) for col in lst_cols})
            .loc[:, df.columns]
        )
    else:
        # at least one list in cells is empty
        return (
            pd.DataFrame(
                {
                    col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                    for col in idx_cols
                }
            )
            .assign(**{col: np.concatenate(df[col].values) for col in lst_cols})
            .append(df.loc[lens == 0, idx_cols])
            .fillna(fill_value)
            .loc[:, df.columns]
        )


def filter_kwargs(kwargs, thing_with_kwargs):
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    filtered_dict = {
        filter_key: kwargs[filter_key]
        for filter_key in filter_keys
        if filter_key in kwargs
    }
    return filtered_dict


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

    # Logging AOD metrics with DLLogger
    DLLogger.log(step=(), data=results_avg_time, verbosity=Verbosity.VERBOSE)
    DLLogger.flush()

    return results_avg_time


if __name__ == "__main__":
    main()

