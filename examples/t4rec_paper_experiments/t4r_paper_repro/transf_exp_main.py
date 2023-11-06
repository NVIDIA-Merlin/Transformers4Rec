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

import glob
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import transformers
from exp_outputs import (
    config_dllogger,
    creates_output_dir,
    log_aot_metric_results,
    log_metric_results,
    log_parameters,
)
from merlin.io import Dataset
from merlin.schema import Tags
from transf_exp_args import DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import is_main_process

import transformers4rec.torch as t4r
from merlin_standard_lib import Schema
from transformers4rec.torch import Trainer
from transformers4rec.torch.utils.data_utils import MerlinDataLoader
from transformers4rec.torch.utils.examples_utils import wipe_memory

logger = logging.getLogger(__name__)


def main():
    # Parsing command line arguments
    (data_args, model_args, training_args) = parse_command_line_args()

    # Ensuring to set W&B run name to null, so that a nice run name is generated
    training_args.run_name = None

    # Loading the schema of the dataset
    schema = Schema().from_proto_text(data_args.features_schema_path)
    if not data_args.use_side_information_features:
        schema = schema.select_by_tag([Tags.ITEM_ID])

    item_id_col = schema.select_by_tag([Tags.ITEM_ID]).column_names[0]
    col_names = schema.column_names
    logger.info("Column names: {}".format(col_names))

    creates_output_dir(training_args)
    config_logging(training_args)
    set_seed(training_args.seed)

    # Getting masking config
    masking_kwargs = get_masking_kwargs(model_args)

    # Obtaining Stochastic Shared embeddings config
    pre_transforms = []
    if model_args.stochastic_shared_embeddings_replacement_prob > 0:
        pre_transforms.append(
            t4r.StochasticSwapNoise(
                pad_token=0,
                replacement_prob=model_args.stochastic_shared_embeddings_replacement_prob,
                schema=schema,
            )
        )

    post_transforms = []

    # Adding input dropout config
    if model_args.input_dropout > 0:
        input_dropout = t4r.TabularDropout(dropout_rate=model_args.input_dropout)
        post_transforms.append(input_dropout)

    # Obtaining feature-wise layer norm config
    if model_args.layer_norm_featurewise:
        post_transforms.append("layer-norm")

    # Configuring categorical features embedding sizes
    embedding_dims = {item_id_col: model_args.item_embedding_dim}
    embedding_dim_default = model_args.item_embedding_dim
    infer_embedding_sizes = not model_args.input_features_aggregation.startswith("element-wise")

    # Configuring embedding initializers
    embeddings_initializers = {}
    for col in col_names:
        if col == item_id_col:
            std = model_args.item_id_embeddings_init_std
        else:
            std = model_args.other_embeddings_init_std
        embeddings_initializers[col] = partial(torch.nn.init.normal_, mean=0.0, std=std)

    # Define input module to process tabular input-features and to prepare masked inputs
    input_module = t4r.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=training_args.max_sequence_length,
        aggregation=model_args.input_features_aggregation,
        d_output=model_args.d_model,
        pre=pre_transforms,
        post=post_transforms,
        # Embedding Features args
        embedding_dims=embedding_dims,
        embedding_dim_default=embedding_dim_default,
        infer_embedding_sizes=infer_embedding_sizes,
        infer_embedding_sizes_multiplier=model_args.embedding_dim_from_cardinality_multiplier,
        embeddings_initializers=embeddings_initializers,
        continuous_soft_embeddings=(
            model_args.numeric_features_soft_one_hot_encoding_num_embeddings > 0
        ),
        soft_embedding_cardinality_default=(
            model_args.numeric_features_soft_one_hot_encoding_num_embeddings
        ),
        soft_embedding_dim_default=model_args.numeric_features_project_to_embedding_dim,
        **masking_kwargs,
    )

    # Loss function: Cross-entropy with label smoothing
    label_smoothing_xe_loss = t4r.LabelSmoothCrossEntropyLoss(
        reduction="mean", smoothing=model_args.label_smoothing
    )

    # Configuring metrics: NDCG@10, NDCG@20, Recall@10, Recall@20
    metrics = [
        t4r.ranking_metric.NDCGAt(top_ks=[10, 20], labels_onehot=True),
        t4r.ranking_metric.RecallAt(top_ks=[10, 20], labels_onehot=True),
    ]

    # Configures the next-item prediction-task
    prediction_task = t4r.NextItemPredictionTask(
        weight_tying=model_args.mf_constrained_embeddings,
        softmax_temperature=model_args.softmax_temperature,
        metrics=metrics,
        loss=label_smoothing_xe_loss,
        sampled_softmax=model_args.sampled_softmax,
        max_n_samples=model_args.sampled_softmax_max_n_samples,
    )

    model_config = get_model_config(training_args, model_args)

    # Generates the final PyTorch model
    model = model_config.to_torch_model(input_module, prediction_task)

    trainer = Trainer(
        model=model,
        args=training_args,
        schema=schema,
        compute_metrics=True,
        incremental_logging=True,
    )

    log_parameters(trainer, data_args, model_args, training_args)

    results_over_time = incremental_train_eval(
        trainer,
        start_time_index=data_args.start_time_window_index,
        end_time_index=data_args.final_time_window_index,
        input_dir=data_args.data_path,
        training_args=training_args,
        data_args=data_args,
    )

    if training_args.do_eval:
        logger.info("Computing and logging AOT (Average Over Time) metrics")
        results_df = pd.DataFrame.from_dict(results_over_time, orient="index")
        results_df.reset_index().to_csv(
            os.path.join(training_args.output_dir, "eval_train_results.csv"),
            index=False,
        )

        results_avg_time = dict(results_df.mean())
        results_avg_time = {f"{k}_AOT": v for k, v in results_avg_time.items()}
        # Logging to W&B / Tensorboard
        trainer.log(results_avg_time)

        log_aot_metric_results(training_args.output_dir, results_avg_time)

    # Mimic the inference by manually computing recall@10 using the evaluation data
    # of the last time-index.
    eval_path = os.path.join(
        data_args.data_path,
        str(
            data_args.final_time_window_index,
        ).zfill(data_args.time_window_folder_pad_digits),
        "test.parquet" if training_args.eval_on_test_set else "valid.parquet",
    )
    prediction_data = pd.read_parquet(eval_path)
    # Extract label
    labels = prediction_data["sess_pid_seq"].apply(lambda x: x[-1]).values

    # Truncate input sequences up to last item - 1 to mimic the inference
    def mask_last_interaction(x):
        return list(x[:-1])

    list_columns = schema.select_by_tag("list").column_names
    for col in list_columns:
        prediction_data[col] = prediction_data[col].apply(mask_last_interaction)
    # Get top-10 predictions
    test_loader = MerlinDataLoader.from_schema(
        schema,
        Dataset(prediction_data),
        training_args.per_device_eval_batch_size,
        max_sequence_length=training_args.max_sequence_length,
        shuffle=False,
    )
    trainer.test_dataloader = test_loader
    trainer.args.predict_top_k = 10
    topk_preds = trainer.predict(test_loader).predictions[0]
    # Compute recall@10
    recall_10 = recall(topk_preds, labels)
    logger.info(f"Recall@10 of manually masked test data = {str(recall_10)}")
    output_file = os.path.join(training_args.output_dir, "eval_results_over_time.txt")
    with open(output_file, "a") as writer:
        writer.write(f"\n***** Recall@10 of simulated inference = {recall_10} *****\n")
    # Verify that the recall@10 from train.evaluate() matches the recall@10 calculated manually
    if not isinstance(input_module.masking, t4r.masking.PermutationLanguageModeling):
        # TODO fix inference discrepancy for permutation language modeling
        assert np.isclose(recall_10, results_over_time[2]["eval_/next-item/recall_at_10"], rtol=0.1)


def recall(predicted_items: np.ndarray, real_items: np.ndarray) -> float:
    bs, top_k = predicted_items.shape
    valid_rows = real_items != 0

    # reshape predictions and labels to compare
    # the top-10 predicted item-ids with the label id.
    real_items = real_items.reshape(bs, 1, -1)
    predicted_items = predicted_items.reshape(bs, 1, top_k)

    num_relevant = real_items.shape[-1]
    predicted_correct_sum = (predicted_items == real_items).sum(-1)
    predicted_correct_sum = predicted_correct_sum[valid_rows]
    recall_per_row = predicted_correct_sum / num_relevant
    return np.mean(recall_per_row)


def incremental_train_eval(
    trainer, start_time_index, end_time_index, input_dir, training_args, data_args
):
    """
    Performs incremental training eand evaluation.
    Iteratively train using data of a given window index and evaluate on the validation data
    of the following index.
    Parameters
    ----------
    start_time_index: int
        The start index for training, it should match the partitions of the data directory
    end_time_index: int
        The end index for training, it should match the partitions of the  data directory
    input_dir: str
        The input directory where the parquet files were saved based on partition column
    Returns
    -------
    results_over_time: dict
        The average over time of ranking metrics.
    """
    results_over_time = {}
    for time_index in range(start_time_index, end_time_index):
        # 1. Set data
        time_index_train = time_index
        time_index_eval = time_index + 1
        train_paths = glob.glob(
            os.path.join(
                input_dir,
                str(time_index_train).zfill(data_args.time_window_folder_pad_digits),
                "train.parquet",
            )
        )
        eval_paths = glob.glob(
            os.path.join(
                input_dir,
                str(time_index_eval).zfill(data_args.time_window_folder_pad_digits),
                "test.parquet" if training_args.eval_on_test_set else "valid.parquet",
            )
        )

        # 2. Train on train data of time_index
        if training_args.do_train:
            print("\n***** Launch training for day %s: *****" % time_index)
            trainer.train_dataset_or_path = train_paths
            trainer.reset_lr_scheduler()
            trainer.train()

        if training_args.do_eval:
            # 3. Evaluate on train data of time_index
            trainer.eval_dataset_or_path = train_paths
            train_metrics = trainer.evaluate(metric_key_prefix="train")
            print("\n***** Evaluation results for day %s (train set):*****\n" % time_index_eval)
            print(train_metrics)

            log_metric_results(
                training_args.output_dir,
                train_metrics,
                prefix="train",
                time_index=time_index_eval,
            )

            # free GPU for next day training
            wipe_memory()

            # 4. Evaluate on valid/test data of time_index+1
            trainer.eval_dataset_or_path = eval_paths
            eval_metrics = trainer.evaluate(metric_key_prefix="eval")
            print("\n***** Evaluation results for day %s (eval set):*****\n" % time_index_eval)
            print(eval_metrics)

            log_metric_results(
                training_args.output_dir,
                eval_metrics,
                prefix="eval",
                time_index=time_index_eval,
            )

            # free GPU for next day training
            wipe_memory()

        results_over_time[time_index_eval] = {
            **eval_metrics,
            **train_metrics,
        }

    return results_over_time


def get_masking_kwargs(model_args):
    kwargs = {}
    if model_args.plm:
        kwargs = {
            "masking": "plm",
            "plm_probability": model_args.plm_probability,
            "max_span_length": model_args.plm_max_span_length,
            "permute_all": model_args.plm_permute_all,
        }
    elif model_args.rtd:
        kwargs = {
            "masking": "rtd",
            "sample_from_batch": model_args.rtd_sample_from_batch,
            # rtd_use_batch_interaction=?
            # rtd_discriminator_loss_weight=?
            # rtd_generator_loss_weight=?
            # rtd_tied_generator=?
        }
    elif model_args.mlm:
        kwargs = {"masking": "mlm", "mlm_probability": model_args.mlm_probability}
    else:
        kwargs = {"masking": "clm"}

    return kwargs


def get_model_config(training_args, model_args):
    kwargs = {}

    if model_args.model_type == "gpt2":
        model_build_fn = t4r.GPT2Config.build
    if model_args.model_type == "xlnet":
        model_build_fn = t4r.XLNetConfig.build
        kwargs = {
            "summary_type": model_args.summary_type,
            "attn_type": model_args.attn_type,
        }
    if model_args.model_type == "electra":
        model_build_fn = t4r.ElectraConfig.build
    if model_args.model_type == "albert":
        model_build_fn = t4r.AlbertConfig.build
        num_hidden_groups = model_args.num_hidden_groups
        if model_args.num_hidden_groups == -1:
            num_hidden_groups = model_args.n_layer
        kwargs = {
            "num_hidden_groups": num_hidden_groups,
            "inner_group_num": model_args.inner_group_num,
        }
    if model_args.model_type == "transfoxl":
        model_build_fn = t4r.TransfoXLConfig.build

    model_config = model_build_fn(
        total_seq_length=training_args.max_sequence_length,
        d_model=model_args.d_model,
        n_head=model_args.n_head,
        n_layer=model_args.n_layer,
        hidden_act=model_args.hidden_act,
        initializer_range=model_args.initializer_range,
        layer_norm_eps=model_args.layer_norm_eps,
        dropout=model_args.dropout,
        pad_token=0,
        **kwargs,
    )

    return model_config


def parse_command_line_args():
    # Parsing command line argument
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    (
        data_args,
        model_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    # Adapting arguments used in the original paper reproducibility script to the new ones
    if training_args.session_seq_length_max:
        training_args.max_sequence_length = training_args.session_seq_length_max

    if training_args.learning_rate_schedule:
        training_args.lr_scheduler_type = training_args.learning_rate_schedule.replace(
            "_with_warmup", ""
        )

    if model_args.input_features_aggregation == "elementwise_sum_multiply_item_embedding":
        model_args.input_features_aggregation = "element-wise-sum-item-multi"

    return data_args, model_args, training_args


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
