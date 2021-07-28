from enum import Enum

import yaml

from transformers4rec.recsys_args import DataArguments, ModelArguments, TrainingArguments
from transformers4rec.recsys_data import fetch_data_loader, get_items_sorted_freq
from transformers4rec.utils.misc_utils import get_label_feature_name, get_parquet_files_names

TrainingArguments.local_rank = -1
TrainingArguments.dataloader_drop_last = True
TrainingArguments.device = "cuda"
TrainingArguments.report_to = []
TrainingArguments.debug = ["r"]
TrainingArguments.n_gpu = 1
TrainingArguments.gradient_accumulation_steps = 32
TrainingArguments.per_device_train_batch_size = 512 * 8
TrainingArguments.per_device_eval_batch_size = 512 * 8
TrainingArguments.output_dir = ""


DataArguments.session_seq_length_max = 20
DataArguments.nvt_part_mem_fraction = 0.7
DataArguments.nvt_part_size = None
DataArguments.start_time_window_index = 1
DataArguments.final_time_window_index = 5
DataArguments.data_path = "/workspace/yoochoose-data/yoochoose_transformed/"
DataArguments.data_loader_engine = "nvtabular"


class DatasetType(Enum):
    train = "train"
    eval = "eval"


def get_dataloaders(data_args, training_args, train_data_paths, eval_data_paths, feature_map):
    train_loader = fetch_data_loader(
        data_args,
        training_args,
        feature_map,
        train_data_paths,
        is_train_set=True,
    )
    eval_loader = fetch_data_loader(
        data_args,
        training_args,
        feature_map,
        eval_data_paths,
        is_train_set=False,
    )

    return train_loader, eval_loader


def fit_and_evaluate(trainer, start_time_window_index, final_time_window_index):

    # Loading features config file
    with open(trainer.data_args.feature_config) as yaml_file:
        feature_map = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # for time_index in range(DataArguments.start_time_window_index, DataArguments.final_time_window_index):

    time_indices_train = list(range(start_time_window_index, final_time_window_index + 1))
    time_index_eval = final_time_window_index + 1
    train_data_paths = get_parquet_files_names(trainer.data_args, time_indices_train, is_train=True)

    eval_data_paths = get_parquet_files_names(
        trainer.data_args,
        [time_index_eval],
        is_train=False,
        eval_on_test_set=trainer.args.eval_on_test_set,
    )

    # Training
    print(
        f"\n\n************* Training (time indices:{time_indices_train[0]}-{time_indices_train[-1]}) *************\n\n"
    )

    train_loader, eval_loader = get_dataloaders(
        trainer.data_args,
        trainer.args,
        train_data_paths,
        eval_data_paths,
        feature_map,
    )

    trainer.set_train_dataloader(train_loader)
    trainer.set_eval_dataloader(eval_loader)

    model_path = (
        trainer.model_args.model_name_or_path
        if trainer.model_args.model_name_or_path is not None
        and os.path.isdir(trainer.model_args.model_name_or_path)
        else None
    )

    trainer.reset_lr_scheduler()
    trainer.train(model_path=model_path)

    print(f"************* Evaluation *************\n")

    # Loading again the data loaders, because some data loaders (e.g. NVTabular do not reset after they are not totally iterated over)
    train_loader, eval_loader = get_dataloaders(
        trainer.data_args,
        trainer.args,
        train_data_paths,
        eval_data_paths,
        feature_map,
    )

    print(f"\nEvaluating on train set (time index:{time_indices_train})....\n")
    trainer.set_train_dataloader(train_loader)
    # Defining temporarily the the train data loader for evaluation
    trainer.set_eval_dataloader(train_loader)
    train_metrics = trainer.evaluate(metric_key_prefix=DatasetType.train.value)
    trainer.wipe_memory()
    print(f"***** {DatasetType.train.value} results (time index): {time_indices_train})*****\n")
    for key in sorted(train_metrics.keys()):
        print("  %s = %s" % (key, str(train_metrics[key])))

    print(f"\nEvaluating on test set (time index:{time_index_eval})....\n")
    trainer.set_eval_dataloader(eval_loader)
    # Defining temporarily the the train data loader for evaluation
    trainer.set_eval_dataloader(train_loader)
    eval_metrics = trainer.evaluate(metric_key_prefix=DatasetType.eval.value)
    trainer.wipe_memory()
    print(f"***** {DatasetType.eval.value} results (time index): {time_index_eval})*****\n")
    for key in sorted(eval_metrics.keys()):
        print("\t  %s = %s" % (key, str(eval_metrics[key])))


def incremental_fit_and_evaluate(trainer, start_time_window_index, final_time_window_index):
    for time_index in range(start_time_window_index, final_time_window_index + 1):
        fit_and_evaluate(trainer, time_index, time_index)
