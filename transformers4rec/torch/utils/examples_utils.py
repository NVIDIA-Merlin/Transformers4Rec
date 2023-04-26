import gc
import glob
import os

import numpy as np
import torch


def list_files(startpath):
    """
    Util function to print the nested structure of a directory
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            print("{}{}".format(subindent, f))


def visualize_response(batch, response, top_k, session_col="session_id"):
    """
    Util function to extract top-k encoded item-ids from logits

    Parameters
    ----------
    batch : cudf.DataFrame
        the batch of raw data sent to triton server.
    response: tritonclient.grpc.InferResult
        the response returned by grpc client.
    top_k: int
        the `top_k` top items to retrieve from predictions.
    """
    sessions = batch[session_col].drop_duplicates().values
    predictions = response.as_numpy("output")
    top_preds = np.argpartition(predictions, -top_k, axis=1)[:, -top_k:]
    for session, next_items in zip(sessions, top_preds):
        print(
            "- Top-%s predictions for session `%s`: %s\n"
            % (top_k, session, " || ".join([str(e) for e in next_items]))
        )


def fit_and_evaluate(trainer, start_time_index, end_time_index, input_dir):
    """
    Util function for time-window based fine-tuning using the T4rec Trainer class.
    Iteratively train using data of a given index and evaluate on the validation data
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
    indexed_by_time_metrics: dict
        The dictionary of ranking metrics: each item is the list of scores over time indices.
    """
    indexed_by_time_metrics = {}
    for time_index in range(start_time_index, end_time_index + 1):
        # 1. Set data
        time_index_train = time_index
        time_index_eval = time_index + 1
        train_paths = glob.glob(os.path.join(input_dir, f"{time_index_train}/train.parquet"))
        eval_paths = glob.glob(os.path.join(input_dir, f"{time_index_eval}/valid.parquet"))

        # 2. Train on train data of time_index
        print("\n***** Launch training for day %s: *****" % time_index)
        trainer.train_dataset_or_path = train_paths
        trainer.reset_lr_scheduler()
        trainer.train()

        # 3. Evaluate on valid data of time_index+1
        trainer.eval_dataset_or_path = eval_paths
        eval_metrics = trainer.evaluate(metric_key_prefix="eval")
        print("\n***** Evaluation results for day %s:*****\n" % time_index_eval)
        for key in sorted(eval_metrics.keys()):
            if "at_" in key:
                print(" %s = %s" % (key.replace("_at_", "@"), str(eval_metrics[key])))
                if "indexed_by_time_" + key.replace("_at_", "@") in indexed_by_time_metrics:
                    indexed_by_time_metrics["indexed_by_time_" + key.replace("_at_", "@")] += [
                        eval_metrics[key]
                    ]
                else:
                    indexed_by_time_metrics["indexed_by_time_" + key.replace("_at_", "@")] = [
                        eval_metrics[key]
                    ]

        # free GPU for next day training
        wipe_memory()

    return indexed_by_time_metrics


def wipe_memory():
    gc.collect()
    torch.cuda.empty_cache()
