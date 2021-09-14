import os

import numpy as np


def list_files(startpath):
    """
    Util function to print the nested structre of a directory
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            print("{}{}".format(subindent, f))


def visualize_response(batch, response, top_k):
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
    sessions = batch["session_id"].drop_duplicates().values
    predictions = response.as_numpy("output")
    top_preds = np.argpartition(predictions, -top_k, axis=1)[:, -top_k:]
    for session, next_items in zip(sessions, top_preds):
        print(
            "- Top-%s predictions for session `%s`: %s\n"
            % (top_k, session, " || ".join([str(e) for e in next_items]))
        )


def fit_and_evaluate(
    trainer, start_time_index, end_time_index, input_dir="./preproc_sessions_by_day_ts"
):
    """
    Util function for time-window based fine-tuning using the T4rec Trainer class.
    Iteratively train using data of a given index and evaluate on the validation data
    of the following index.

    Parameters
    ----------
    start_time_index: int
        the start index for training, it should match the partitions of the data directory
    end_time_index: int
        the end index for training, it should match the partitions of the  data directory
    input_dir: str
        The input directory where the parquet files were saved based on partition column

    Returns
    -------
    aot_metrics: dict
        The average over time of ranking metrics.
    """
    import glob
    import os

    aot_metrics = {}
    for time_index in range(start_time_index, end_time_index + 1):
        # 1. Set data
        time_index_train = time_index
        time_index_eval = time_index + 1
        train_paths = glob.glob(os.path.join(input_dir, f"{time_index_train}/train.parquet"))
        eval_paths = glob.glob(os.path.join(input_dir, f"{time_index_eval}/valid.parquet"))

        # 2. Train on day related to time_index
        print("\n" + "*" * 27)
        print("Launch training for day %s:" % time_index)
        print("*" * 27 + "\n")
        trainer.train_dataset_or_path = train_paths
        trainer.reset_lr_scheduler()
        trainer.train()
        trainer.state.global_step += 1

        # 3. Evaluate on the following day
        trainer.eval_dataset_or_path = eval_paths
        eval_metrics = trainer.evaluate(metric_key_prefix="eval")
        print("\t\t" * 3 + "*" * 30)
        print("\t\t" * 3 + "Evaluation results for day %s:" % time_index_eval)
        print("\t\t" * 3 + "*" * 30)
        for key in sorted(eval_metrics.keys()):
            if "at_" in key:
                print("\t\t" * 3 + " %s = %s" % (key.replace("at_", "@"), str(eval_metrics[key])))
                if "AOT_" + key.replace("at_", "@") in aot_metrics:
                    aot_metrics["AOT_" + key.replace("_at_", "@")] += [eval_metrics[key]]
                else:
                    aot_metrics["AOT_" + key.replace("_at_", "@")] = [eval_metrics[key]]

        # free GPU for next day training
        trainer.wipe_memory()

    return aot_metrics
