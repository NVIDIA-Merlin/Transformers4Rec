import glob
import os

import numpy as np
import torch
from torch.nn.modules.loss import _WeightedLoss


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
    aot_metrics: dict
        The average over time of ranking metrics.
    """
    aot_metrics = {}
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
                print(" %s = %s" % (key.replace("at_", "@"), str(eval_metrics[key])))
                if "AOT_" + key.replace("at_", "@") in aot_metrics:
                    aot_metrics["AOT_" + key.replace("_at_", "@")] += [eval_metrics[key]]
                else:
                    aot_metrics["AOT_" + key.replace("_at_", "@")] = [eval_metrics[key]]

        # free GPU for next day training
        trainer.wipe_memory()

    return aot_metrics


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    """Constructor for cross-entropy loss with label smoothing

    Parameters:
    ----------
    smoothing: float
        The label smoothing factor. it should be between 0 and 1.
    weight: torch.Tensor
        The tensor of weights given to each class.
    reduction: str
        Specifies the reduction to apply to the output,
        possible values are `none` | `sum` | `mean`

    Adapted from https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch
    """

    def __init__(self, weight: torch.Tensor = None, reduction: str = "mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing: float = 0.0):
        assert 0 <= smoothing < 1, f"smoothing factor {smoothing} should be between 0 and 1"
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(
            targets, inputs.size(-1), self.smoothing
        )
        lsm = inputs

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError(
                f"{self.reduction} is not supported, please choose one of the following values"
                " [`sum`, `none`, `mean`]"
            )

        return loss
