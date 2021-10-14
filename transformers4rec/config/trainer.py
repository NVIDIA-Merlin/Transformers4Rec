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

from dataclasses import dataclass, field
from typing import Optional

from transformers import TFTrainingArguments, TrainingArguments


@dataclass
class T4RecTrainingArguments(TrainingArguments):
    """
    Class that inherits HF TrainingArguments and add on top of it arguments needed for
    session-based and sequential-based recommendation

    Parameters
    ----------
    shuffle_buffer_size:
    validate_every: Optional[int], int
        Run validation set every this epoch.
        -1 means no validation is used
        by default -1
    eval_on_test_set:
    eval_steps_on_train_set:
    predict_top_k:  Option[int], int
        Truncate recommendation list to the highest top-K predicted items
        (do not affect evaluation metrics computation)
        by default 10
    log_predictions : Optional[bool], bool
        log predictions, labels and metadata features each --compute_metrics_each_n_steps
        (for test set).
        by default False
    log_attention_weights : Optional[bool], bool
        Logs the inputs and attention weights
        each --eval_steps (only test set)"
        bu default False
    learning_rate_num_cosine_cycles_by_epoch : Optional[int], int
        Number of cycles for by epoch when --lr_scheduler_type = cosine_with_warmup.
        The number of waves in the cosine schedule
        (e.g. 0.5 is to just decrease from the max value to 0, following a half-cosine).
        by default 1.25
    experiments_group: Optional[str], str
        Name of the Experiments Group, for organizing job runs logged on W&B
        by default "default"
    """

    max_sequence_length: Optional[int] = field(
        default=None,
        metadata={"help": "maximum length of sequence"},
    )

    shuffle_buffer_size: int = field(
        default=0,
        metadata={
            "help": "Number of samples to keep in the buffer for shuffling."
            "shuffle_buffer_size=0 means no shuffling"
        },
    )

    data_loader_engine: str = field(
        default="nvtabular",
        metadata={
            "help": "Parquet data loader engine. "
            "'nvtabular': GPU-accelerated parquet data loader from NVTabular, 'pyarrow': read whole parquet into memory."
        },
    )

    eval_on_test_set: bool = field(
        default=False,
        metadata={"help": "Evaluate on test set (by default, evaluates on the validation set)."},
    )

    eval_steps_on_train_set: int = field(
        default=20,
        metadata={"help": "Number of steps to evaluate on train set (which is usually large)"},
    )

    predict_top_k: int = field(
        default=10,
        metadata={
            "help": "Truncate recommendation list to the highest top-K predicted items (do not affect evaluation metrics computation)"
        },
    )

    learning_rate_num_cosine_cycles_by_epoch: float = field(
        default=1.25,
        metadata={
            "help": "Number of cycles for by epoch when --learning_rate_schedule = cosine_with_warmup."
            "The number of waves in the cosine schedule "
            "(e.g. 0.5 is to just decrease from the max value to 0, following a half-cosine)."
        },
    )

    log_predictions: bool = field(
        default=False,
        metadata={
            "help": "Logs predictions, labels and metadata features each --compute_metrics_each_n_steps (for test set)."
        },
    )

    compute_metrics_each_n_steps: int = field(
        default=1,
        metadata={"help": "Log metrics each n steps (for train, validation and test sets)"},
    )

    experiments_group: str = field(
        default="default",
        metadata={"help": "Name of the Experiments Group, for organizing job runs logged on W&B"},
    )

    @property
    def place_model_on_device(self):
        """
        Override the method to allow running training on cpu
        """
        if self.device.type == "cuda":
            return True
        return False


class T4RecTrainingArgumentsTF(T4RecTrainingArguments, TFTrainingArguments):
    """
    Prepare Training arguments for TFTrainer,
    Inherit arguments from T4RecTrainingArguments and TFTrainingArguments
    """

    pass
