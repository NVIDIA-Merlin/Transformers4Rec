from dataclasses import dataclass

from transformers import TFTrainingArguments, TrainingArguments


@dataclass
class T4RecTrainerConfig(TrainingArguments):
    def __init__(
        self,
        output_dir: str,
        avg_session_length: int,
        eval_on_last_item_seq_only: bool = True,
        validate_every: int = -1,
        predict_top_k: int = 10,
        log_predictions: bool = False,
        log_attention_weights: bool = False,
        learning_rate_schedule: str = "constant_with_warmup",
        learning_rate_num_cosine_cycles_by_epoch: float = 1.25,
        experiments_group: str = "default",
        **kwargs
    ):
        """
        Parameters:
        -----------
        output_dir: str
            The output directory where the model predictions and checkpoints will be written.
        avg_session_length : int
            the avg. session length (rounded up to the next int),
            It is used to estimate the number of interactions from the batch_size (# sessions)
            so that the tensor that accumulates all predictions is sufficient
            to concatenate all predictions
        eval_on_last_item_seq_only : Optional[bool], bool
            Evaluate metrics only on predictions for the last item of the sequence
            (rather then evaluation for all next-item predictions).
            by default True
        validate_every: Optional[int], int
            Run validation set every this epoch.
            -1 means no validation is used
            by default -1
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
        learning_rate_schedule: Optional[str], str
            Learning Rate schedule (restarted for each training day).
            Valid values: constant_with_warmup | linear_with_warmup | cosine_with_warmup
            by defaut constant_with_warmup
        learning_rate_num_cosine_cycles_by_epoch : Optional[int], int
            Number of cycles for by epoch when --learning_rate_schedule = cosine_with_warmup.
            The number of waves in the cosine schedule
            (e.g. 0.5 is to just decrease from the max value to 0, following a half-cosine).
            by default 1.25
        experiments_group: Optional[str], str
            Name of the Experiments Group, for organizing job runs logged on W&B
            by default "default"
        """

        self.output_dir = output_dir
        self.avg_session_length = avg_session_length
        self.eval_on_last_item_seq_only = eval_on_last_item_seq_only
        self.validate_every = validate_every
        self.predict_top_k = predict_top_k
        self.log_predictions = log_predictions
        self.log_attention_weights = log_attention_weights
        self.learning_rate_schedule = learning_rate_schedule
        self.learning_rate_num_cosine_cycles_by_epoch = learning_rate_num_cosine_cycles_by_epoch
        self.experiments_group = experiments_group

        super().__init__(self.output_dir, **kwargs)


class T4RecTrainerConfigTF(T4RecTrainerConfig, TFTrainingArguments):
    """
    Prepare Training arguments for TFTrainer,
    Inherit arguments from T4RecTrainerConfig and TFTrainingArguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
