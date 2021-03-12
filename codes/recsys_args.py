
from typing import Any, Callable, Dict, List, NewType, Tuple, Optional
from dataclasses import dataclass, field

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    TrainingArguments as HfTrainingArguments
)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments(HfTrainingArguments):
    validate_every: int = field(default=-1, 
        metadata={"help": "Run validation set every this epoch. "
            "-1 means no validation is used (default: -1)"}
    )

    eval_on_test_set: bool = field(default=False, metadata={"help": "Evaluate on test set (by default, evaluates on the validation set)."})

    compute_metrics_each_n_steps: int = field(default=1, metadata={"help": "Log metrics each n steps (for train, validation and test sets)"})

    log_predictions: bool = field(default=False, metadata={"help": "Logs predictions, labels and metadata features each --compute_metrics_each_n_steps (for test set)."})
    log_attention_weights: bool = field(default=False, metadata={"help": "Logs the inputs and attention weights each --compute_metrics_each_n_steps (only test set)"})    

    experiments_group: str = field(default="default", metadata={"help": "Name of the Experiments Group, for organizing job runs logged on W&B"})

    learning_rate_schedule: str = field(default="constant_with_warmup", metadata={"help": "Learning Rate schedule (restarted for each training day). Valid values: constant_with_warmup | linear_with_warmup | cosine_with_warmup"})

    learning_rate_warmup_steps: int = field(default=0, metadata={"help": "Number of steps to linearly increase the learning rate from 0 to the specified initial learning rate schedule. Valid for --learning_rate_schedule = constant_with_warmup | linear_with_warmup | cosine_with_warmup"})
    learning_rate_num_cosine_cycles_by_epoch: float = field(default=1.25, metadata={"help": "Number of cycles for by epoch when --learning_rate_schedule = cosine_with_warmup. The number of waves in the cosine schedule (e.g. 0.5 is to just decrease from the max value to 0, following a half-cosine)."})

    shuffle_buffer_size: int = field(default=0, metadata={"help": "Number of samples to keep in the buffer for shuffling. shuffle_buffer_size=0 means no shuffling"})

    pyprof: bool = field(default=False, metadata={"help": "Enables pyprof logging to inspect with NSights System and DLProf pluging for Tensorboard. Warning: It slows down training, so it should be used only for profiling"})    

    pyprof_start_step: int = field(default=0, metadata={"help": "Start step to profile with PyProf"})
    pyprof_stop_step: int = field(default=0, metadata={"help": "Stop step to profile with PyProf"})

    predict_top_k: int = field(default=10, metadata={"help": "Truncate recommendation list to the highest top-K predicted items (do not affect evaluation metrics computation)"})

    eval_steps_on_train_set: int = field(default=20, metadata={"help": "Number of steps to evaluate on train set (which is usually large)"})

    

    

    

@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default='',
        metadata={
            "help": "Path to dataset."
        },
    )
    
    pad_token: Optional[int] = field(
        default=0, metadata={"help": "pad token"}
    )
    mask_token: Optional[int] = field(
        default=0, metadata={"help": "mask token"}
    )

    # args for selecting which engine to use
    data_loader_engine: Optional[str] = field(
        default='pyarrow', metadata={"help": "Parquet data loader engine. "
            "'nvtabular': GPU-accelerated parquet data loader from NVTabular, 'pyarrow': read whole parquet into memory. 'petastorm': read chunck by chunck"
        }
    )
    
    nvt_part_mem_fraction: Optional[float] = field(
        default=0.1, metadata={"help": "Percentage of GPU to allocate for NVTabular dataset / dataloader"}
    )
    
    # args for petastorm
    petastorm_reader_pool_type: Optional[str] = field(
        default='thread', metadata={"help": "A string denoting the reader pool type. \
            Should be one of ['thread', 'process', 'dummy'] denoting a thread pool, \
                process pool, or running everything in the master thread. Defaults to 'thread'"}
    )
    workers_count: Optional[int] = field(
        default=8, metadata={"help": "An int for the number of workers to use in the reader pool. \
            This only is used for the thread or process pool"}
    )    

    feature_config: Optional[str] = field(
        default="config/recsys_input_feature.yaml",
        metadata={"help": "yaml file that contains feature information (columns to be read from Parquet file, its dtype, etc)"}
    )
    feature_prefix_neg_sample: Optional[str] = field(
        default="_neg_",
        metadata={"help": "prefix of the column name in input parquet file for negative samples"}
    )

    
    session_seq_length_max: Optional[int] = field(
        default=20, metadata={"help": "The maximum length of a session (for padding and trimming). For sequential recommendation, this is the maximum length of the sequence"}
    )

    session_aware: bool = field(default=False, metadata={"help": "Configure the pipeline for session-aware recommendation, where the model can access information of past users sessions. For that, each session feature might "
                                                                "have a correspondent feature for past sessions with a standard prefix (e.g. --session_aware_features_prefix 'bef_', means that values for current session will be in 'sess_pid_seq' feature and "
                                                                " past sessions interactions in 'bef_sess_pid_seq'). Those past session features need also to be included in the features config (yaml)"})

    session_aware_features_prefix: Optional[str]  = field(
        default="bef_", metadata={"help": "how to compute similarity of sequences for negative sampling: 'cosine' OR 'concat_mlp'"}
    )      

    session_aware_past_seq_length_max: int = field(
        default=20, metadata={"help": "For session-aware recommendation, this is the length of the past interactions sessions"}
    )    

    start_time_window_index: Optional[int] = field(
        default=1,
        metadata={
            "help": "The start time window index to use for training & evaluation."
                    "The framework expects data to be split in contiguous non-overlapping time windows, with fixed time units (e.g. days, hours)."
                    
        },
    )

    final_time_window_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "The final time window index to use for training & evaluation."
                    "The framework expects data to be split in contiguous non-overlapping time windows, with fixed time units (e.g. days, hours)."
                    
        },
    )

    time_window_folder_pad_digits: Optional[int] = field(
        default=4,
        metadata={
            "help": "The framework expects data to be split in contiguous non-overlapping time windows, with fixed time units (e.g. days, hours)."
                    "Each time window is represented as a folder within --data_path, named with padded zeros on the left, according to the number"
                    " of digits defined in --time_window_folder_digits. For example, for --time_window_folder_digits 4, folders should be: '0001', '0002', ..."
                    
        },
    )

    no_incremental_training: bool = field(default=False, metadata={"help": "Indicates whether the model should be trained incrementally over time (False) or trained using only the time window (True) defined in --training_time_window_size"})    


    training_time_window_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "Window size of time units (e.g. days or hours) to use for training, when --no_incremental_training."
                    "The framework expects data to be split in contiguous non-overlapping time windows, with fixed time units (e.g. days, hours)."
                    
        },
    )


    @property
    def total_seq_length(self) -> int:
        """
        The total sequence length = session length + past session interactions (if --session_aware)
        """
        total_sequence_length = self.session_seq_length_max
        #For session aware, increments with the length of past interactions features
        if self.session_aware:
            total_sequence_length += self.session_aware_past_seq_length_max 
        return total_sequence_length



@dataclass
class ModelArguments:

    # args for Hugginface default ones

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # args for RecSys Meta model
    inp_merge: Optional[str] = field(
        default="mlp", metadata={"help": "input merge mechanism: 'mlp' OR 'attn'"}
    )

    loss_type: Optional[str] = field(
        default="cross_entropy", metadata={"help": "Type of Loss function: either 'cross_entropy', 'top1', 'top1_max', 'bpr', 'bpr_max_reg'"}
    )
    model_type: Optional[str] = field(
        default='transfoxl',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    similarity_type: Optional[str] = field(
        default="concat_mlp", metadata={"help": "how to compute similarity of sequences for negative sampling: 'cosine' OR 'concat_mlp'"}
    )
    tf_out_activation: Optional[str] = field(
        default="relu", metadata={"help": "transformer output activation: 'tanh' OR 'relu'"}
    )
    margin_loss: Optional[float] = field(
        default=0.0, metadata={"help": "margin value for margin-hinge loss"}
    )
    mlm: bool = field(default=False, metadata={"help": "Use Masked Language Modeling (Cloze objective) for training."})

    mlm_probability: Optional[float] = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask (set target) from an original sequence. There is a hard constraint that ensures that each sequence have at least one target (masked) and one non-masked item, for effective learning. Thus, if the sequence has more than 2 items, this is the probability of the additional items to be masked"}
    )

    # args for Transformers or RNNs

    d_model: Optional[int] = field(
        default=256, metadata={"help": "size of hidden states (or internal states) for RNNs and Transformers"}
    )
    n_layer: Optional[int] = field(
        default=12, metadata={"help": "number of layers for RNNs and Transformers"}
    )
    n_head: Optional[int] = field(
        default=4, metadata={"help": "number of attention heads for Transformers"}
    )
    layer_norm_eps: Optional[float] = field(
        default=1e-12, metadata={"help": "The epsilon used by the layer normalization layers for Transformers"}
    )
    initializer_range: Optional[float] = field(
        default=0.02, metadata={"help": "The standard deviation of the truncated_normal_initializer for initializing all weight matrices for Transformers"}
    )
    hidden_act: Optional[str] = field(
        default="gelu", metadata={"help": "The non-linear activation function (function or string) in Transformers. 'gelu', 'relu' and 'swish' are supported."}
    )
    dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout probability for all fully connected layers in the embeddings, encoder, and decoders for Transformers and RNNs"}
    )
    all_rescale_factor: Optional[float] = field(
        default=1.0, metadata={"help": "rescale cross entropy loss to match with hinge-loss"}
    )
    neg_rescale_factor: Optional[float] = field(
        default=0.0, metadata={"help": "rescale hinge loss to match with cross entropy loss"}
    )

    loss_scale_factor: Optional[float] = field(
        default=1.0, metadata={"help": "Rescale the loss. The scale of different losses types are very different (e.g. cross_entropy > bpr_max > top1_max) and this scaling might help to avoid underflow with fp16"}
    )

    disable_positional_embeddings: bool = field(default=False, metadata={"help": "Disable usage of (sum) positional embeddings with items embeddings."})

    attn_type: str = field(
        default="uni", metadata={"help": "The type of attention. Use 'uni' for Causal LM and 'bi' for Masked LM"}
    )

    # args for XLNET
    summary_type: Optional[str] = field(
        default = 'last', metadata = {
            "help": "How to summarize the vector representation of the sequence'last', 'first', 'mean', 'attn' are supported"})

    # args for Reformer : chunked lsh-attention and axial positional encoding
    axial_pos_shape_first_dim: Optional[int] = field(
        default = 4, metadata = {"help": "[Reformer] First dimension of axial position encoding "}
    )

    attn_chunk_length: Optional[int] = field(
        default = 4, metadata = {"help": " [Reformer] Length of chunk (for bothLocalSelfAttention or LSHSelfAttention layers) which attends to itself. Chunking reduces memory complexity from sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk length (chunked self attention)."}
    )

    num_chunks_before: Optional[int] = field(
        default = 1, metadata = {"help": "[Reformer] Number of previous neighbouring chunks to attend in LocalSelfAttention or LSHSelfAttention layer to itself."}
    )

    num_chunks_after: Optional[int] = field(
        default = 0, metadata = {"help": "[Reformer] Number of following neighbouring chunks to attend in LocalSelfAttention or LSHSelfAttention layer to itself(for Masked LM)."}
    )

    chunk_size_feed_forward: Optional[int] = field(
        default = 0, metadata = {"help": "[Reformer] The chunk size of all feed forward layers in the residual attention blocks. A chunk size of 0 means that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes n < sequence_length embeddings at a time."}
    )    

    lsh_num_hashes: Optional[int] = field(
        default = 2, metadata = {"help": "[Reformer] Number of hashing rounds (e.g. number of random rotations) in Local Sensitive Hashing scheme. The higher `num_hashes`, the more accurate the `LSHSelfAttention` becomes, but also the more memory and time intensive the hashing becomes."}
    )

    eval_on_last_item_seq_only: bool = field(default=False, metadata={"help": "Evaluate metrics only on predictions for the last item of the sequence (rather then evaluation for all next-item predictions)."})
    train_on_last_item_seq_only: bool = field(default=False, metadata={"help": "Train only for predicting the last item of the sequence (rather then training to predict for all next-item predictions) (only for Causal LM)."})


    use_ohe_item_ids_inputs : bool = field(default=False, metadata={"help": "Uses the one-hot encoding of the item ids as inputs followed by a MLP layer, instead of using item embeddings"})	
    
    constrained_embeddings : bool = field(default=False, metadata={"help": "Use the weight matrix of the output layer as the embedding layer of item_ids. Requires the item id embedding to have the same dimension as the second-to-last layer."})	
    
    mf_constrained_embeddings : bool = field(default=False, metadata={"help": "Performs a matrix factorization (dot product multiplication) operation between the Transformers output and the item embeddings, to produce output logits for all items. Requires the item id embeddings to have the same dimension as the second-to-last layer."})	

    mf_constrained_embeddings_disable_bias: bool = field(default=False, metadata={"help": "Disable the bias addition for --mf_constrained_embeddings."})	
    item_embedding_with_dmodel_dim: bool = field(default=False, metadata={"help": "Makes the item embedding have the same dimension of d_model (this is default for --constrained_embeddings and --mf_constrained_embeddings). Just to debug the effect of constrained embeddings"})	

    item_embedding_dim: Optional[int] = field(
        default=None, metadata={"help": "Dimension of the item embedding. If it is None, a heuristic method used to define the dimension based on items cardinality. "
                            "If --mf_constrained_embeddings or --constrained_embeddings are enabled, the output of transformers (dimension defined by --d_model) will "
                            "be projected to the same dimension as the item embedding (tying embedding), just before the output layer."})


    
    bpr_max_reg_lambda: Optional[float] = field(	
        default=0.0, metadata={"help": "regularization hyper-param of the loss function:  BPR-MAX"}	
    )


    negative_sampling: bool = field(default=False, metadata={"help": "Compute negative samples for pairwise losses"}
                                     )
    
    neg_sampling_store_size: Optional[int] = field(
        default=1000000, metadata={"help": "number of samples to store in cache"}
    )

    neg_sampling_extra_samples_per_batch: Optional[int] = field(
        default=32, metadata={"help": "Number of negative samples to add to mini-batch samples"}
    )

    
    neg_sampling_alpha: Optional[float] = field(
        default=0.0, metadata={"help": "parameter of sampling: = 0 and = 1 are uniform and popularity-based sapling respectively "}
    )

                                                           
