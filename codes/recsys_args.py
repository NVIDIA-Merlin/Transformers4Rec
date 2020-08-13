
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

    # misc
    fast_test: bool = field(default=False, metadata={"help": "Quick test by running only one loop."})

    log_predictions: bool = field(default=False, metadata={"help": "Logs all predictions, labels and metadata features for test sets in parquet files."})
    

@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to dataset."
        },
    )
    start_date: Optional[str] = field(
        default=None,
        metadata={
            "help": "start date of the data."
        },
    )
    end_date: Optional[str] = field(
        default=None,
        metadata={
            "help": "end date of the data."
        },
    )
    pad_token: Optional[int] = field(
        default=0, metadata={"help": "pad token"}
    )
    max_seq_len: Optional[int] = field(
        default=1024, metadata={"help": "maximum sequence length; it is used to create Positional Encoding in Transfomrer"}
    )
    # args for selecting which engine to use
    engine: Optional[str] = field(
        default='pyarrow', metadata={"help": "Parquet data loader engine. "
            "'pyarrow': read whole parquet into memory. 'petastorm': read chunck by chunck"
        }
    )
    # args for petastorm
    reader_pool_type: Optional[str] = field(
        default='thread', metadata={"help": "A string denoting the reader pool type. \
            Should be one of ['thread', 'process', 'dummy'] denoting a thread pool, \
                process pool, or running everything in the master thread. Defaults to 'thread'"}
    )
    workers_count: Optional[int] = field(
        default=10, metadata={"help": "An int for the number of workers to use in the reader pool. \
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
        default="cross_entropy", metadata={"help": "Type of Loss function: either 'cross_entropy' OR 'margin_hinge'"}
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
        default=1.0, metadata={"help": "margin value for margin-hinge loss"}
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
        default=1.0, metadata={"help": "rescale hinge loss to match with cross entropy loss"}
    )
