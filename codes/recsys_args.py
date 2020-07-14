
from typing import Any, Callable, Dict, List, NewType, Tuple, Optional
from dataclasses import dataclass, field
from transformers import MODEL_WITH_LM_HEAD_MAPPING


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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

    loss_type: Optional[str] = field(
        default="cross_entropy", metadata={"help": "Type of Loss function: either 'cross_entropy' OR 'margin_hinge'"}
    )
    model_type: Optional[str] = field(
        default='xlnet',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    merge_inputs: Optional[str] = field(
        default="elem_add", metadata={"help": "how to merge multiple input sequences: either 'elem_add' OR 'concat_mlp'"}
    )
    similarity_type: Optional[str] = field(
        default="cos", metadata={"help": "how to compute similarity of sequences for negative sampling based margin loss: 'cos'"}
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

    # misc

    fast_test: bool = field(default=False, metadata={"help": "Quick test by running only one loop."})
