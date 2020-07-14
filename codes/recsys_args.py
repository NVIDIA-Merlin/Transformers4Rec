
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

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default='xlnet',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    fast_test: bool = field(default=False, metadata={"help": "Quick test by running only one loop."})
