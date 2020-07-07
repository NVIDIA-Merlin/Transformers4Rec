"""
How torun : 
    CUDA_VISIBLE_DEVICES=0 python main_runner.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset/sessions_with_neg_samples_example/ --per_device_train_batch_size 128 --model_type xlnet
"""
import os
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NewType, Tuple, Optional

from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

# load transformer model and its configuration classes
from transformers.modeling_xlnet import XLNetModel
from transformers.configuration_xlnet import XLNetConfig
from transformers.modeling_gpt2 import GPT2Model
from transformers.configuration_gpt2 import GPT2Config
from transformers.modeling_longformer import LongformerModel
from transformers.configuration_longformer import LongformerConfig

from trainer import RecSysTrainer
from meta_model import RecSysMetaModel
from utils import wc, get_filenames, get_dataset_len
from metrics import compute_recsys_metrics

from recsys_data_schema import recsys_schema_small, f_feature_extract, vocab_sizes


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def fetch_data_loaders(data_args, training_args):
    d_path = data_args.data_path if data_args.data_path else ''
    
    # TODO: make this at outer-loop for making evaluation based on days-data-partition
    train_data_path = [
        d_path + "session_start_date=2019-10-01",
        d_path + "session_start_date=2019-10-01",
    ]

    eval_data_path = [
        d_path + "session_start_date=2019-10-01",
    ]

    train_data_path = get_filenames(train_data_path)
    eval_data_path = get_filenames(eval_data_path)

    train_data_len = get_dataset_len(train_data_path)
    eval_data_len = get_dataset_len(eval_data_path)

    train_loader = DataLoaderWithLen(
        make_batch_reader(train_data_path, 
            num_epochs=None,
            schema_fields=recsys_schema_small,
        ), 
        batch_size=training_args.per_device_train_batch_size,
        len=train_data_len,
    )

    eval_loader = DataLoaderWithLen(
        make_batch_reader(eval_data_path, 
            num_epochs=None,
            schema_fields=recsys_schema_small,
        ), 
        batch_size=training_args.per_device_eval_batch_size,
        len=eval_data_len,
    )
    return train_loader, eval_loader


def create_model(model_args):

    if model_args.model_type == 'xlnet':
        model_cls = XLNetModel
        config = XLNetConfig(
            d_model=512,
            n_layer=12,
            n_head=8,
            d_inner=2048,
            ff_activation="gelu",
            untie_r=True,
            attn_type="bi",
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            dropout=0.1,
        )

    #NOTE: gpt2 and longformer are not fully tested supported yet.

    elif model_args.model_type == 'gpt2':
        model_cls = GPT2Model
        config = GPT2Config(
            d_model=512,
            n_layer=12,
            n_head=8,
            d_inner=2048,
            ff_activation="gelu",
            untie_r=True,
            attn_type="bi",
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            dropout=0.1,
        )

    elif model_args.model_type == 'longformer':
        model_cls = LongformerModel
        config = LongformerConfig(
            d_model=512,
            n_layer=12,
            n_head=8,
            d_inner=2048,
            ff_activation="gelu",
            untie_r=True,
            attn_type="bi",
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            dropout=0.1,
        )

    else:
        raise NotImplementedError

    if model_args.model_name_or_path:
        transformer_model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        transformer_model = model_cls(config)
    
    model = RecSysMetaModel(transformer_model, vocab_sizes, d_model=config.d_model)

    return model


class DataLoaderWithLen(DataLoader):
    def __init__(self, *args, **kwargs):
        if 'len' not in kwargs:
            self.len = 0
        else:
            self.len = kwargs.pop('len')

        super(DataLoaderWithLen, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.len


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


def main():

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    train_loader, eval_loader = fetch_data_loaders(data_args, training_args)

    model = create_model(model_args)

    trainer = RecSysTrainer(
        train_loader=train_loader, 
        eval_loader=eval_loader,        
        model=model,
        args=training_args,
        f_feature_extract=f_feature_extract,
        compute_metrics=compute_recsys_metrics,
        fast_test=model_args.fast_test)

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == "__main__":
    main()      