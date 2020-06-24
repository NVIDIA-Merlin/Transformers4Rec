"""
How torun : 
    python example_recsys_runner.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset/ecommerce_preproc_2019-10/ecommerce_preproc.parquet/

"""
import os
import math
import glob
import logging
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NewType, Tuple, Optional

import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader
from petastorm.unischema import UnischemaField
from petastorm.unischema import Unischema
from petastorm.codecs import NdarrayCodec

from custom_trainer import RecSysTrainer
from custom_xlnet_config import XLNetConfig
from custom_modeling_xlnet import RecSysXLNetLMHeadModel as XLNetLMHeadModel

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)

# TODO: they are used for training progress tracking
# currently these are arbitrary number. It needs to be pre-computed 
TRAIN_DATA_LEN = 10000
EVAL_DATA_LEN = 10000


def f_feature_extract(inputs):
    """
    This function will be used inside of trainer.py (_training_step) right before being 
    passed inputs to a model. 
    """
    product_seq = inputs["sess_pid_seq"][:-1].long()
    category_seq = inputs["sess_dtime_seq"][:-1].long()
    time_delta_seq = inputs["sess_ccid_seq"][:-1].long()
    labels = inputs["sess_pid_seq"][1:].long()
    
    return product_seq, category_seq, time_delta_seq, labels


# A schema that we use to read specific columns from parquet data file
recsys_schema_small = [
    UnischemaField('sess_pid_seq', np.int64, (None,), None, True),
    UnischemaField('sess_dtime_seq', np.int64, (None,), None, True),
    UnischemaField('sess_ccid_seq', np.int64, (None,), None, True),
]

# Full Schema
# recsys_schema_full = [
#     UnischemaField('user_idx', np.int, (), None, True),
#     #   UnischemaField('user_session', str_, (), None, True),
#     UnischemaField('sess_seq_len', np.int, (), None, False),
#     UnischemaField('session_start_ts', np.int64, (), None, True),
#     UnischemaField('user_seq_length_bef_sess', np.int, (), None, False),
#     UnischemaField('user_elapsed_days_bef_sess', np.float, (), None, True),
#     UnischemaField('user_elapsed_days_log_bef_sess_norm', np.double, (), None, True),
#     UnischemaField('sess_pid_seq', np.int64, (None,), None, True),
#     UnischemaField('sess_etime_seq', np.int64, (None,), None, True),
#     UnischemaField('sess_etype_seq', np.int, (None,), None, True),
#     UnischemaField('sess_csid_seq', np.int, (None,), None, True),
#     UnischemaField('sess_ccid_seq', np.int, (None,), None, True),
#     UnischemaField('sess_bid_seq', np.int, (None,), None, True),
#     UnischemaField('sess_price_seq', np.float, (None,), None, True),
#     UnischemaField('sess_dtime_seq', np.float, (None,), None, True),
#     UnischemaField('sess_product_recency_seq', np.float, (None,), None, True),
#     UnischemaField('sess_relative_price_to_avg_category_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_hour_sin_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_hour_cos_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_month_sin_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_month_cos_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_dayofweek_sin_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_dayofweek_cos_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_dayofmonth_sin_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_dayofmonth_cos_seq', np.float, (None,), None, True),
#     UnischemaField('user_pid_seq_bef_sess', np.int64, (None,), None, True),
#     UnischemaField('user_etime_seq_bef_sess', np.int64, (None,), None, True),
#     UnischemaField('user_etype_seq_bef_sess', np.int, (None,), None, True),
#     UnischemaField('user_csid_seq_bef_sess', np.int, (None,), None, True),
#     UnischemaField('user_ccid_seq_bef_sess', np.int, (None,), None, True),
#     UnischemaField('user_bid_seq_bef_sess', np.int, (None,), None, True),
#     UnischemaField('user_price_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_dtime_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_product_recency_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_relative_price_to_avg_category_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_hour_sin_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_hour_cos_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_month_sin_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_month_cos_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_dayofweek_sin_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_dayofweek_cos_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_dayofmonth_sin_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_dayofmonth_cos_seq_bef_sess', np.float, (None,), None, True),
# ]

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class DataLoaderWithLen(DataLoader):
    def __init__(self, *args, **kwargs):
        if 'drop_last' not in kwargs:
            self.drop_last = False
        else:
            self.drop_last = kwargs.pop('drop_last')

        if 'len' not in kwargs:
            self.len = 0
        else:
            self.len = kwargs.pop('len')

        super(DataLoaderWithLen, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.len

    def _iter_impl(self):
        """
        Override original method at DataLoader class to enable drop_last option. 
        Without drop_last, we have an error that caused by different batch-sizes in multi-gpu pipeline
        similar issue: https://github.com/huggingface/transformers/issues/1220
        """
        # As we iterate over incoming samples, we are going to store them in `self._batch_acc`, until we have a batch of
        # the requested batch_size ready.

        keys = None
        if self.shuffling_queue_capacity > 0:
            # We can not know what is the reasonable number to use for the extra capacity, so we set a huge number
            # and give up on the unbound growth protection mechanism.
            min_after_dequeue = self.shuffling_queue_capacity - 1
            self._shuffling_buffer = RandomShufflingBuffer(self.shuffling_queue_capacity,
                                                           min_after_retrieve=min_after_dequeue,
                                                           extra_capacity=100000000)
        else:
            self._shuffling_buffer = NoopShufflingBuffer()

        for row in self.reader:
            # Default collate does not work nicely on namedtuples and treat them as lists
            # Using dict will result in the yielded structures being dicts as well
            row_as_dict = row._asdict()

            keys = row_as_dict.keys()

            # Promote some types that are incompatible with pytorch to be pytorch friendly.
            _sanitize_pytorch_types(row_as_dict)

            # Add rows to shuffling buffer
            if not self.reader.is_batched_reader:
                self._shuffling_buffer.add_many([row_as_dict])
            else:
                # Transposition:
                #   row_as_dict:        {'a': [1,2,3], 'b':[4,5,6]}
                #   row_group_as_tuple: [(1, 4), (2, 5), (3, 6)]
                # The order within a tuple is defined by key order in 'keys'
                row_group_as_tuple = list(zip(*(row_as_dict[k] for k in keys)))

                # Adding data as 'row-by-row' into a shuffling buffer. This is a pretty
                # slow implementation though. Probably can comeup with a faster way to shuffle,
                # perhaps at the expense of a larger memory consumption...
                self._shuffling_buffer.add_many(row_group_as_tuple)

            # _yield_batches will emit as much batches as are allowed by the shuffling_buffer (RandomShufflingBuffer
            # will avoid underflowing below a certain number of samples to guarantee some samples decorrelation)
            for batch in self._yield_batches(keys):
                yield batch

        # Once reader can not read new rows, we might still have a bunch of rows waiting in the shuffling buffer.
        # Telling shuffling buffer that we are finished allows to deplete the buffer completely, regardless its
        # min_after_dequeue setting.
        self._shuffling_buffer.finish()

        for batch in self._yield_batches(keys):
            yield batch

        # Yield the last and partial batch
        if self._batch_acc and not self.drop_last:
            yield self.collate_fn(self._batch_acc)

def get_filenames(data_paths):
    paths = [['file://' + p for p in glob.glob(path + "/*.parquet")] for path in data_paths]
    return list(itertools.chain.from_iterable(paths))


@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to dataset."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


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

    train_data_path = [
        data_args.data_path + "session_start_date=2019-10-01",
        data_args.data_path + "session_start_date=2019-10-02",
    ]

    eval_data_path = [
        data_args.data_path + "session_start_date=2019-10-03",
    ]

    train_data_path = get_filenames(train_data_path)
    eval_data_path = get_filenames(eval_data_path)

    train_loader = DataLoaderWithLen(
        make_batch_reader(train_data_path, 
            num_epochs=None,
            schema_fields=recsys_schema_small,
        ), 
        batch_size=training_args.per_device_train_batch_size,
        len=TRAIN_DATA_LEN,
        drop_last=True,
    )

    eval_loader = DataLoaderWithLen(
        make_batch_reader(eval_data_path, 
            num_epochs=None,
            schema_fields=recsys_schema_small,
        ), 
        batch_size=training_args.per_device_eval_batch_size,
        len=EVAL_DATA_LEN,
        drop_last=True,
    )

    config = XLNetConfig(
        product_vocab_size=300000, 
        category_vocab_size=1000, 
        brand_vocab_size=500,         
        d_model=1024,
        n_layer=24,
        n_head=16,
        d_inner=4096,
        ff_activation="gelu",
        untie_r=True,
        attn_type="bi",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        dropout=0.1,
    )

    if model_args.model_name_or_path:
        model = XLNetLMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = XLNetLMHeadModel(config)

    trainer = RecSysTrainer(
        train_loader=train_loader, 
        eval_loader=eval_loader,        
        model=model,
        args=training_args,
        f_feature_extract=f_feature_extract)

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

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