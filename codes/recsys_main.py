"""
How torun : 
    CUDA_VISIBLE_DEVICES=0 python main_runner.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset/sessions_with_neg_samples_example/ --per_device_train_batch_size 128 --model_type xlnet
"""
import os
import sys
import math
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    set_seed,
)

from recsys_models import get_recsys_model
from recsys_meta_model import RecSysMetaModel
from recsys_trainer import RecSysTrainer
from recsys_metrics import EvalMetrics
from recsys_args import DataArguments, ModelArguments, TrainingArguments
from recsys_data import (
    f_feature_extract_pos,
    f_feature_extract_posneg, 
    fetch_data_loaders,
    get_avail_data_dates
)


logger = logging.getLogger(__name__)

# this code use Version 3
assert sys.version_info[0] > 2

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

    # embedding size
    seq_model, config = get_recsys_model(model_args, data_args, training_args)
    rec_model = RecSysMetaModel(seq_model, config, model_args, data_args)

    f_feature_extract = f_feature_extract_posneg \
        if model_args.loss_type in ['margin_hinge', 'cross_entropy_neg'] else f_feature_extract_pos

    eval_metrics = EvalMetrics()

    trainer = RecSysTrainer(
        model=rec_model,
        args=training_args,
        f_feature_extract=f_feature_extract,
        compute_metrics=eval_metrics,
        fast_test=model_args.fast_test
        )

    data_dates = get_avail_data_dates(data_args)
    results_dates = {}

    for date_idx in range(1, len(data_dates)):
        train_date, eval_date, test_date = data_dates[date_idx - 1], data_dates[date_idx -1], data_dates[date_idx]

        train_loader, eval_loader, test_loader \
            = fetch_data_loaders(data_args, training_args, train_date, eval_date, test_date,
                                 neg_sampling=(model_args.loss_type in ['margin_hinge', 'cross_entropy_neg']))

        trainer.set_rec_train_dataloader(train_loader)
        trainer.set_rec_eval_dataloader(eval_loader)
        trainer.set_rec_test_dataloader(test_loader)

        # Training
        if training_args.do_train:
            logger.info("*** Train (date:{})***".format(train_date))

            model_path = (
                model_args.model_name_or_path
                if model_args.model_name_or_path is not None \
                    and os.path.isdir(model_args.model_name_or_path)
                else None
            )
            trainer.train(model_path=model_path)

        # Evaluation (on testset)
        if training_args.do_eval:
            logger.info("*** Evaluate (date:{})***".format(eval_date))

            eval_output = trainer.predict()
            eval_metrics = eval_output.metrics

            output_eval_file = os.path.join(training_args.output_dir, "eval_results_dates.txt")
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results (date:{})*****".format(eval_date))
                    writer.write("***** Eval results (date:{})*****".format(eval_date))
                    for key in sorted(eval_metrics.keys()):
                        logger.info("  %s = %s", key, str(eval_metrics[key]))
                        writer.write("%s = %s\n" % (key, str(eval_metrics[key])))

            results_dates[eval_date] = eval_metrics
        
    logger.info("train and eval for all dates are done")
    trainer.save_model()

    if training_args.do_eval and trainer.is_world_master():
        
        eval_df = pd.DataFrame.from_dict(results_dates, orient='index')
        np.save(os.path.join(training_args.output_dir, "eval_results_dates.npy"), eval_df)
        eval_avg_days = dict(eval_df.mean())
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_avg.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results (avg over dates)*****")
            for key in sorted(eval_avg_days.keys()):
                logger.info("  %s = %s", key, str(eval_avg_days[key]))
                writer.write("%s = %s\n" % (key, str(eval_avg_days[key])))

    return results_dates


if __name__ == "__main__":
    main()      