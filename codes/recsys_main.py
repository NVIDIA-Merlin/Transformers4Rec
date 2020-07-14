"""
How torun : 
    CUDA_VISIBLE_DEVICES=0 python main_runner.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset/sessions_with_neg_samples_example/ --per_device_train_batch_size 128 --model_type xlnet
"""
import os
import math
import logging

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from recsys_models import get_recsys_model
from recsys_meta_model import RecSysMetaModel
from recsys_trainer import RecSysTrainer
from recsys_metrics import compute_recsys_metrics
from recsys_args import DataArguments, ModelArguments
from recsys_data import (
    recsys_schema_small, 
    f_feature_extract, 
    vocab_sizes, 
    fetch_data_loaders
)


logger = logging.getLogger(__name__)


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

    # embedding size
    d_model = 512 
    max_seq_len = 2048

    _model = get_recsys_model(model_args, d_model, max_seq_len)
    model = RecSysMetaModel(_model, vocab_sizes, d_model=d_model)

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