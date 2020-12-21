#!/bin/bash


source ~/miniconda3/etc/profile.d/conda.sh
conda activate transf4rec

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1 python3  recsys_main.py \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/home/gmoreira/dataset/ecommerce/2019-10" \
    --feature_config config/recsys_input_feature_full_noneg.yaml \
    --engine "pyarrow" \
    --reader_pool_type "process" \
    --workers_count 10 \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 1000 \
--start_date 2019-10-01 \
--end_date 2019-10-15 \
--model_type gpt2 \
--loss_type cross_entropy \
--per_device_eval_batch_size 128 \
--similarity_type concat_mlp \
--tf_out_activation tanh \
--all_rescale_factor 1.0 \
--neg_rescale_factor 0.0 \
--inp_merge mlp \
--learning_rate_warmup_steps 0 \
--learning_rate_num_cosine_cycles 1.25 \
--hidden_act gelu_new \
--dataloader_drop_last \
--compute_metrics_each_n_steps 50 \
--max_seq_len 20 \
--eval_on_last_item_seq_only \
--warmup_days 1 \
--num_train_epochs 10 \
--per_device_train_batch_size 192 \
--learning_rate 0.00014969647714359603 \
--learning_rate_warmup 0.00014529247619095396 \
--learning_rate_schedule linear_with_warmup \
--learning_rate_schedule_warmup constant_with_warmup \
--dropout 0.1 \
--weight_decay 6.211639773976265e-05 \
--d_model 320 \
--n_layer 1 \
--n_head 2