#!/bin/bash


source ~/miniconda3/etc/profile.d/conda.sh
conda activate transf4rec

cd hf4rec/
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/DATA_PATH/" \
    --feature_config datasets/ecommerce_rees46/config/features/session_based_features_pid.yaml \
    --data_loader_engine nvtabular \
    --workers_count 2 \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
    --start_time_window_index 1 \
    --final_time_window_index 15 \
    --time_window_folder_pad_digits 4 \
    --model_type gpt2 \
    --loss_type cross_entropy \
    --similarity_type concat_mlp \
    --tf_out_activation tanh \
    --inp_merge mlp \
    --hidden_act gelu_new \
    --dataloader_drop_last \
    --compute_metrics_each_n_steps 1 \
    --session_seq_length_max 20 \
    --eval_on_last_item_seq_only \
    --num_train_epochs 10 \
    --per_device_train_batch_size 192 \
    --per_device_eval_batch_size 128 \
    --learning_rate 0.00014969647714359603 \
    --learning_rate_schedule linear_with_warmup \
    --learning_rate_warmup_steps 0 \
    --mf_constrained_embeddings \
    --dropout 0.1 \
    --weight_decay 6.211639773976265e-05 \
    --d_model 320 \
    --n_layer 1 \
    --n_head 2