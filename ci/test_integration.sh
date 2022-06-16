#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/bin/bash
set -e

# Get latest Transformers4Rec version
cd /transformers4rec/
git pull origin main

container=$1
devices=$2

# Run only for Merlin PyTorch Container
if [ "$container" != "merlin-pytorch" ]; then
   exit 0
fi

## Install requirements
cd examples/t4rec_paper_experiments
pip install -r requirements.txt

## Get data
cd t4r_paper_repro
FEATURE_SCHEMA_PATH=../datasets_configs/ecom_rees46/rees46_schema.pbtxt
pip install gdown
gdown https://drive.google.com/uc?id=1payLwuwfa_QG6GvFVg4KT1w7dSkO-jPZ
apt-get install unzip -y
DATA_PATH=rees46/
unzip -d $DATA_PATH "rees46_ecom_dataset_small_for_ci.zip"

## Run tests
export CUDA_VISIBLE_DEVICES="$devices"
### GPT-2 (CLM) - Item Id feature
python3 transf_exp_main.py --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_PATH --features_schema_path $FEATURE_SCHEMA_PATH --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 2 --time_window_folder_pad_digits 4 --model_type gpt2 --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 5 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 128 --learning_rate 0.0008781937894379981 --dropout 0.2 --input_dropout 0.4 --weight_decay 1.4901138106122045e-05 --d_model 128 --item_embedding_dim 448 --n_layer 1 --n_head 1 --label_smoothing 0.9 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.03 --other_embeddings_init_std 0.034999999999999996 --eval_on_test_set --seed 100 --report_to none

### Transformer-XL (CLM) - Item Id feature
python3 transf_exp_main.py --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_PATH --features_schema_path $FEATURE_SCHEMA_PATH --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 2 --time_window_folder_pad_digits 4 --model_type transfoxl --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 5 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 128 --learning_rate 0.001007765821083962 --dropout 0.1 --input_dropout 0.30000000000000004 --weight_decay 1.0673054163921092e-06 --d_model 448 --item_embedding_dim 320 --n_layer 1 --n_head 1 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.02 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.01 --eval_on_test_set --seed 100 --report_to none

### BERT (MLM) - Item Id feature
python3 transf_exp_main.py --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_PATH --features_schema_path $FEATURE_SCHEMA_PATH --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 2 --time_window_folder_pad_digits 4 --model_type albert --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 5 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --mlm --num_hidden_groups -1 --inner_group_num 1 --per_device_train_batch_size 128 --learning_rate 0.0004904752786458524 --dropout 0.0 --input_dropout 0.1 --weight_decay 9.565968888623912e-05 --d_model 320 --item_embedding_dim 320 --n_layer 2 --n_head 8 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.06 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.025 --mlm_probability 0.6000000000000001 --eval_on_test_set --seed 100 --report_to none

### XLNet (PLM) - Item Id feature
python3 transf_exp_main.py --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_PATH --features_schema_path $FEATURE_SCHEMA_PATH --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 2 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 5 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --plm --per_device_train_batch_size 128 --learning_rate 0.0003387925502203725 --dropout 0.0 --input_dropout 0.2 --weight_decay 2.1769664191492473e-05 --d_model 384 --item_embedding_dim 384 --n_layer 4 --n_head 16 --label_smoothing 0.7000000000000001 --stochastic_shared_embeddings_replacement_prob 0.02 --item_id_embeddings_init_std 0.13 --other_embeddings_init_std 0.005 --plm_probability 0.5 --plm_max_span_length 3 --eval_on_test_set --seed 100 --report_to none

### XLNet (MLM) - Item Id feature
python3 transf_exp_main.py --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_PATH --features_schema_path $FEATURE_SCHEMA_PATH --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 2 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 5 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --per_device_train_batch_size 128 --learning_rate 0.0006667377132554976 --dropout 0.0 --input_dropout 0.1 --weight_decay 3.910060265627374e-05 --d_model 192 --item_embedding_dim 448 --n_layer 3 --n_head 16 --label_smoothing 0.0 --stochastic_shared_embeddings_replacement_prob 0.1 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.02 --mlm_probability 0.30000000000000004 --eval_on_test_set --seed 100 --max_steps 20 --report_to none

### XLNET (MLM) - CONCAT + SOFT ONE-HOT ENCODING - All features
python3 transf_exp_main.py --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_PATH --features_schema_path $FEATURE_SCHEMA_PATH --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 2 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 5 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation concat --per_device_train_batch_size 128 --learning_rate 0.00034029107417129616 --dropout 0.0 --input_dropout 0.1 --weight_decay 3.168336235732841e-05 --d_model 448 --item_embedding_dim 384 --n_layer 2 --n_head 8 --label_smoothing 0.6000000000000001 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.06999999999999999 --other_embeddings_init_std 0.085 --mlm_probability 0.30000000000000004 --embedding_dim_from_cardinality_multiplier 1.0 --numeric_features_project_to_embedding_dim 20 --numeric_features_soft_one_hot_encoding_num_embeddings 5 --eval_on_test_set --seed 100 --use_side_information_features --report_to none
