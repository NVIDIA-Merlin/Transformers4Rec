mkdir -p ./tmp/

# for multiple GPU, use 0,1 not fully supported yet

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python recsys_main.py \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/root/dataset/ecommerce_preproc_neg_samples_50_strategy_cooccurrence_19_days_first_10k_sessions.parquet/" \
    --start_date 2019-10-01 \
    --end_date 2019-10-19 \
    --per_device_train_batch_size 128 \
    --model_type "xlnet" \
    --loss_type "margin_hinge" \
    --d_model 256 \
    --n_layer 6 \
    --n_head 4 \
    --dropout 0.2 

# model_type: xlnet, gpt2, longformer
# loss_type: cross_entropy, hinge_loss
# fast-test: quickly finish loop over examples to check code after the loop
# d_model: size of hidden states (or internal states) for RNNs and Transformers
# n_layer: number of layers for RNNs and Transformers
# n_head: number of attention heads for Transformers
# hidden_act: non-linear activation function (function or string) in Transformers. 'gelu', 'relu' and 'swish' are supported
# dropout: dropout probability for all fully connected layers
