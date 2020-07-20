mkdir -p ./tmp/

# for multiple GPU, use 0,1 not fully supported yet

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1 python recsys_main.py \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/root/dataset/ecommerce_preproc_neg_samples_50_strategy_recent_popularity-2019-10/" \
    --start_date "2019-10-01" \
    --end_date "2019-10-19" \
    --engine "pyarrow" \
    --reader_pool_type "process" \
    --workers_count 10 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 128 \
    --model_type "xlnet" \
    --loss_type "margin_hinge" \
    --margin_loss 0.0 \
    --logging_steps 20 \
    --d_model 64 \
    --n_layer 4 \
    --n_head 2 \
    --dropout 0.2 \
    --learning_rate 1e-04 \
    --validate_every 10 \
    --num_train_epochs 400

# model_type: xlnet, gpt2, longformer
# loss_type: cross_entropy, margin_hinge
# fast-test: quickly finish loop over examples to check code after the loop
# d_model: size of hidden states (or internal states) for RNNs and Transformers
# n_layer: number of layers for RNNs and Transformers
# n_head: number of attention heads for Transformers
# hidden_act: non-linear activation function (function or string) in Transformers. 'gelu', 'relu' and 'swish' are supported
# dropout: dropout probability for all fully connected layers
# engine: select which parquet data loader to use either 'pyarrow' or 'petastorm' (default: pyarrow)
# reader_pool_type: petastorm reader arg: process or thread (default: thread)
# workers_count: petastorm reader arg: number of workers (default: 10)
# logging_steps: how often do logging (every n examples)