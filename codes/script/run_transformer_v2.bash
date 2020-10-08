mkdir -p ./tmp/

experiments_group=$1 # Name of the experiment group for this run. Used only to organize jobs in W&B
feature_type=$2 # full, pidcid , single
sampling_type=$3 # recent_popularity, uniform, session_cooccurrence

# engine: select which parquet data loader to use either 'pyarrow' or 'petastorm' (default: pyarrow)
# reader_pool_type: petastorm reader arg: process or thread (default: thread)
# workers_count: petastorm reader arg: number of workers (default: 10)
# logging_steps: how often do logging (every n examples)

echo "Running training script"
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1 python recsys_main.py \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/data/" \
    --feature_config config/recsys_input_feature_${feature_type}.yaml \
    --engine "pyarrow" \
    --reader_pool_type "process" \
    --workers_count 10 \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
    --experiments_group ${experiments_group} \
    ${@:4} # Forwarding Remaining parameters to the script

#--data_path "/root/dataset/ecommerce_preproc_neg_samples_50_strategy_${sampling_type}-2019-10/" \

train_exit_status=$?    
    
echo "Copying outputs to NGC results dir"
cp -r ./tmp/pred_logs/ /results
cp -r ./tmp/attention_weights/ /results
cp ./tmp/*.txt /results
cp ./tmp/*.json /results
cp -r ./runs/ /results

#Ensuring that the job exit status is the same of the trainings scripts
exit ${train_exit_status}

# Additional recsys_main.py command line parameters
#--start_date #Example: "2019-10-01"
#--end_date #Example: "2019-10-15"
#--per_device_train_batch_size #Example: 256
#--per_device_eval_batch_size #Example: 128
#--model_type #Example: gpt2, gru, avgseq
#--loss_type #Example: cross_entropy, margin_hinge, cross_entropy_neg
#--margin_loss #Example: 0.0
#--d_model #Example: 64 - size of hidden states (or internal states) for RNNs and Transformers
#--n_layer #Example: 4
#--n_head #Example: 2 - number of attention heads for Transformers
#--hidden_act: non-linear activation function (function or string) in Transformers. 'gelu', 'relu' and 'swish' are supported
#--dropout #Example: 0.2 - dropout probability for all fully connected layers
#--learning_rate #Example:
#--validate_every #Example:
#--similarity_type #Example: concat_mlp, cosine
#--num_train_epochs #Example: 15
#--all_rescale_factor #Example: 1.0
#--neg_rescale_factor #Example: 0.0
#--inp_merge #Example:  mlp or attn
#--tf_out_activation #Example: relu or tanh
#--fast-test: quickly finish loop over examples to check code after the loop