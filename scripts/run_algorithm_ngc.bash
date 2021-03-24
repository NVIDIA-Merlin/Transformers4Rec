#!/bin/bash

mkdir -p ./tmp/

cuda_visible_devices=$1 #GPU devices available for the program, separated by commas. Example: 0,1
python_module=$2 #Can be: codes.recsys_main OR codes.baselines.recsys_baselines_main
experiments_group=$3 # Name of the experiment group for this run. Used only to organize jobs in W&B

echo "Running training script"
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m ${python_module} \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/data/" \
    --data_loader_engine nvtabular \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
    --experiments_group ${experiments_group} \
    ${@:4} # Forwarding Remaining parameters to the script

train_exit_status=$?    
    
echo "Copying outputs to NGC results dir"
cp -r ./tmp/pred_logs/ /results
cp -r ./tmp/attention_weights/ /results
cp ./tmp/*.txt /results
cp ./tmp/*.json /results
cp ./tmp/*.csv /results
cp -r ./runs/ /results

#Ensuring that the job exit status is the same of the trainings scripts
exit ${train_exit_status}