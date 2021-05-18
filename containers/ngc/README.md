
# NGC image

This section has the instructions on how to build, run locally and push images to NGC.

## Build the image

Replace `prj-recsys` in the image name by your team on NGC.  
P.s. The GitHub repository must be cloned within the image. For that, you need to provide in the --build-arg the path of the private SSH key whose public SSH key was set as a Deploy Key in the private repo at GitHub.

```bash
cd Transformers4Rec/
docker build --no-cache --tag nvcr.io/nvidian/prj-recsys/transformers4rec:0.2-hf-4.6.0-nvtabular-0.5.1 --build-arg SSH_KEY="$(cat ~/.ssh/transf4rec_ngc_repo_key)" -f containers/ngc/Dockerfile.ngc .
```

### Baselines

```bash
cd Transformers4Rec/
docker build --no-cache --tag nvcr.io/nvidian/prj-recsys/transformers4rec:0.2-hf-4.6.0-nvtabular-0.5.1-theano1.0.5 --build-arg SSH_KEY="$(cat ~/.ssh/transf4rec_ngc_repo_key)" -f containers/ngc/Dockerfile.ngc_theano .
```


## Try locally

```bash
DATA_PATH=~/dataset/
docker run --gpus all -it --rm -p 6006:6006 -p 8888:8888 -v $DATA_PATH:/data --workdir /workspace/ nvcr.io/nvidian/prj-recsys/transformers4rec:0.2-hf-4.6.0-nvtabular-0.5.1 /bin/bash
```

Run inside the container

```bash
cd /workspace/Transformers4Rec/

#Pulling the main branch with the latest changes
git pull origin main


#Login into Weights&Biases
wandb login 
Access https://wandb.ai/authorize
Paste your W&B API key

#source activate merlin

DATA_PATH="/data/ecommerce_dataset/ecommerce_preproc_v5_with_repetitions_day_folders"

#Run training script
export CUDA_VISIBLE_DEVICES="0"
export SCRIPT_MODULE="transformers4rec.recsys_main"
export EXPERIMENT_GROUP_NAME="local_experiments"
bash scripts/run_algorithm_ngc.bash $CUDA_VISIBLE_DEVICES $SCRIPT_MODULE $EXPERIMENT_GROUP_NAME \
    --data_path "/data" \
    --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml \
    --data_loader_engine nvtabular \
    --start_time_window_index 1 \
    --final_time_window_index 5 \
    --time_window_folder_pad_digits 4 \
    --model_type transformerxl \
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
    --n_head 2 \
    --fp16  #if GPU with fp16 support is available
```

## Push image to NGC
If you have not, you need to login Docker into NGC container registry (nvcr.io) using your [NGC API key](https://ngc.nvidia.com/setup/api-key). 

```bash
docker login -u \$oauthtoken -p <NGCAPI> nvcr.io 
```

Then you will be able to push your image to NGC
```bash
docker push nvcr.io/nvidian/prj-recsys/transformers4rec:0.2-hf-4.6.0-nvtabular-0.5.1
```

## Run a Job on NGC

```bash
export CUDA_VISIBLE_DEVICES="0"
export SCRIPT_MODULE="transformers4rec.recsys_main"
export EXPERIMENT_GROUP_NAME="<EXPERIMENT GROUP NAME>"
export WANDB_API_KEY="<W&B API KEY here>"
ngc batch run --name "ml-model.gpt2 feat.itemid ds.ecom_rees46" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.1.norm --commandline "bash -c 'nvidia-smi && source activate rapids && wandb login $WANDB_API_KEY && date && git pull origin main && date && bash scripts/run_algorithm_ngc.bash $CUDA_VISIBLE_DEVICES $SCRIPT_MODULE $EXPERIMENT_GROUP_NAME --data_path /data/ --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type gpt2 --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 384 --learning_rate 0.0008781937894379981 --dropout 0.2 --input_dropout 0.4 --weight_decay 1.4901138106122045e-05 --d_model 128 --item_embedding_dim 448 --n_layer 1 --n_head 1 --label_smoothing 0.9 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.03 --other_embeddings_init_std 0.034999999999999996 && date'" --result /results --image "nvidian/prj-recsys/transformers4rec:0.2-hf-4.6.0-nvtabular-0.5" --org nvidian --team prj-recsys --datasetid 74861:/data