# NGC image

## Build the  image

```
cd recsys/transformers4recsys/
docker build --tag transf4rec_exp:0.1.0 --tag nvcr.io/nvidian/prj-recsys/transf4rec_exp:0.1.0 --no-cache container/


```

## Try locally

```
docker run --gpus all -it --rm transf4rec_exp:0.1.0
```

Run inside the container
```
cd /workspace/recsys/transformers4recsys/codes

git pull origin experimentation

#Download the eCommerce preprocessed dataset
bash script/get_dataset.bash

#Login into Weights&Biases
wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12

#Run training script
bash script/run_transformer_v2.bash full session_cooccurrence \
  --start_date "2019-10-01" \
  --end_date "2019-10-15" \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 128 \
  --model_type gpt2 \
  --loss_type cross_entropy_neg \
  --margin_loss 0.0 \
  --d_model 64 \
  --n_layer 4 \
  --n_head 2 \
  --hidden_act gelu \
  --dropout 0.2 \
  --learning_rate 1e-03  \
  --similarity_type concat_mlp \
  --num_train_epochs 15 \
  --all_rescale_factor 1.0 \
  --neg_rescale_factor 0.0 \
  --inp_merge mlp \
  --tf_out_activation tanh
```

## Push image to NGC
```
docker push nvcr.io/nvidian/prj-recsys/transf4rec_exp:0.1.0
```

## Run a Job on NGC
```
ngc batch run --name "tranf4rec-job-$(date +%Y%m%d%H%M%S)" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --port 8888 \
--commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/get_dataset.bash && date && bash script/run_transformer_v2.bash full session_cooccurrence \
  --start_date 2019-10-01 \
  --end_date 2019-10-15 \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 128 \
  --model_type gpt2 \
  --loss_type cross_entropy_neg \
  --margin_loss 0.0 \
  --d_model 64 \
  --n_layer 4 \
  --n_head 2 \
  --hidden_act gelu \
  --dropout 0.2 \
  --learning_rate 1e-03  \
  --similarity_type concat_mlp \
  --num_train_epochs 15 \
  --all_rescale_factor 1.0 \
  --neg_rescale_factor 0.0 \
  --inp_merge mlp \
  --tf_out_activation tanh \
  && date"
```


# Development image

## Build the development image

docker build --no-cache -t transf4rec/dev -f container/Dockerfile.dev_nvt .

# Enable sampling in containers

Nsight Systems samples CPU activity and gets backtraces using the Linux kernel’s perf subsystem. To collect thread scheduling data and instruction pointer (IP) samples, the perf paranoid level on the target system must be ≤2. Run the following command to check the level:

cat /proc/sys/kernel/perf_event_paranoid
If the output is >2, then run the following command to temporarily adjust the paranoid level (after each reboot):

sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
To make the change permanent, run the following command:

sudo sh -c 'echo kernel.perf_event_paranoid=1 > /etc/sysctl.d/local.conf'


## Run the container

docker run --gpus all --ipc=host -it --rm --cap-add=SYS_ADMIN  \
 --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  \
 -p 6006:6006 -p 8888:8888 -v /home/gmoreira/projects/nvidia/recsys:/workspace -v /home/gmoreira/dataset/ecommerce/ecommerce_preproc_split/ecommerce_preproc_2019-10:/data --workdir /workspace/transformers4recsys/codes transf4rec/dev /bin/bash  


## Set environment variables

export WANDB_API_KEY=76eea90114bb1cdcbafe151b262e4a5d4ff60f12

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1


## Train the model


TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1 python3  recsys_main.py \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/home/gmoreira/dataset/ecommerce/2019-10" \
    --feature_config config/recsys_input_feature_full_noneg.yaml \
    --engine "pyarrow" \
    --reader_pool_type "process" \
    --workers_count 8 \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
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


## Profile the model with DLProf

Based in this article: https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof/

export WANDB_API_KEY=76eea90114bb1cdcbafe151b262e4a5d4ff60f12

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0


dlprof --mode=pytorch \
       --force=true \
       --output_path=nsights_files/fp16_pyarrow_loader_nativeamp_v5 \
       --tb_dir=tensorboard_event_files \
       --nsys_base_name=nsys_profile_fp16_pyarrow_loader_nativeamp_v5 \
       --reports=all \
       --nsys_opts="--sample=cpu --trace 'nvtx,cuda,osrt,cudnn'" \
       --iter_start=1 --iter_stop=10 \
python recsys_main.py \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/data" \
    --feature_config config/recsys_input_feature_full_noneg.yaml \
    --reader_pool_type "process" \
    --workers_count 8 \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
--start_date 2019-10-01 \
--end_date 2019-10-02 \
--model_type gpt2 \
--loss_type cross_entropy \
--per_device_eval_batch_size 640 \
--similarity_type concat_mlp \
--tf_out_activation tanh \
--all_rescale_factor 1.0 \
--neg_rescale_factor 0.0 \
--inp_merge mlp \
--learning_rate_warmup_steps 0 \
--learning_rate_num_cosine_cycles 1.25 \
--hidden_act gelu_new \
--compute_metrics_each_n_steps 50 \
--max_seq_len 20 \
--eval_on_last_item_seq_only \
--warmup_days 1 \
--num_train_epochs 1 \
--per_device_train_batch_size 640 \
--learning_rate 0.00014969647714359603 \
--learning_rate_warmup 0.00014529247619095396 \
--learning_rate_schedule linear_with_warmup \
--learning_rate_schedule_warmup constant_with_warmup \
--dropout 0.1 \
--weight_decay 6.211639773976265e-05 \
--d_model 320 \
--n_layer 1 \
--n_head 2 \
--data_loader_engine nvtabular \
--fp16 \
--pyprof







----------------------------------------------------------------------




Run the Rapids image
NVT Data Loader can read sequences now!!!
docker run --gpus all --ipc=host -it --rm --cap-add=SYS_ADMIN   --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864   -p 6006:6006 -p 8888:8888 -v /home/gmoreira/projects/nvidia/recsys:/workspace -v /home/gmoreira/dataset/ecommerce/2019-10:/data --workdir /workspace/transformers4recsys/codes nvcr.io/nvidia/rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04 /bin/bash 

apt update

mkdir /nvtabular0.3
cd /nvtabular0.3 && git clone https://github.com/NVIDIA/NVTabular.git && cd NVTabular && pip install -e . && pip install nvtx

pip install torch
pip install -r requirements.txt

#DLProf
pip install nvidia-pyindex
pip install nvidia-pyprof
pip install nvidia-dlprof

#Apex
mkdir /apex
cd /apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

apt install gcc
pip install apex


---------------------------------------------------------
Image with RAPIDS 0.16, NVT 0.3, PyTorch, DLProf

docker build -t transf4rec/nvt_dl --no-cache -f container/Dockerfile.dev_nvt .

docker run --gpus all --ipc=host -it --rm --cap-add=SYS_ADMIN   --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864   -p 6006:6006 -p 8888:8888 -v /home/gmoreira/projects/nvidia/recsys:/workspace -v /home/gmoreira/dataset/ecommerce/2019-10:/data --workdir /workspace/transformers4recsys/codes transf4rec/nvt_dl /bin/bash 


tensorboard --bind_all --logdir . 

----------------------------------------------------
Build NGC image with NVTabular dependencies

docker build -t nvcr.io/nvidian/prj-recsys/transf4rec_nvt_exp:0.1.0 -f Dockerfile.ngc_nvt .

docker run --gpus all -it -t nvcr.io/nvidian/prj-recsys/transf4rec_nvt_exp:0.1.0 bash

docker push nvcr.io/nvidian/prj-recsys/transf4rec_nvt_exp:0.1.0


## Alternative - Setup enviromnent with conda

conda install -c rapidsai -c nvidia -c numba -c conda-forge -c defaults cudf=0.16 nvtabular=0.3