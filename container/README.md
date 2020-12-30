# NVIDIA Transformers4RecSys - Docker setup

This document provides example command lines to build, run locally and push Docker containers to NGC.


# Development image

## Build

```bash
cd recsys/transformers4recsys/
docker build --no-cache -t transf4rec_dev -f container/Dockerfile.dev_nvt .
```

## Run the container in interactive mode

```bash
docker run --gpus all -it --rm -p 6006:6006 -p 8888:8888 -v ~/projects/nvidia/recsys:/workspace -v ~/dataset/:/data --workdir /workspace/transformers4recsys/codes transf4rec_dev /bin/bash 
```

## Profile the model with NVIDIA DLProf and Nsight Systems
It is possible to use the [NVIDIA Deep Learning Profiler (DLProf)](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html) and [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) tools to profile the training / evaluation loop, identify bottlenecks and get some performance hints (e.g. getting to know if all TensorCores eligible ops are being used when using AMP (--fp16) or whether the data loaders are consuming a large time or letting the GPUs idle while reading).


### Setup in the host
The first step is to enable sampling in containers in the host machine.

Nsight Systems samples CPU activity and gets backtraces using the Linux kernel’s perf subsystem. To collect thread scheduling data and instruction pointer (IP) samples, the perf paranoid level on the target system must be ≤2. Run the following command to check the level:

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

If the output is >2, then run the following command to temporarily adjust the paranoid level (after each reboot):

```bash
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```

To make the change permanent, run the following command:

```bash
sudo sh -c 'echo kernel.perf_event_paranoid=1 > /etc/sysctl.d/local.conf'
```

### Run the container for profiling

```bash
docker run --gpus all --ipc=host -it --rm --cap-add=SYS_ADMIN --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  \
 -p 6006:6006 -p 8888:8888 -v ~/projects/nvidia/recsys:/workspace -v ~/dataset/:/data --workdir /workspace/transformers4recsys/codes transf4rec_dev /bin/bash 
```

### Profiling the model with DLProf

The [NVIDIA Deep Learning Profiler (DLProf)](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html) is a tool for profiling deep learning models to help data scientists understand and improve performance of their models visually via Tensorboard or by analyzing text reports. 

DLProf uses the [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) to profile the training session and create the necessary event files needed to view the results in TensorBoard.

Run the following command inside the container to generate profile training steps between `--iter_start` and `--iter_stop`.

```bash
#Disables W&B logging
export WANDB_MODE=dryrun
#Enables only one GPU (limitation of DLProf)
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

dlprof --mode=pytorch \
       --force=true \
       --output_path=nsights_files/output_path \
       --tb_dir=tensorboard_event_files_path \
       --nsys_base_name=nsys_profile_path \
       --reports=all \
       --nsys_opts="--sample=cpu --trace 'nvtx,cuda,osrt,cudnn'" \
       --iter_start=1 --iter_stop=10 \
python recsys_main.py --output_dir ./tmp/ --do_train --do_eval --data_path /data/ --start_date 2019-10-01 --end_date 2019-10-02 --data_loader_engine nvtabular --per_device_train_batch_size 320 --per_device_eval_batch_size 512 --model_type gpt2 --loss_type cross_entropy --logging_steps 10 --d_model 256 --n_layer 2 --n_head 8 --dropout 0.1 --learning_rate 0.001 --similarity_type concat_mlp --num_train_epochs 1 --all_rescale_factor 1 --neg_rescale_factor 0 --feature_config ../datasets/ecommerce-large/config/features/session_based_features_pid.yaml --inp_merge mlp --tf_out_activation tanh --experiments_group local_test --weight_decay 1.3e-05 --learning_rate_schedule constant_with_warmup --learning_rate_warmup_steps 0 --learning_rate_num_cosine_cycles 1.25 --dataloader_drop_last --compute_metrics_each_n_steps 1 --hidden_act gelu_new --save_steps 0 --eval_on_last_item_seq_only --fp16 --overwrite_output_dir --session_seq_length_max 20 --predict_top_k 1000 --eval_accumulation_steps 1 \
--max_steps 20 --pyprof
```

After running this profiling, will be generated a folder in the path specified in `dlprof --output_path`. In that folder, there will be a file with the `.qdrep` extension, which you can open with the NVIDIA Nsights Systems app to check the usage of CPU and GPU processing and memory during the training steps. 

Additionally, it will be created a subfolder with the name defined in the `--tb_dir` argument, where it will be generated files that result in some reports for Tensorboard, which you can see by running `tensorboard --bind_all --logdir tensorboard_event_files_path`. Those reports highlight the slowest ops and also some hints for improving the performance of the model (e.g. using AMP and async data loaders).


These instructions are based in the article [Profiling and Optimizing Deep Neural Networks with DLProf and PyProf](https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof), where you can find more information.





# NGC image

This section has the instructions on how to build, run locally and push images to NGC.

## Build the image

Please replace `prj-recsys` in the image name by your team on NGC.

```bash
cd recsys/transformers4recsys/
docker build --tag transf4rec_exp:0.1.0 --tag nvcr.io/nvidian/prj-recsys/transf4rec_test:0.1.0 --no-cache container/
```

## Try locally

```bash
docker run --gpus all -it -p 6006:6006 -p 8888:8888 -v ~/projects/nvidia/recsys:/workspace -v ~/dataset/:/data --workdir /workspace/transformers4recsys/codes -t nvcr.io/nvidian/prj-recsys/transf4rec_test:0.1.0 /bin/bash
```

Run inside the container

```bash
cd /workspace/recsys/transformers4recsys/codes

#Pulling the branch for NGC experimentation
git pull origin experimentation

#Download the eCommerce preprocessed dataset
bash script/dowload_dataset_from_gdrive.bash

#Login into Weights&Biases
wandb login <W&B API KEY here>

#Run training script
bash script/run_transformer_ngc.bash experiment_group_name --data_path ~/dataset/ --feature_config ../datasets/ecommerce-large/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_num_cosine_cycles 1.25 --hidden_act gelu_new --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --warmup_days 0 --num_train_epochs 2 --per_device_train_batch_size 512 --learning_rate 0.00019534113832496156 --learning_rate_schedule cosine_with_warmup --dropout 0.1 --weight_decay 8.81237861957528e-05 --d_model 448 --n_layer 4 --n_head 4
```

## Push image to NGC
If you have not, you need to login Docker into NGC container registry (nvcr.io) using your [NGC API key(https://ngc.nvidia.com/setup/api-key)]. 

```bash
docker login -u \$oauthtoken -p <NGCAPI> nvcr.io 
```

Then you will be able to push your image to NGC
```bash
docker push nvcr.io/nvidian/prj-recsys/transf4rec_test:0.1.0
```

## Run a Job on NGC

```bash
ngc batch run --name "tranf4rec-job-$(date +%Y%m%d%H%M%S)" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --result /results --image "nvidian/prj-recsys/transf4rec_test:0.1.0" --org nvidian --team prj-recsys --datasetid 71255:/data --commandline "bash -c 'nvidia-smi && source activate rapids && wandb login <W&B API KEY here> && date && git pull origin experimentation && date && bash script/run_transformer_ngc.bash experiment_group_name --data_path /data/with_repetitions/ --feature_config ../datasets/ecommerce-large/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_num_cosine_cycles 1.25 --hidden_act gelu_new --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --warmup_days 0 --num_train_epochs 2 --per_device_train_batch_size 512 --learning_rate 0.00019534113832496156 --learning_rate_schedule cosine_with_warmup --dropout 0.1 --weight_decay 8.81237861957528e-05 --d_model 448 --n_layer 4 --n_head 4 && date'" 
```
