# NVIDIA Transformers4RecSys - Docker setup

This document provides example command lines to build, run locally and push Docker containers to NGC.


# Development image

## NVIDIA Container Toolkit Installation

To build and locally run the Docker images described in this directory,
it will be very likely that you will need to install the NVIDIA Container Toolkit stack.

This NVIDIA Container Toolkit stack allows your host OS to run Docker containers, and to have
your host OS's NVIDIA GPU capabilities accessible to the these Docker
containers.

The architecture of the NVIDIA Container Toolkit may best be visualized by this
diagram:

![here](https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png).

The NVIDIA Container Toolkit Installation instructions have been last verified to work 
in mid January 2021. 

### Host OS

You will need a development machine with a host OS capable of running the 
Docker container engine and NVIDIA Container Toolkit. A popular choice for this team 
that has been verified to work is the Ubuntu 18.04 OS distribution. 

The installation of Ubuntu 18.04 follows a typical setup involving, for
example, a USB Flash drive containing an Ubuntu 18.04 together with the appropriate boot
loaders.

This setup for Ubuntu 18.04 is detailed
[here](https://help.ubuntu.com/community/BurningIsoHowto),
while the actual Ubuntu 18.04 images may be downloaded
[here](http://releases.ubuntu.com/18.04/).


### Setup NVIDIA CUDA Drivers

You will need NVIDIA's CUDA software to full access and support the
GPU's that are hopefully connected to the machine running your host's OS.

We recommend having your OS distribution's package manager manage the
CUDA package for ease of package dependency and compatibility management 
between the CUDA packages and other OS packages, as well as ease of future 
CUDA package upgrades.

Full installation instructions may be found
[here](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

For Ubuntu, the package that you will eventually install is `nvidia-cuda`.


### Setup Docker

The Docker framework will allow you to run Docker containers on your host OS.

Full instructions for Ubuntu may be followed
[here](https://docs.docker.com/engine/install/ubuntu/).

Helpful post-installation steps for Linux installation, like allowing non-sudo 
invocations of Docker commands, may be followed
[here](https://docs.docker.com/engine/install/linux-postinstall/).

Note that `docker-ce` is the package we will install, while the `docker.io`
and `docker` (if they exist) packages are depreciated. 

### Setup NVIDIA Container Toolkit

The NVIDIA Container Toolkit will finally allow the running containers to access most,
if not all, of the full capabilities of your host machine's GPU.

The packages in the NVIDIA Container Toolkit can best visualized by the 
architectural diagram found at the top of this page:
[here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/arch-overview.html).

Note that we will eventually install the `nvidia-docker2` package, over the 
now-depreciated `nvidia-docker` package.

Full instructions to install NVIDIA Container Toolkit may be followed 
[here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

The GitHub project homepage for NVIDIA Container Toolkit is located [here](https://github.com/NVIDIA/nvidia-docker).

### Potential Errors
   
If you see this error, or similar

    `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]',

then it is likely that Docker is installed on your host OS, but that the NVIDIA
CUDA package and NV Docker package have not yet been correctly installed. 
Try re-doing the steps in the Setup NVIDIA CUDA and Setup NVIDIA Container Toolkit sections.

With the installation of the NVIDIA Container Toolkit stack, you will now be able to locally
build and locally run the containers documented in this README below. 

## Build

```bash
cd Transformers4Rec/
docker build --no-cache -t transformers4rec_dev:0.2-hf-4.6.0-nvtabular-0.5.1 -f containers/Dockerfile.dev: .
```

## Run the container in interactive mode

```bash
PROJECT_PATH=~/projects/nvidia/Transformers4Rec
DATA_PATH=~/dataset/
docker run --gpus all -it --rm -p 6006:6006 -p 8888:8888 -v $PROJECT_PATH:/workspace -v $DATA_PATH:/data --workdir /workspace/ transformers4rec_dev:0.2-hf-4.6.0-nvtabular-0.5.1 /bin/bash 
```


#### Run inside the container

```bash
cd /workspace/

#source activate merlin

#Login into Weights&Biases
wandb login 
Access https://wandb.ai/authorize
Paste your W&B API key

#Or disable WANDB login
export WANDB_MODE=dryrun


DATA_PATH="/data/ecommerce_dataset/ecommerce_preproc_v5_with_repetitions_day_folders"

#Run training script
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python3 -m transformers4rec.recsys_main \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path $DATA_PATH \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
    --experiments_group "local_experiments" \
    --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml \
    --data_loader_engine nvtabular \
    --start_time_window_index 1 \
    --final_time_window_index 5 \
    --time_window_folder_pad_digits 4 \
    --model_type transfoxl \
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
PROJECT_PATH=~/projects/nvidia/Transformers4Rec
DATA_PATH=~/dataset/

#source activate merlin

docker run --gpus all --ipc=host -it --rm --cap-add=SYS_ADMIN --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  \
 -p 6006:6006 -p 8888:8888 -v $PROJECT_PATH:/workspace -v $DATA_PATH/:/data --workdir /workspace/ transformers4rec_dev:0.2-hf-4.6.0-nvtabular-0.5.1 /bin/bash 
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

#source activate merlin

DATA_PATH=/data

dlprof --mode=pytorch \
       --force=true \
       --output_path=nsights_files/output_path \
       --tb_dir=tensorboard_event_files_path \
       --nsys_base_name=nsys_profile_path \
       --reports=all \
       --nsys_opts="--sample=cpu --trace 'nvtx,cuda,osrt,cudnn'" \
       --iter_start=1 --iter_stop=10 \
python3 -m transformers4rec.recsys_main \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path $DATA_PATH \
    --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml \
    --data_loader_engine nvtabular \
    --fp16 \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
    --start_time_window_index 1 \
    --final_time_window_index 2 \
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
    --n_head 2 \
    --pyprof
```

After running this profiling, it will be generated a folder in the path specified in `dlprof --output_path`. In that folder, there will be a file with the `.qdrep` extension, which you can open with the NVIDIA Nsights Systems app to check the usage of CPU and GPU processing and memory during the training steps. 

Additionally, it will be created a subfolder with the name defined in the `--tb_dir` argument, where it will be generated files that result in some reports for Tensorboard, which you can see by running `tensorboard --bind_all --logdir tensorboard_event_files_path`. Those reports highlight the slowest ops and also some hints for improving the performance of the model (e.g. using AMP and async data loaders).


These instructions are based in the article [Profiling and Optimizing Deep Neural Networks with DLProf and PyProf](https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof), where you can find more information.




```