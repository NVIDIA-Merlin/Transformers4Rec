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
cd hf4rec/
docker build --no-cache -t hf4rec_dev -f container/Dockerfile.dev_nvt .
```

## Run the container in interactive mode

```bash
docker run --gpus all -it --rm -p 6006:6006 -p 8888:8888 -v ~/projects/nvidia/hf4rec:/workspace -v ~/dataset/:/data --workdir /workspace/ hf4rec_dev /bin/bash 
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
 -p 6006:6006 -p 8888:8888 -v ~/projects/nvidia/recsys:/workspace -v ~/dataset/:/data --workdir /workspace/ hf4rec_dev /bin/bash 
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
python3 -m hf4rec.recsys_main \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/DATA_PATH/" \
    --feature_config datasets/ecommerce_rees46/config/features/session_based_features_pid.yaml \
    --data_loader_engine nvtabular \
    --fp16 \
    --workers_count 2 \
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





# NGC image

This section has the instructions on how to build, run locally and push images to NGC.

## Build the image

Replace `prj-recsys` in the image name by your team on NGC.  
P.s. The GitHub repository must be cloned within the image. For that, you need to provide in the --build-arg the path of the private SSH key whose public SSH key was set as a Deploy Key in the private repo at GitHub.

```bash
cd hf4rec/
docker build --no-cache --tag nvcr.io/nvidian/prj-recsys/hf4rec:0.1-hf4.1.1-nvtabular0.3 --build-arg SSH_KEY="$(cat ~/.ssh/hf4rec_repo_key)" -f containers/Dockerfile.ngc_nvt .
```

## Try locally

```bash
docker run --gpus all -it -p 6006:6006 -p 8888:8888 -v ~/projects/nvidia/hf4rec:/workspace -v ~/dataset/:/data --workdir /workspace/ -t nvcr.io/nvidian/prj-recsys/hf4rec:0.1-hf4.1.1-nvtabular0.3 /bin/bash
```

Run inside the container

```bash
#cd /workspace/

#Pulling the main branch with the latest changes
git pull origin main


#Login into Weights&Biases
wandb login <W&B API KEY here>

#Run training script
export CUDA_VISIBLE_DEVICES="0"
export SCRIPT_MODULE="hf4rec.recsys_main"
export EXPERIMENT_GROUP_NAME="local_experiments"
bash scripts/run_algorithm_ngc.bash $CUDA_VISIBLE_DEVICES $SCRIPT_MODULE $EXPERIMENT_GROUP_NAME \
    --data_path "/data" \
    --feature_config datasets/ecommerce_rees46/config/features/session_based_features_pid.yaml \
    --data_loader_engine nvtabular \
    --workers_count 2 \
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
    --n_head 2 \
    --fp16  #if GPU with fp16 support is available
```

## Push image to NGC
If you have not, you need to login Docker into NGC container registry (nvcr.io) using your [NGC API key(https://ngc.nvidia.com/setup/api-key)]. 

```bash
docker login -u \$oauthtoken -p <NGCAPI> nvcr.io 
```

Then you will be able to push your image to NGC
```bash
docker push nvcr.io/nvidian/prj-recsys/hf4rec:0.1-hf4.1.1-nvtabular0.3
```

## Run a Job on NGC

```bash
export CUDA_VISIBLE_DEVICES="0"
export SCRIPT_MODULE="hf4rec.recsys_main"
export EXPERIMENT_GROUP_NAME="<ngc_experiments>"
export WANDB_API_KEY="<W&B API KEY here>"
ngc batch run --name "ml-model.hg4rec-gpt2-$(date +%Y%m%d%H%M%S)" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.1.norm --commandline "bash -c 'nvidia-smi && source activate rapids && wandb login $WANDB_API_KEY && date && git pull origin main && date && scripts/run_algorithm_ngc.bash $CUDA_VISIBLE_DEVICES $SCRIPT_MODULE $EXPERIMENT_GROUP_NAME --data_path /data/with_repetitions/ --feature_config ../datasets/ecommerce-large/config/features/session_based_features_pid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 15 --time_window_folder_pad_digits 4 --model_type transfoxl --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_num_cosine_cycles 1.25 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --num_train_epochs 10 --per_device_train_batch_size 512 --learning_rate 0.00020505005005475648 --learning_rate_schedule linear_with_warmup --dropout 0.1 --weight_decay 3.0393589859266068e-05 --d_model 384 --n_layer 3 --n_head 8 && date'" --result /results --image "nvidian/prj-recsys/hf4rec:0.1-hf4.1.1-nvtabular0.3" --org nvidian --team prj-recsys --datasetid 71255:/data
```