# Transformers4Rec Example Notebooks

We have created a collection of example Jupyter notebooks using different datasets to demonstrate how to use Transformers4Rec with PyTorch and TensorFlow APIs. Each example notebook provides incremental information about Transformers4Rec features and modules.

## Available Examples

### 1. [Getting started - Session-based recommendation with Synthetic Data](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/getting-started-session-based/)

This example notebook is focusing primarily on the basic concepts of Transformers4Rec, which includes:
- Generating synthetic data of user interactions
- Preprocessing sequential data with NVTabular on GPU
- Using the NVTabular dataloader with Pytorch
- Training a session-based recommendation model with a Transformer architecture (XLNET)

### 2. [End-to-end session-based recommendation](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/end-to-end-session-based/)

This end-to-end example notebook is focuses on:
- Preprocessing the Yoochoose e-commerce dataset
- Generating session features with on GPU
- Using the NVTabular dataloader with the Pytorch
- Training a session-based recommendation model with a Transformer architecture (XLNET)
- Exporting the preprocessing workflow and trained model to Triton Inference Server (TIS)
- Sending request to TIS and generating next-item predictions for each session


### 3. [Tutorial - End-to-End Session-Based Recommendation on GPU](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial)

This tutorial was presented at the ACM RecSys 2021. It covers the following topics:

- The main concepts for session-based recommendation
- Implementation of preprocessing and feature engineering techniques for session-based recommendation model with NVTabular on GPU
- How to build, train and evaluate a session-based recommendation models based on RNN and Transformer architectures with Transformers4Rec library
- How to deploy a session-based recommendation pipeline (preprocessing workflow and trained model) to the Triton Inference Server

### 4. [Transformers4Rec paper experiments reproducibility](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/t4rec_paper_experiments)

This example contains scripts to reproduce the experiments reported in the Transformers4Rec [paper](https://dl.acm.org/doi/10.1145/3460231.3474255) at RecSys 2021. The experiments focused in the session-based recommendation using incremental training and evaluation over time on two e-commerce and two news datasets.


## Running the Example Notebooks

You can run the example notebooks by [installing Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) and its dependencies using `pip` or `conda`. Alternatively, Docker containers are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. Depending on which example you want to run, you should use any one of these Docker containers:
- Merlin-Tensorflow-Training (contains NVTabular with TensorFlow)
- Merlin-Pytorch-Training (contains NVTabular with PyTorch)
- Merlin-Inference (contains NVTabular with PyTorch and Triton Inference support)

To run the example notebooks using Docker containers, do the following:

1. Pull the container by running the following command:
   ```
   docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host <docker container> /bin/bash
   ```

   **NOTES**: 
   
   - If you are running on Docker version 19 and higher, change ```--runtime=nvidia``` to ```--gpus all```.
   - The most recent PyTorch and TensorFlow docker containers are nvcr.io/nvidia/merlin/merlin-tensorflow-training:21.09 and nvcr.io/nvidia/merlin/merlin-pytorch-training:21.09
   - If you are running examples that require input data (examples 2 or 3) you need to add `-v <path_to_models>:/workspace/models/ -v <path to data>:/workspace/data/ ` to the docker script above. Here `<path_to_models>` is a local directory in your system, and the same directory should also be mounted to the `merlin-inference`container if you would like to run the inference example. Please follow the `launch and start triton server` instructions given in the notebooks. 

   The container will open a shell when the run command execution is completed. You will have to start JupyterLab on the Docker container. It should look similar to this:
   ```
   root@2efa5b50b909:
   ```

2. Go to the Transformers4Rec examples folder: `cd /transformers4rec/examples`

3. Install jupyter-lab with `pip` by running the following command:
   ```
   pip install jupyterlab
   ```
   
   For more information, see [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). 
   
4. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --port 8888 --NotebookApp.token='<password>'
   ```

5. Open a browser from the host OS to access the jupyter-lab server using `http://<MachineIP>:8888`.

6. Once in the server, navigate and try out the examples in `/transformers4rec/examples`.
