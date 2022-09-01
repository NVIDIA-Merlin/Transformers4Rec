# Transformers4Rec Example Notebooks

We have collection of example Jupyter notebooks using different datasets to demonstrate how to use Transformers4Rec with the PyTorch API. Each example notebook provides incremental information about Transformers4Rec features and modules.

## Inventory

### [Getting started - Session-based recommendation with Synthetic Data](./getting-started-session-based/)

This example notebook is focusing primarily on the basic concepts of Transformers4Rec, which includes:

- Generating synthetic data of user interactions
- Preprocessing sequential data with NVTabular on GPU
- Using the NVTabular dataloader with Pytorch
- Training a session-based recommendation model with a Transformer architecture (XLNET)

### [End-to-end session-based recommendation](./end-to-end-session-based/)

This end-to-end example notebook is focuses on:

- Preprocessing the Yoochoose e-commerce dataset
- Generating session features with on GPU
- Using the NVTabular dataloader with PyTorch
- Training a session-based recommendation model with a Transformer architecture (XLNET)
- Exporting the preprocessing workflow and trained model to Triton Inference Server (TIS)
- Sending request to TIS and generating next-item predictions for each session

### [Tutorial - End-to-End Session-Based Recommendation on GPU](./tutorial)

This tutorial was presented at the ACM RecSys 2021. It covers the following topics:

- The main concepts for session-based recommendation
- Implementation of preprocessing and feature engineering techniques for session-based recommendation model with NVTabular on GPU
- How to build, train and evaluate a session-based recommendation models based on RNN and Transformer architectures with Transformers4Rec library
- How to deploy a session-based recommendation pipeline (preprocessing workflow and trained model) to the Triton Inference Server

### [Transformers4Rec paper experiments reproducibility](./t4rec_paper_experiments)

This example contains scripts to reproduce the experiments reported in the Transformers4Rec [paper](https://dl.acm.org/doi/10.1145/3460231.3474255) at RecSys 2021. The experiments focused in the session-based recommendation using incremental training and evaluation over time on two e-commerce and two news datasets.

## Running the Example Notebooks

You can run the examples with Docker containers.
Docker containers are available from the NVIDIA GPU Cloud.
Access the catalog of containers at <http://ngc.nvidia.com/catalog/containers>.

The `merlin-pytorch` container is suited to running the Transformers4Rec notebooks.
The container is available from the NGC catalog at the following URL:

<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch>

This container includes the Merlin Core, Merlin Models, Merlin Systems, NVTabular and PyTorch libraries.

To run the example notebooks using the container, perform the following steps:

1. If you haven't already created a Docker volume to share models and datasets
   between containers, create the volume by running the following command:

   ```shell
   docker volume create merlin-examples
   ```

1. Pull and start the container by running the following command:

   ```shell
   docker run --gpus all --rm -it \
     -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
     -v merlin-examples:/workspace/data \
     <docker container> /bin/bash
   ```

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   ```shell
   root@2efa5b50b909:
   ```

1. Install JupyterLab with `pip` by running the following command:

   ```shell
   pip install jupyterlab
   ```

   For more information, see the JupyterLab [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).


1. Start the JupyterLab server by running the following command:

   ```shell
   jupyter-lab --allow-root --ip='0.0.0.0'
   ```

   View the messages in your terminal to identify the URL for JupyterLab.
   The messages in your terminal show similar lines to the following example:

   ```shell
   Or copy and paste one of these URLs:
   http://2efa5b50b909:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   or http://127.0.0.1:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   ```

1. Open a browser and use the `127.0.0.1` URL provided in the messages by JupyterLab.

1. After you log in to JupyterLab, navigate to the `/transformers4rec/examples` directory to try out the example notebooks.