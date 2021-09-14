# Transformers4Rec Example Notebooks

We have created a collection of Jupyter notebooks based on different datasets. These example notebooks demonstrate how to use Transformers4REc with PyTorch and TensorFlow. Each example provides additional information about Transformers4Rec modules.

## Available Example Notebooks

### 1. [Getting started session-based with Synthetic Data](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/getting-started-session-based/)

This example notebook is focusing primarily on the basic concepts of Transformers4Rec, which includes:
- Generating session-based features with NVTabular on GPU
- Creating sequential inputs for model
- Using the NVTabular dataloader with the Pytorch
- Training an XLNET based session-based recommendation model

### 2. [End-to-end session-based with Yoochoose dataset](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/end-to-end-session-based/)

This end-to-end example notebook is focuses on:
- Preprocessing Yoochoose ecommeerce dataset
- Generating session-based features with NVTabular on GPU
- Creating sequential inputs for model
- Using the NVTabular dataloader with the Pytorch
- Training an XLNET based session-based recommendation model
- Serving the saved ensemble mode to Triton Inference Server (TIS)
- Sending request to TIS, retrieving responses and converting to the prediction results to the recommended next item-id for each session


### 3. [Tutorial with Rees46 Ecommerce dataset](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/tutorial/)

The tutorial notebooks are designed to teach the users

- the main concepts for session-based recommendation
- implementation of preprocessing and feature engineering techniques for session-based recommendation model with NVTAbular on GPU
- how to build, train and evaluate a session-based recommendation model based on RNN and Transformer architectures with Transformers4Rec library
- how to deploy a trained model to the Triton Inference Server


## Running the Example Notebooks

You can run the example notebooks by [installing Transformers4Rec]() and other required libraries. Alternatively, Docker containers are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. Depending on which example you want to run, you should use any one of these Docker containers:
- Merlin-Tensorflow-Training (contains NVTabular with TensorFlow)
- Merlin-Pytorch-Training (contains NVTabular with PyTorch)
- Merlin-Training (contains NVTabular with HugeCTR)
- Merlin-Inference (contains NVTabular with TensorFlow and Triton Inference support)

To run the example notebooks using Docker containers, do the following:

1. Pull the container by running the following command:
   ```
   docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host <docker container> /bin/bash
   ```

   **NOTES**: 
   
   - If you are running on Docker version 19 and higher, change ```--runtime=nvidia``` to ```--gpus all```.
   - If you are running `Tutorial or end-to-end-session-based notebooks`  you need to add `-v <path_to_models>:/workspace/models/ -v <path to data>:/workspace/data/ ` to the docker script above. Here `<path_to_models>` is a local directory in your system, and the same directory should also be mounted to the `merlin-inference`container if you would like to run the inference example. Please follow the `launch and start triton server` instructions given in the notebooks. 

   The container will open a shell when the run command execution is completed. You will have to start JupyterLab on the Docker container. It should look similar to this:
   ```
   root@2efa5b50b909:
   ```

2. Install jupyter-lab with `pip` by running the following command:
   ```
   pip install jupyterlab
   ```
   
   For more information, see [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). 
   
3. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```

4. Open any browser to access the jupyter-lab server using <MachineIP>:8888.

5. Once in the server, navigate to the `/Transformers4Rec/` directory and try out the examples.
