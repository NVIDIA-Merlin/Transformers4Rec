# Transformers4Rec Example Notebooks

We have created a collection of Jupyter notebooks based on different datasets. These example notebooks demonstrate how to use Transformers4REc with PyTorch and TensorFlow. Each example provides additional information about Transformers4Rec modules.

## Structure

The example notebooks are structured as follows and should be reviewed in this order:
- 01-ETL-with-NVTabular.ipynb: Demonstrates how to execute the preprocessing and feature engineering pipeline (ETL) with NVTabular on the GPU.
- 02-Training-with-PyTorch.ipynb: Demonstrates how to train a session-based recommndation model with PyTorch based on the ETL output.
- 02-Training-with-TF.ipynb: Demonstrates how to train a session-based recommndation model with TensorFlow based on the ETL output.

## Available Example Notebooks

### 1. [Getting Started with Synthetic Data](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/pytorch)

This example notebook is focusing primarily on the basic concepts of Transformers4Rec, which includes:
- Generating session-based features with NVTabular 
- Creating sequential inputs for model
- Using the NVTabular dataloader with the Pytorch
- Using the NVTabular dataloader with Tensorflow
- Training an XLNET based session-based recommendation model

### 2. [End-to-end session-based with Yoochoose dataset](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/end-to-end-session-based/)

### 3. [Tutorial with Rees46 Ecommerce dataset](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/tutorial/)

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
   - If you are running `Tutorial notebooks`  you need to add `-v ${PWD}:/models/ -v <path to data>:/data/ ` to the docker script above. Here `PWD` is a local directory in your system, and the same directory should also be mounted to the `merlin-inference`container if you would like to run the inference example. Please follow the `start and launch triton server` instructions given in the inference notebook. 

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

5. Once in the server, navigate to the ```/nvtabular/``` directory and try out the examples.
