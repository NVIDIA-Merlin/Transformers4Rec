# Session-based Recommendation on GPU with Transformers4Rec

Session-based recommendation, a sub-area of sequential recommendation, has been an important task in online services like e-commerce and news portals. Session-based recommenders provide relevant and  personalized recommendations even when prior user history is not available or their tastes change over time. They recently gained popularity due to their ability to capture short-term or contextual user preferences towards items.


**Learning Objectives**

In this tutorial section, we created a series of notebooks that will help you to learn:

- the main concepts and algorithms for session-based recommendation
- implementation of preprocessing and feature engineering techniques for session-based recommendation model on GPU with [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular)
- how to build, train and evaluate a session-based recommendation model based on RNN and Transformer architectures with Transformers4Rec library
- how to deploy a trained model to the [Triton Inference Server](https://github.com/triton-inference-server/server)


**Each Jupyter notebook covers the following:<br>**

- Preprocessing with cuDF and NVTabular
- Feature engineering with NVTabular
- Introduction to Transformers4Rec
    - Introduction to session-based recommendation
    - Accelerated dataloaders for PyTorch
    - Traning and evaluating an RNN-based session based recommendation model for next item prediction task
    - Traning and evaluating Transformer architecture based session-based recommendation model next item prediction task
    - Using side information (additional features) to improve the accuracy of a model
- Deploying to inference with Triton


## Getting Started

### Download data

In this tutorial, we are going to use a subset of the publicly available eCommerce dataset. The e-commerce behavior data contains 7 months data (from October 2019 to April 2020) from a large multi-category online store. Each row in the file represents an event. All events are related to products and users. Each event is like many-to-many relation between products and users. Data collected by Open CDP project and the source of the dataset is REES46 Marketing Platform.

We use only 2019-Oct.csv file for training our models, so you can visit this site and download the csv file: https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store.


### Launch Docker Container for Training

You need to pull https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training container to be able to run the ETL and training notebooks. Please note that to use this Docker container, you'll first need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker.

Follow the steps in this [README.md](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/examples/README.md) for Docker launch and instructions for both training and inference containers.