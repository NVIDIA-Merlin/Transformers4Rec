# Tutorial: End-to-end Session-based Recommendation

Session-based recommendation, a sub-area of sequential recommendation, has been an important task in online services like e-commerce and news portals. Session-based recommenders provide relevant and  personalized recommendations even when prior user history is not available or their tastes change over time. They recently gained popularity due to their ability to capture short-term or contextual user preferences towards items.

## Learning Objectives

The example notebooks cover the following concepts and tasks:

- Preprocessing with cuDF and [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular).
- Feature engineering with NVTabular.
- Introduction to Transformers4Rec.
  - Introduction to session-based recommendation.
  - Accelerated dataloaders for PyTorch.
  - Training and evaluating an RNN-based session based recommendation model for next item prediction task.
  - Training and evaluating Transformer architecture based session-based recommendation model next item prediction task.
  - Using side information (additional features) to improve the accuracy of a model.
- Deploying to inference with [Triton Inference Server](https://github.com/triton-inference-server/server).


## Getting Started

In this tutorial, we use a subset of the publicly-available eCommerce dataset.
The e-commerce behavior data contains 7 months of data (from October 2019 to April 2020) from a large multi-category online store.
Each row in the file represents an event.
All events are related to products and users.
Each event is a many-to-many relation between products and users.
The data were collected by the Open CDP project and the source of the dataset is REES46 Marketing Platform.

We use only the `2019-Oct.csv` file for training our models, so you can visit this site and download the csv file: <https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store>.

Refer to the following notebooks:

- [Preliminary Preprocessing](01-preprocess.ipynb)
- [ETL with NVTabular](02-ETL-with-NVTabular.ipynb)
- [Session-based Recommendation](03-Session-based-recsys.ipynb)
- [Triton for Recommender Systems](04-Inference-with-Triton.ipynb)