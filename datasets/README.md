# Datasets

This folder contains information and resources for different user interaction dataset used in our experiments with Transformer architectures for sequential and session-based recommendation

## REES46 eCommerce dataset

This is a large dataset comprising 7 months (from October 2019 to April 2020) from a large multi-category online store. It contains more than 411 million interactions, 89 million sessions, 15 million users and 386 thousand items. You can find more stats on this dataset in this [EDA notebook](ecommerce_rees46/preprocessing/pyspark/eda/ecom_dataset_eda_temporal_user_behaviour.ipynb).  
The raw dataset and more info can be found on [Kaggle Datasets](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store)

### Pre-processing

As this is a large dataset, its pre-processing was implemented mostly using PySpark. In general, the user interactions are split into sessions, and sessions are saved in parquet files. The parquet files are split by day, to allow incremental training and evaluation.  

The notebooks for pre-processing can be found in the [ecommerce_rees46/preprocessing/pyspark/](ecommerce_rees46/preprocessing/pyspark/) folder and must be executed in the order of the notebooks prefixes (01, 02, and 03).

### Preprocessed dataset

An already pre-processed version of this dataset comprising 1 month of data (October 2019) is available in this [Google Drive](https://drive.google.com/drive/u/0/folders/1LK24lJYn2mLUM2710iS5L6Pq9gGyj2_s).  
With this pre-processed dataset, you are able to run pipelines with Transformers4Rec.


## G1 news dataset

This is a dataset with users interactions logs (page views) from the [G1](https://g1.globo.com/), the most popular news portal in Brazil, which was provided by Globo.com.

The dataset contains a sample of user interactions (page views) in the news portal from Oct. 1 to 16, 2017, including about 3 million clicks, distributed in more than 1 million sessions from 314,000 users who read more than 46,000 different news articles during that period.

The raw datasets and more info can be found on [Kaggle Datasets](https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom).

### Pre-processing

AS this dataset is not very large, the preprocessing for this dataset was implemented using Pandas. In general, the user interactions are split into sessions, and sessions are saved in parquet files. The parquet files are split by hour, to allow incremental training and evaluation.  

The preprocessing notebook can be found in the [news_g1/preprocessing/G1_news_preprocess.ipynb](news_g1/preprocessing/G1_news_preprocess.ipynb) folder.



### Preprocessed dataset

An already pre-processed version of this dataset is available in this [Google Drive](https://drive.google.com/drive/u/0/folders/1qSUWRqBflR8EvMKoyLIkSyB3JlwNPNTT).  
With this pre-processed dataset, you are able to run pipelines with Transformers4Rec.


## Features config files
The Transformers4Rec uses a features config file (YAML) to get to know which features are available for the model. The only required feature is the one that contains the sequence of item ids. But more features can be provided, like item metadata /content features and user contextual features, generally improving models accuracy.
You can find examples of features config files for the [REES46 eCommerce dataset](ecommerce_rees46/config/features/) and [G1 news](news_g1/config/features/) datasets.