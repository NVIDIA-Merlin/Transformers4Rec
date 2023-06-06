#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2022 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================

# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_transformers4rec_getting-started-session-based-01-etl-with-nvtabular/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # ETL with NVTabular

# In this notebook we are going to generate synthetic data and then create sequential features with [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular). Such data will be used in the next notebook to train a session-based recommendation model.
# 
# NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.

# ### Import required libraries

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob

import cudf
import numpy as np
import pandas as pd

import nvtabular as nvt
from nvtabular.ops import *
from merlin.schema.tags import Tags


# ### Define Input/Output Path

# In[3]:


INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")


# ## Create a Synthetic Input Data

# In[4]:


NUM_ROWS = os.environ.get("NUM_ROWS", 100000)


# In[5]:


long_tailed_item_distribution = np.clip(np.random.lognormal(3., 1., int(NUM_ROWS)).astype(np.int32), 1, 50000)
# generate random item interaction features 
df = pd.DataFrame(np.random.randint(70000, 90000, int(NUM_ROWS)), columns=['session_id'])
df['item_id'] = long_tailed_item_distribution

# generate category mapping for each item-id
df['category'] = pd.cut(df['item_id'], bins=334, labels=np.arange(1, 335)).astype(np.int32)
df['age_days'] = np.random.uniform(0, 1, int(NUM_ROWS)).astype(np.float32)
df['weekday_sin']= np.random.uniform(0, 1, int(NUM_ROWS)).astype(np.float32)

# generate day mapping for each session 
map_day = dict(zip(df.session_id.unique(), np.random.randint(1, 10, size=(df.session_id.nunique()))))
df['day'] =  df.session_id.map(map_day)


# Visualize couple of rows of the synthetic dataset:

# In[6]:


df.head()


# ## Feature Engineering with NVTabular

# Deep Learning models require dense input features. Categorical features are sparse, and need to be represented by dense embeddings in the model. To allow for that, categorical features first need to be encoded as contiguous integers `(0, ..., |C|)`, where `|C|` is the feature cardinality (number of unique values), so that their embeddings can be efficiently stored in embedding layers.  We will use NVTabular to preprocess the categorical features, so that all categorical columns are encoded as contiguous integers. Note that the `Categorify` op encodes OOVs or nulls to `0` automatically. In our synthetic dataset we do not have any nulls. On the other hand `0` is also used for padding the sequences in input block, therefore, you can set `start_index=1` arg in the Categorify op if you want the encoded null or OOV values to start from `1` instead of `0` because we reserve `0` for padding the sequence features.

# Here our goal is to create sequential features. To do so, we are grouping the features together at the session level in the following cell. In this synthetically generated example dataset, we do not have a timestamp column, but if we had one (that's the case for most real-world datasets), we would be sorting the interactions by the timestamp column as in this [example notebook](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/examples/end-to-end-session-based/01-ETL-with-NVTabular.ipynb). Note that we also trim each feature sequence in a  session to a certain length. Here, we use the NVTabular library so that we can easily preprocess and create features on GPU with a few lines.

# In[7]:


SESSIONS_MAX_LENGTH =20

# Categorify categorical features
categ_feats = ['item_id', 'category'] >> nvt.ops.Categorify()

# Define Groupby Workflow
groupby_feats = categ_feats + ['session_id', 'day', 'age_days', 'weekday_sin']

# Group interaction features by session
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["session_id"], 
    aggs={
        "item_id": ["list", "count"],
        "category": ["list"],     
        "day": ["first"],
        "age_days": ["list"],
        'weekday_sin': ["list"],
        },
    name_sep="-")

# Select and truncate the sequential features
sequence_features_truncated = (
    groupby_features['category-list']
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
)

sequence_features_truncated_item = (
    groupby_features['item_id-list']
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
    >> TagAsItemID()
)  
sequence_features_truncated_cont = (
    groupby_features['age_days-list', 'weekday_sin-list'] 
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
    >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
)

# Filter out sessions with length 1 (not valid for next-item prediction training and evaluation)
MINIMUM_SESSION_LENGTH = 2
selected_features = (
    groupby_features['item_id-count', 'day-first', 'session_id'] + 
    sequence_features_truncated_item +
    sequence_features_truncated + 
    sequence_features_truncated_cont
)
    
filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["item_id-count"] >= MINIMUM_SESSION_LENGTH)

seq_feats_list = filtered_sessions['item_id-list', 'category-list', 'age_days-list', 'weekday_sin-list'] >>  nvt.ops.ValueCount()

workflow = nvt.Workflow(filtered_sessions['session_id', 'day-first'] + seq_feats_list)

dataset = nvt.Dataset(df)

# Generate statistics for the features and export parquet files
# this step will generate the schema file
workflow.fit_transform(dataset).to_parquet(os.path.join(INPUT_DATA_DIR, "processed_nvt"))


# It is possible to save the preprocessing workflow. That is useful to apply the same preprocessing to other data (with the same schema) and also to deploy the session-based recommendation pipeline to Triton Inference Server.

# In[8]:


workflow.output_schema


# Save NVTabular workflow.

# In[9]:


workflow.save(os.path.join(INPUT_DATA_DIR, "workflow_etl"))


# ## Export pre-processed data by day

# In this example we are going to split the preprocessed parquet files by days, to allow for temporal training and evaluation. There will be a folder for each day and three parquet files within each day folder: `train.parquet`, `validation.parquet` and `test.parquet`.

# In[10]:


OUTPUT_DIR = os.environ.get("OUTPUT_DIR",os.path.join(INPUT_DATA_DIR, "sessions_by_day"))


# In[11]:


# Read in the processed parquet file
sessions_gdf = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet"))


# In[12]:


print(sessions_gdf.head(3))


# In[13]:


from transformers4rec.utils.data_utils import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= OUTPUT_DIR,
                       partition_col='day-first',
                       timestamp_col='session_id', 
                      )


# ## Check out the preprocessed outputs

# In[14]:


TRAIN_PATHS = os.path.join(OUTPUT_DIR, "1", "train.parquet")


# In[15]:


df = pd.read_parquet(TRAIN_PATHS)
df.head()


# In[16]:


import gc
del df
gc.collect()


# You have  just created session-level features to train a session-based recommendation model using NVTabular. Now you can move to the the next notebook,`02-session-based-XLNet-with-PyT.ipynb` to train a session-based recommendation model using [XLNet](https://arxiv.org/abs/1906.08237), one of the state-of-the-art NLP model. Please shut down this kernel to free the GPU memory before you start the next one.
