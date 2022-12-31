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
# =======


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_transformers4rec_getting-started-session-based-01-etl-with-nvtabular/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # ETL with NVTabular

# In this notebook we are going to generate synthetic data and then create sequential features with [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular). Such data will be used in the next notebook to train a session-based recommendation model.
# 
# NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.

# ### Import required libraries

# In[2]:


import os
import glob

import numpy as np
import pandas as pd

import cudf
import cupy as cp
import nvtabular as nvt
from nvtabular.ops import *
from merlin.schema.tags import Tags


# ### Define Input/Output Path

# In[3]:


INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")


# ## Create a Synthetic Input Data

# In[4]:


NUM_ROWS = 100000
long_tailed_item_distribution = np.clip(np.random.lognormal(3., 1., NUM_ROWS).astype(np.int32), 1, 50000)

# generate random item interaction features 
df = pd.DataFrame(np.random.randint(70000, 90000, NUM_ROWS), columns=['session_id'])
df['item_id'] = long_tailed_item_distribution

# generate category mapping for each item-id
df['category'] = pd.cut(df['item_id'], bins=334, labels=np.arange(1, 335)).astype(np.int32)
df['age_days'] = np.random.uniform(0, 1, NUM_ROWS).astype(np.float32)
df['weekday_sin']= np.random.uniform(0, 1, NUM_ROWS).astype(np.float32)

# generate day mapping for each session 
map_day = dict(zip(df.session_id.unique(), np.random.randint(1, 10, size=(df.session_id.nunique()))))
df['day'] =  df.session_id.map(map_day)


# Visualize couple of rows of the synthetic dataset:

# In[5]:


df.head()


# ## Feature Engineering with NVTabular

# Deep Learning models require dense input features. Categorical features are sparse, and need to be represented by dense embeddings in the model. To allow for that, categorical features first need to be encoded as contiguous integers `(0, ..., |C|)`, where `|C|` is the feature cardinality (number of unique values), so that their embeddings can be efficiently stored in embedding layers.  We will use NVTabular to preprocess the categorical features, so that all categorical columns are encoded as contiguous integers. Note that the `Categorify` op encodes OOVs or nulls to `0` automatically. In our synthetic dataset we do not have any nulls. On the other hand `0` is also used for padding the sequences in input block, therefore, you can set `start_index=1` arg in the Categorify op if you want the encoded null or OOV values to start from `1` instead of `0` because we reserve `0` for padding the sequence features.

# Here our goal is to create sequential features.  In this cell, we are creating temporal features and grouping them together at the session level, sorting the interactions by time. Note that we also trim each feature sequence in a  session to a certain length. Here, we use the NVTabular library so that we can easily preprocess and create features on GPU with a few lines.

# In[6]:


SESSIONS_MAX_LENGTH =20

# Categorify categorical features
categ_feats = ['session_id', 'item_id', 'category'] >> nvt.ops.Categorify()

# Define Groupby Workflow
groupby_feats = categ_feats + ['day', 'age_days', 'weekday_sin']

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
    >> nvt.ops.ValueCount()
)

sequence_features_truncated_item = (
    groupby_features['item_id-list']
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
    >> TagAsItemID()
    >> nvt.ops.ValueCount()
)  
sequence_features_truncated_cont = (
    groupby_features['age_days-list', 'weekday_sin-list'] 
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
    >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
    >> nvt.ops.ValueCount()
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


workflow = nvt.Workflow(filtered_sessions['session_id', 'day-first', 'item_id-count'] + seq_feats_list)

dataset = nvt.Dataset(df, cpu=False)
# Generate statistics for the features
workflow.fit(dataset)
# Apply the preprocessing and return an NVTabular dataset
sessions_ds = workflow.transform(dataset)
# Convert the NVTabular dataset to a Dask cuDF dataframe (`to_ddf()`) and then to cuDF dataframe (`.compute()`)
sessions_gdf = sessions_ds.to_ddf().compute()


# In[7]:


sessions_gdf.head(3)


# In[8]:


sessions_gdf.dtypes


# It is possible to save the preprocessing workflow. That is useful to apply the same preprocessing to other data (with the same schema) and also to deploy the session-based recommendation pipeline to Triton Inference Server.

# In[9]:


workflow.output_schema


# The following will generate `schema.pbtxt` file in the provided folder.

# In[10]:


workflow.fit_transform(dataset).to_parquet(os.path.join(INPUT_DATA_DIR, "processed_nvt"))


# In[11]:


workflow.save('workflow_etl')


# ## Export pre-processed data by day

# In this example we are going to split the preprocessed parquet files by days, to allow for temporal training and evaluation. There will be a folder for each day and three parquet files within each day folder: `train.parquet`, `validation.parquet` and `test.parquet`.

# In[12]:


OUTPUT_DIR = os.environ.get("OUTPUT_DIR",os.path.join(INPUT_DATA_DIR, "sessions_by_day"))
get_ipython().system('mkdir -p $OUTPUT_DIR')


# In[13]:


from transformers4rec.data.preprocessing import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= OUTPUT_DIR,
                       partition_col='day-first',
                       timestamp_col='session_id', 
                      )


# ## Checking the preprocessed outputs

# In[14]:


TRAIN_PATHS = sorted(glob.glob(os.path.join(OUTPUT_DIR, "1", "train.parquet")))


# In[15]:


gdf = cudf.read_parquet(TRAIN_PATHS[0])
gdf


# You have  just created session-level features to train a session-based recommendation model using NVTabular. Now you can move to the the next notebook,`02-session-based-XLNet-with-PyT.ipynb` to train a session-based recommendation model using [XLNet](https://arxiv.org/abs/1906.08237), one of the state-of-the-art NLP model. Please shut down this kernel to free the GPU memory before you start the next one.
