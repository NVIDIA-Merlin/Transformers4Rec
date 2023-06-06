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
# ==============================================================================

# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_transformers4rec_end-to-end-session-based-01-etl-with-nvtabular/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # ETL with NVTabular
# 
# This notebook is created using the latest stable [merlin-pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch) container.
# 
# **Launch the docker container**
# ```
# docker run -it --gpus device=0 -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 -v <path_to_data>:/workspace/data/  nvcr.io/nvidia/merlin/merlin-pytorch:22.XX
# ```
# This script will mount your local data folder that includes your data files to `/workspace/data` directory in the merlin-pytorch docker container.

# ## Overview

# This notebook demonstrates how to use NVTabular to perform the feature engineering that is needed to model the `YOOCHOOSE` dataset which contains a collection of sessions from a retailer. Each session  encapsulates the click events that the user performed in that session.
# 
# The dataset is available on [Kaggle](https://www.kaggle.com/chadgostopp/recsys-challenge-2015). You need to download it and copy to the `DATA_FOLDER` path. Note that we are only using the `yoochoose-clicks.dat` file.
# 
# First, let's start by importing several libraries:

# In[2]:


import os
import glob
import numpy as np
import gc

import cudf
import cupy
import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags


# Avoid Numba low occupancy warnings:

# In[3]:


from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


# #### Define Data Input and Output Paths

# In[4]:


DATA_FOLDER = "/workspace/data/"
FILENAME_PATTERN = 'yoochoose-clicks.dat'
DATA_PATH = os.path.join(DATA_FOLDER, FILENAME_PATTERN)

OUTPUT_FOLDER = "./yoochoose_transformed"
OVERWRITE = False


# ## Load and clean raw data

# In[5]:


interactions_df = cudf.read_csv(DATA_PATH, sep=',', 
                                names=['session_id','timestamp', 'item_id', 'category'], 
                                dtype=['int', 'datetime64[s]', 'int', 'int'])


# #### Remove repeated interactions within the same session

# In[6]:


print("Count with in-session repeated interactions: {}".format(len(interactions_df)))
# Sorts the dataframe by session and timestamp, to remove consecutive repetitions
interactions_df.timestamp = interactions_df.timestamp.astype(int)
interactions_df = interactions_df.sort_values(['session_id', 'timestamp'])
past_ids = interactions_df['item_id'].shift(1).fillna()
session_past_ids = interactions_df['session_id'].shift(1).fillna()
# Keeping only no consecutive repeated in session interactions
interactions_df = interactions_df[~((interactions_df['session_id'] == session_past_ids) & (interactions_df['item_id'] == past_ids))]
print("Count after removed in-session repeated interactions: {}".format(len(interactions_df)))


# #### Create new feature with the timestamp when the item was first seen

# In[7]:


items_first_ts_df = interactions_df.groupby('item_id').agg({'timestamp': 'min'}).reset_index().rename(columns={'timestamp': 'itemid_ts_first'})
interactions_merged_df = interactions_df.merge(items_first_ts_df, on=['item_id'], how='left')
print(interactions_merged_df.head())


# Let's save the interactions_merged_df to disk to be able to use in the inference step.

# In[8]:


interactions_merged_df.to_parquet(os.path.join(DATA_FOLDER, 'interactions_merged_df.parquet'))


# In[9]:


# print the total number of unique items in the dataset
print(interactions_merged_df.item_id.nunique())


# In[10]:


# free gpu memory
del interactions_df, session_past_ids, items_first_ts_df
gc.collect()


# ##  Define a preprocessing workflow with NVTabular

# NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.
# 
# NVTabular supports different feature engineering transformations required by deep learning (DL) models such as Categorical encoding and numerical feature normalization. It also supports feature engineering and generating sequential features. 
# 
# More information about the supported features can be found <a href=https://nvidia-merlin.github.io/NVTabular/main/index.html> here. </a>

# ### Feature engineering: Create and Transform items features

# In this cell, we are defining three transformations ops: 
# 
# - 1. Encoding categorical variables using `Categorify()` op. We set `start_index` to 1 so that encoded null values start from `1` instead of `0` because we reserve `0` for padding the sequence features.
# - 2. Deriving temporal features from timestamp and computing their cyclical representation using a custom lambda function. 
# - 3. Computing the item recency in days using a custom op. Note that item recency is defined as the difference between the first occurrence of the item in dataset and the actual date of item interaction. 
# 
# For more ETL workflow examples, visit NVTabular [example notebooks](https://github.com/NVIDIA-Merlin/NVTabular/tree/main/examples).

# In[11]:


# Encodes categorical features as contiguous integers
cat_feats = ColumnSelector(['category', 'item_id']) >> nvt.ops.Categorify(start_index=1)

# create time features
session_ts = ColumnSelector(['timestamp'])
session_time = (
    session_ts >> 
    nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='s')) >> 
    nvt.ops.Rename(name = 'event_time_dt')
)
sessiontime_weekday = (
    session_time >> 
    nvt.ops.LambdaOp(lambda col: col.dt.weekday) >> 
    nvt.ops.Rename(name ='et_dayofweek')
)

# Derive cyclical features: Define a custom lambda function 
def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin

weekday_sin = sessiontime_weekday >> (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_sin')

# Compute Item recency: Define a custom Op 
class ItemRecency(nvt.ops.Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            item_first_timestamp = gdf['itemid_ts_first']
            delta_days = (col - item_first_timestamp) / (60*60*24)
            gdf[column + "_age_days"] = delta_days * (delta_days >=0)
        return gdf

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        self._validate_matching_cols(input_schema, parents_selector, "computing input selector")
        return parents_selector

    def column_mapping(self, col_selector):
        column_mapping = {}
        for col_name in col_selector.names:
            column_mapping[col_name + "_age_days"] = [col_name]
        return column_mapping

    @property
    def dependencies(self):
        return ["itemid_ts_first"]

    @property
    def output_dtype(self):
        return np.float64
    
recency_features = session_ts >> ItemRecency() 
# Apply standardization to this continuous feature
recency_features_norm = recency_features >> nvt.ops.LogOp() >> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(name='product_recency_days_log_norm')

time_features = (
    session_time +
    sessiontime_weekday +
    weekday_sin + 
    recency_features_norm
)

features = ColumnSelector(['session_id', 'timestamp']) + cat_feats + time_features 


# ### Define the preprocessing of sequential features

# Once the item features are generated, the objective of this cell is to group interactions at the session level, sorting the interactions by time. We additionally truncate all sessions to first 20 interactions and filter out sessions with less than 2 interactions.

# In[12]:


# Define Groupby Operator
groupby_features = features >> nvt.ops.Groupby(
    groupby_cols=["session_id"], 
    sort_cols=["timestamp"],
    aggs={
        'item_id': ["list", "count"],
        'category': ["list"],  
        'timestamp': ["first"],
        'event_time_dt': ["first"],
        'et_dayofweek_sin': ["list"],
        'product_recency_days_log_norm': ["list"]
        },
    name_sep="-")

# Truncate sequence features to first interacted 20 items 
SESSIONS_MAX_LENGTH = 20 


item_feat = groupby_features['item_id-list'] >> nvt.ops.TagAsItemID()
cont_feats = groupby_features['et_dayofweek_sin-list', 'product_recency_days_log_norm-list'] >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])


groupby_features_list =  item_feat + cont_feats + groupby_features['category-list']
groupby_features_truncated = groupby_features_list >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH)

# Calculate session day index based on 'event_time_dt-first' column
day_index = ((groupby_features['event_time_dt-first'])  >> 
             nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
             nvt.ops.Rename(f = lambda col: "day_index") >>
             nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
            )

# tag session_id column for serving with legacy api
sess_id = groupby_features['session_id'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])

# Select features for training 
selected_features = sess_id + groupby_features['item_id-count'] + groupby_features_truncated + day_index

# Filter out sessions with less than 2 interactions 
MINIMUM_SESSION_LENGTH = 2
filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["item_id-count"] >= MINIMUM_SESSION_LENGTH) 


# ### Execute NVTabular workflow

# Once we have defined the general workflow (`filtered_sessions`), we provide our cudf dataset to `nvt.Dataset` class which is optimized to split data into chunks that can fit in device memory and to handle the calculation of complex global statistics. Then, we execute the pipeline that fits and transforms data to get the desired output features.

# In[13]:


dataset = nvt.Dataset(interactions_merged_df)
workflow = nvt.Workflow(filtered_sessions)
# Learn features statistics necessary of the preprocessing workflow
# The following will generate schema.pbtxt file in the provided folder and export the parquet files.
workflow.fit_transform(dataset).to_parquet(os.path.join(DATA_FOLDER, "processed_nvt"))


# Let's print the head of our preprocessed dataset. You can notice that now each example (row) is a session and the sequential features with respect to user interactions were converted to lists with matching length.

# In[14]:


workflow.output_schema


# #### Save the preprocessing workflow

# In[15]:


workflow.save(os.path.join(DATA_FOLDER, "workflow_etl"))


# ### Export pre-processed data by day

# In this example we are going to split the preprocessed parquet files by days, to allow for temporal training and evaluation. There will be a folder for each day and three parquet files within each day: `train.parquet`, `validation.parquet` and `test.parquet`.
#   
# P.s. It is worthwhile to note that the dataset has a single categorical feature (category), which, however, is inconsistent over time in the dataset. All interactions before day 84 (2014-06-23) have the same value for that feature, whereas many other categories are introduced afterwards. Thus for this example, we save only the last five days.

# In[16]:


# read in the processed train dataset
sessions_gdf = cudf.read_parquet(os.path.join(DATA_FOLDER, "processed_nvt/part_0.parquet"))
sessions_gdf = sessions_gdf[sessions_gdf.day_index>=178]


# In[17]:


print(sessions_gdf.head(3))


# In[18]:


from transformers4rec.utils.data_utils import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir=os.path.join(DATA_FOLDER, "preproc_sessions_by_day"),
                       partition_col='day_index',
                       timestamp_col='session_id', 
                      )


# In[19]:


# free gpu memory
del  sessions_gdf
gc.collect()


# That's it! We created our sequential features, now we can go to the next notebook to train a PyTorch session-based model.
