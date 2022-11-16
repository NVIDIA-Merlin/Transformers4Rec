#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
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


# # ETL with NVTabular
# ## 1. Introduction

# In this notebook, we will create a preprocessing and feature engineering pipeline with [Rapids cuDF](https://github.com/rapidsai/cudf) and [Merlin NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) libraries to prepare our dataset for session-based recommendation model training. 
# 
# NVTabular is a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems. It provides high-level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS Dask-cuDF library, and is designed to be interoperable with both PyTorch and TensorFlow using dataloaders that have been developed as extensions of native framework code.
# 
# Our main goal is to create sequential features. In order to do that, we are going to perform the following:
# 
# - Categorify categorical features with `Categorify()` op
# - Create temporal features with a `user-defined custom` op and `Lambda` op
# - Transform continuous features using `Log` and `Normalize` ops
# - Group all these features together at the session level sorting the interactions by time with `Groupby`
# - Finally export the preprocessed datasets to parquet files by hive-partitioning.

# ### 1.1. Dataset

# In our hands-on exercise notebooks we are going to use a subset of the publicly available [eCommerce dataset](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store). The eCommerce behavior data contains 7 months of data (from October 2019 to April 2020) from a large multi-category online store. Each row in the file represents an event. All events are related to products and users. Each event is like many-to-many relation between products and users.
# 
# Data collected by Open CDP project and the source of the dataset is [REES46 Marketing Platform](https://rees46.com/).

# ## 2. Import Libraries

# In[2]:


import os

import numpy as np 
import cupy as cp
import glob

import cudf
import nvtabular as nvt
from nvtabular import ColumnSelector


# ## 3. Set up Input and Output Data Paths

# In[3]:


# define data path about where to get our data
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")


# ## 4. Read the Input Parquet file

# We already performed certain preprocessing steps on the first month (Oct-2019) of the raw dataset in the `01-preprocess` notebook: <br>
# 
# - we created `event_time_ts` column from `event_time` column which shows the time when event happened at (in UTC).
# - we created `prod_first_event_time_ts` column which indicates the timestamp that an item was seen first time.
# - we removed the rows where the `user_session` is Null. As a result, 2 rows were removed.
# - we categorified the `user_session` column, so that it now has only integer values.
# - we removed consequetively repeated (user, item) interactions. For example, an original session with `[1, 2, 4, 1, 2, 2, 3, 3, 3]` product interactions has become `[1, 2, 4, 1, 2, 3]` after removing the repeated interactions on the same item within the same session.

# Even though the original dataset contains 7 months data files, we are going to use the first seven days of the `Oct-2019.csv` ecommerce dataset. We use cuDF to read the parquet file. 

# In[4]:


get_ipython().run_cell_magic('time', '', "df = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, 'Oct-2019.parquet'))  \ndf.head(5)\n")


# In[5]:


df.shape


# Let's check if there is any column with nulls.

# In[6]:


df.isnull().any()


# We see that `'category_code'` and `'brand'` columns have null values, and in the following cell we are going to fill these nulls with via categorify op, and then all categorical columns will be encoded to continuous integers. Note that we add `start_index=1` in the `Categorify op` for the categorical columns, the reason for that we want the encoded null values to start from `1` instead of `0` because we reserve `0` for padding the sequence features.

# ## 5. Initialize NVTabular Workflow
# 
# ### 5.1. Categorical Features Encoding

# In[7]:


# categorify features 
cat_feats = ['user_session', 'category_code', 'brand', 'user_id', 'product_id', 'category_id', 'event_type'] >> nvt.ops.Categorify(start_index=1)


# ### 5.2. Extract Temporal Features

# In[8]:


# create time features
session_ts = ['event_time_ts']

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


# Now let's create cycling features from the `sessiontime_weekday` column. We would like to use the temporal features (hour, day of week, month, etc.) that have inherently cyclical characteristic. We represent the day of week as a cycling feature (sine and cosine), so that it can be represented in a continuous space. That way, the difference between the representation of two different days is the same, in other words, with cyclical features we can convey closeness between data. You can read more about it [here](https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/).

# In[9]:


def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin

def get_cycled_feature_value_cos(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_cos = np.cos(2*np.pi*value_scaled)
    return value_cos


# In[10]:


weekday_sin = sessiontime_weekday >> (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_sin')
weekday_cos= sessiontime_weekday >> (lambda col: get_cycled_feature_value_cos(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_cos')


# ### 5.2.1 Add Product Recency feature

# - Let's define a custom op to calculate product recency in days

# In[11]:


from nvtabular.ops import Operator

class ItemRecency(Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            item_first_timestamp = gdf['prod_first_event_time_ts']
            delta_days = (col - item_first_timestamp) / (60*60*24)
            gdf[column + "_age_days"] = delta_days * (delta_days >=0)
        return gdf
            
    def output_column_names(self, columns):
        return ColumnSelector([column + "_age_days" for column in columns.names])

    def dependencies(self):
        return ["prod_first_event_time_ts"]
    
    
recency_features = ['event_time_ts'] >> ItemRecency() 
recency_features_norm = recency_features >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='product_recency_days_log_norm')


# In[12]:


time_features = (
    session_time +
    sessiontime_weekday +
    weekday_sin +
    weekday_cos +
    recency_features_norm
)


# ### 5.3. Normalize Continuous Features¶

# In[13]:


# Smoothing price long-tailed distribution and applying standardization
price_log = ['price'] >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='price_log_norm')


# In[14]:


# Relative price to the average price for the category_id
def relative_price_to_avg_categ(col, gdf):
    epsilon = 1e-5
    col = ((gdf['price'] - col) / (col + epsilon)) * (col > 0).astype(int)
    return col
    
avg_category_id_pr = ['category_id'] >> nvt.ops.JoinGroupby(cont_cols =['price'], stats=["mean"]) >> nvt.ops.Rename(name='avg_category_id_price')
relative_price_to_avg_category = avg_category_id_pr >> nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=['price']) >> nvt.ops.Rename(name="relative_price_to_avg_categ_id")


# ### 5.4. Grouping interactions into sessions

# #### Aggregate by session id and creates the sequential features

# In[15]:


groupby_feats = ['event_time_ts', 'user_session'] + cat_feats + time_features + price_log + relative_price_to_avg_category


# In[16]:


# Define Groupby Workflow
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["user_session"], 
    sort_cols=["event_time_ts"],
    aggs={
        'user_id': ['first'],
        'product_id': ["list", "count"],
        'category_code': ["list"],  
        'event_type': ["list"], 
        'brand': ["list"], 
        'category_id': ["list"], 
        'event_time_ts': ["first"],
        'event_time_dt': ["first"],
        'et_dayofweek_sin': ["list"],
        'et_dayofweek_cos': ["list"],
        'price_log_norm': ["list"],
        'relative_price_to_avg_categ_id': ["list"],
        'product_recency_days_log_norm': ["list"]
        },
    name_sep="-")


# - Select columns which are list

# In[17]:


groupby_features_list = groupby_features['product_id-list',
        'category_code-list',  
        'event_type-list', 
        'brand-list', 
        'category_id-list', 
        'et_dayofweek_sin-list',
        'et_dayofweek_cos-list',
        'price_log_norm-list',
        'relative_price_to_avg_categ_id-list',
        'product_recency_days_log_norm-list']


# In[18]:


SESSIONS_MAX_LENGTH = 20 
MINIMUM_SESSION_LENGTH = 2


# We truncate the sequence features in length according to sessions_max_length param, which is set as 20 in our example.

# In[19]:


groupby_features_trim = groupby_features_list >> nvt.ops.ListSlice(0,SESSIONS_MAX_LENGTH) >> nvt.ops.Rename(postfix = '_seq')


# - Create a `day_index` column in order to partition sessions by day when saving the parquet files.

# In[20]:


# calculate session day index based on 'timestamp-first' column
day_index = ((groupby_features['event_time_dt-first'])  >> 
    nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
    nvt.ops.Rename(f = lambda col: "day_index")
)


# - Select certain columns to be used in model training

# In[21]:


selected_features = groupby_features['user_session', 'product_id-count'] + groupby_features_trim + day_index


# - Filter out the session that have less than 2 interactions.

# In[22]:


filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["product_id-count"] >= MINIMUM_SESSION_LENGTH)


# In[23]:


# avoid numba warnings
from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


# - Initialize the NVTabular dataset object and workflow graph.

# NVTabular's preprocessing and feature engineering workflows are directed graphs of operators. When we initialize a Workflow with our pipeline, workflow organizes the input and output columns.

# In[24]:


dataset = nvt.Dataset(df)

workflow = nvt.Workflow(filtered_sessions)
workflow.fit(dataset)
sessions_gdf = workflow.transform(dataset).to_ddf()


# Above, we created an NVTabular Dataset object using our input dataset. Then, we calculate statistics for this workflow on the input dataset, i.e. on our training set, using the `workflow.fit()` method so that our Workflow can use these stats to transform any given input.

# Let's print the head of our preprocessed dataset. You can notice that now each example (row) is a session and the sequential features with respect to user interactions were converted to lists with matching length.

# In[25]:


sessions_gdf.head(3)


# In[26]:


workflow.output_schema.column_names


# - Save NVTabular workflow to load at the inference step.

# In[27]:


workflow_path = os.path.join(INPUT_DATA_DIR, 'workflow_etl')
workflow.save(workflow_path)


# ## 6. Exporting data

# We export dataset to parquet partitioned by the session `day_index` column.

# In[28]:


# define partition column
PARTITION_COL = 'day_index'

# define output_folder to store the partitioned parquet files
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", INPUT_DATA_DIR + "sessions_by_day")
get_ipython().system('mkdir -p $OUTPUT_FOLDER')


# In this section we are going to create a folder structure as shown below. As we explained above, this is just to structure parquet files so that it would be easier to do incremental training and evaluation.

# ```
# /sessions_by_day/
# |-- 1
# |   |-- train.parquet
# |   |-- valid.parquet
# |   |-- test.parquet
# 
# |-- 2
# |   |-- train.parquet
# |   |-- valid.parquet
# |   |-- test.parquet
# ```

# `gpu_preprocessing` function converts the process df to a Dataset object and write out hive-partitioned data to disk.

# In[29]:


from transformers4rec.data.preprocessing import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= OUTPUT_FOLDER,
                       partition_col=PARTITION_COL,
                       timestamp_col='user_session', 
                      )


# In[30]:


# check out the OUTPUT_FOLDER
get_ipython().system('ls $OUTPUT_FOLDER')


# ## 7. Wrap Up 

# That's it! We finished our first task. We reprocessed our dataset and created new features to train a session-based recommendation model. Please run the cell below to shut down the kernel before moving on to the next notebook.

# In[31]:


import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

