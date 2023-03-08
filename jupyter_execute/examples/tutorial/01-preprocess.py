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


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_transformers4rec_tutorial-01-preprocess/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Preliminary Preprocessing
# 
# **Read and Process E-Commerce data**

# In this notebook, we are going to use a subset of a publicly available [eCommerce dataset](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store). The full dataset contains 7 months data (from October 2019 to April 2020) from a large multi-category online store. Each row in the file represents an event. All events are related to products and users. Each event is like many-to-many relation between products and users.
# Data collected by Open CDP project and the source of the dataset is [REES46 Marketing Platform](https://rees46.com/).

# We use only `2019-Oct.csv` file for training our models, so you can visit this site and download the csv file: https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store.

# ### Import the required libraries

# In[2]:


import os
import numpy as np 
import gc
import shutil
import glob

import cudf
import nvtabular as nvt


# ### Read Data via cuDF from CSV

# At this point we expect that you have already downloaded the `2019-Oct.csv` dataset and stored it in the `INPUT_DATA_DIR` as defined below. It is worth mentioning that the raw dataset is ~ 6 GB, therefore a single GPU with 16 GB or less memory might run out of memory.

# In[3]:


# define some information about where to get our data
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")


# In[4]:


get_ipython().run_cell_magic('time', '', "raw_df = cudf.read_csv(os.path.join(INPUT_DATA_DIR, '2019-Oct.csv')) \nraw_df.head()\n")


# In[5]:


raw_df.shape


# ### Convert timestamp from datetime

# In[6]:


raw_df['event_time_dt'] = raw_df['event_time'].astype('datetime64[s]')
raw_df['event_time_ts']= raw_df['event_time_dt'].astype('int')
raw_df.head()


# In[7]:


# check out the columns with nulls
raw_df.isnull().any()


# In[8]:


# Remove rows where `user_session` is null.
raw_df = raw_df[raw_df['user_session'].isnull()==False]
len(raw_df)


# We no longer need `event_time` column.

# In[9]:


raw_df = raw_df.drop(['event_time'],  axis=1)


# ### Categorify `user_session` column
# Although `user_session` is not used as an input feature for the model, it is useful to convert those raw long string to int values to avoid potential failures when grouping interactions by `user_session` in the next notebook.

# In[10]:


cols = list(raw_df.columns)
cols.remove('user_session')
cols


# In[11]:


# load data 
df_event = nvt.Dataset(raw_df) 

# categorify user_session 
cat_feats = ['user_session'] >> nvt.ops.Categorify()

workflow = nvt.Workflow(cols + cat_feats)
workflow.fit(df_event)
df = workflow.transform(df_event).to_ddf().compute()


# In[12]:


df.head()


# In[13]:


raw_df = None
del(raw_df)


# In[14]:


gc.collect()


# ### Removing consecutive repeated (user, item) interactions

# We keep repeated interactions on the same items, removing only consecutive interactions, because it might be due to browser tab refreshes or different interaction types (e.g. click, add-to-card, purchase)

# In[15]:


get_ipython().run_cell_magic('time', '', 'df = df.sort_values([\'user_session\', \'event_time_ts\']).reset_index(drop=True)\n\nprint("Count with in-session repeated interactions: {}".format(len(df)))\n# Sorts the dataframe by session and timestamp, to remove consecutive repetitions\ndf[\'product_id_past\'] = df[\'product_id\'].shift(1).fillna(0)\ndf[\'session_id_past\'] = df[\'user_session\'].shift(1).fillna(0)\n#Keeping only no consecutive repeated in session interactions\ndf = df[~((df[\'user_session\'] == df[\'session_id_past\']) & \\\n             (df[\'product_id\'] == df[\'product_id_past\']))]\nprint("Count after removed in-session repeated interactions: {}".format(len(df)))\ndel(df[\'product_id_past\'])\ndel(df[\'session_id_past\'])\n\ngc.collect()\n')


# ### Include the item first time seen feature (for recency calculation)

# We create `prod_first_event_time_ts` column which indicates the timestamp that an item was seen first time.

# In[16]:


item_first_interaction_df = df.groupby('product_id').agg({'event_time_ts': 'min'}) \
            .reset_index().rename(columns={'event_time_ts': 'prod_first_event_time_ts'})
item_first_interaction_df.head()
gc.collect()


# In[17]:


df = df.merge(item_first_interaction_df, on=['product_id'], how='left').reset_index(drop=True)


# In[18]:


df.head()


# In[19]:


del(item_first_interaction_df)
item_first_interaction_df=None
gc.collect()


# In this tutorial, we only use one week of data from Oct 2019 dataset.

# In[20]:


# check the min date
df['event_time_dt'].min()


# In[21]:


# Filters only the first week of the data.
df = df[df['event_time_dt'] < np.datetime64('2019-10-08')].reset_index(drop=True)


# We verify that we only have the first week of Oct-2019 dataset.

# In[22]:


df['event_time_dt'].max()


# We drop `event_time_dt` column as it will not be used anymore.

# In[23]:


df = df.drop(['event_time_dt'],  axis=1)


# In[24]:


df.head()


# Save the data as a single parquet file to be used in the ETL notebook.

# In[25]:


# save df as parquet files on disk
df.to_parquet(os.path.join(INPUT_DATA_DIR, 'Oct-2019.parquet'))


# - Shut down the kernel

# In[ ]:


import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

