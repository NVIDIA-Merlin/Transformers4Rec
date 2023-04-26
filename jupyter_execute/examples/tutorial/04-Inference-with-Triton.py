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


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_transformers4rec_tutorial-04-inference-with-triton/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Triton for Recommender Systems

# NVIDIA [Triton Inference Server (TIS)](https://github.com/triton-inference-server/server) simplifies the deployment of AI models at scale in production. The Triton Inference Server allows us to deploy and serve our model for inference. It supports a number of different machine learning frameworks such as TensorFlow and PyTorch.
# 
# The last step of machine learning (ML)/deep learning (DL) pipeline is to deploy the ETL workflow and saved model to production. In the production setting, we want to transform the input data as done during training (ETL). We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integer before we use the DL model for a prediction. Therefore, we deploy the NVTabular workflow with the PyTorch model as an ensemble model to Triton Inference. The ensemble model guarantees that the same transformation is applied to the raw inputs.

# ![](_images/torch_triton.png)

# **Objectives:**
# 
# Learn how to deploy a model to Triton
# 1. Deploy saved NVTabular and PyTorch models to Triton Inference Server
# 2. Sent requests for predictions

# ## Pull and start Inference docker container

# At this point, we start the Triton Inference Server (TIS) and then load the exported ensemble `t4r_pytorch` to the inference server. You can start triton server with the command below. Note that, you need to provide correct path of the models folder.
# 
# ```
# tritonserver --model-repository=<path_to_models> --model-control-mode=explicit
# ```
# The model-repository path for our example is `/workspace/models`. The models haven't been loaded, yet. Below, we will request the Triton server to load the saved ensemble model.

# ## 1. Deploy PyTorch and NVTabular Model to Triton Inference Server

# Our Triton server has already been launched and is ready to make requests. Remember we already exported the saved PyTorch model in the previous notebook, and generated the config files for Triton Inference Server.

# In[2]:


# Import dependencies
import os
from time import time

import numpy as np
import sys
import cudf


# ## 1.2 Review exported files

# Triton expects a specific directory structure for our models as the following format:

# ```
# <model-name>/
# [config.pbtxt]
# <version-name>/
#   [model.savedmodel]/
#     <pytorch_saved_model_files>/
#       ...
# ```

# Let's check out our model repository layout. You can install tree library with `apt-get install tree`, and then run `!tree /workspace/models/` to print out the model repository layout as below:
# 
# ```
# ├── t4r_pytorch
# │   ├── 1
# │   └── config.pbtxt
# ├── t4r_pytorch_nvt
# │   ├── 1
# │   │   ├── model.py
# │   │   ├── __pycache__
# │   │   │   └── model.cpython-38.pyc
# │   │   └── workflow
# │   │       ├── categories
# │   │       │   ├── cat_stats.category_id.parquet
# │   │       │   ├── unique.brand.parquet
# │   │       │   ├── unique.category_code.parquet
# │   │       │   ├── unique.category_id.parquet
# │   │       │   ├── unique.event_type.parquet
# │   │       │   ├── unique.product_id.parquet
# │   │       │   ├── unique.user_id.parquet
# │   │       │   └── unique.user_session.parquet
# │   │       ├── metadata.json
# │   │       └── workflow.pkl
# │   └── config.pbtxt
# └── t4r_pytorch_pt
#     ├── 1
#     │   ├── model_info.json
#     │   ├── model.pkl
#     │   ├── model.pth
#     │   ├── model.py
#     │   └── __pycache__
#     │       └── model.cpython-38.pyc
#     └── config.pbtxt
# ```

# Triton needs a [config file](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) to understand how to interpret the model. Let's look at the generated config file. It defines the input columns with datatype and dimensions and the output layer. Manually creating this config file can be complicated and NVTabular generates it with the `export_pytorch_ensemble()` function, which we used in the previous notebook.
# 
# The [config file](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) needs the following information:
# * `name`: The name of our model. Must be the same name as the parent folder.
# * `platform`: The type of framework serving the model.
# * `input`: The input our model expects.
#   * `name`: Should correspond with the model input name.
#   * `data_type`: Should correspond to the input's data type.
#   * `dims`: The dimensions of the *request* for the input. For models that support input and output tensors with variable-size dimensions, those dimensions can be listed as -1 in the input and output configuration.
# * `output`: The output parameters of our model.
#   * `name`: Should correspond with the model output name.
#   * `data_type`: Should correspond to the output's data type.
#   * `dims`: The dimensions of the output.

# ## 1.3. Loading Model

# Next, let's build a client to connect to our server. The `InferenceServerClient` object is what we'll be using to talk to Triton.

# In[3]:


import tritonhttpclient

try:
    triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))
triton_client.is_server_live()


# In[4]:


triton_client.get_model_repository_index()


# We load the ensemble model

# In[6]:


model_name = "t4r_pytorch"
#triton_client.load_model(model_name=model_name)


# If all models are loaded successfully, you should be seeing successfully loaded status next to each model name on your terminal.

# ## 2. Sent Requests for Predictions

# Load raw data for inference: We select the first 50 interactions and filter out sessions with less than 2 interactions. For this tutorial, just as an example we use the `Oct-2019` dataset that we used for model training.

# In[7]:


INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")
df= cudf.read_parquet(os.path.join(INPUT_DATA_DIR, 'Oct-2019.parquet'))
df=df.sort_values('event_time_ts')
batch = df.iloc[:50,:]


# In[8]:


sessions_to_use = batch.user_session.value_counts()
filtered_batch = batch[batch.user_session.isin(sessions_to_use[sessions_to_use.values>1].index.values)]


# In[9]:


filtered_batch.head()


# In[10]:


import warnings

warnings.filterwarnings("ignore")


# In[11]:


import nvtabular.inference.triton as nvt_triton
import tritonclient.grpc as grpcclient

inputs = nvt_triton.convert_df_to_triton_input(filtered_batch.columns, filtered_batch, grpcclient.InferInput)

output_names = ["output"]

outputs = []
for col in output_names:
    outputs.append(grpcclient.InferRequestedOutput(col))
    
MODEL_NAME_NVT = "t4r_pytorch"

with grpcclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer(MODEL_NAME_NVT, inputs)
    print(col, ':\n', response.as_numpy(col))


# #### Visualise top-k predictions

# In[12]:


from transformers4rec.torch.utils.examples_utils import visualize_response
visualize_response(filtered_batch, response, top_k=5, session_col='user_session')


# As you see we first got prediction results (logits) from the trained model head, and then by using a handy util function `visualize_response` we extracted top-k encoded item-ids from logits. Basically, we  generated recommended items for a given session.
# 
# This is the end of the tutorial. You successfully ...
# 1. performed feature engineering with NVTabular
# 2. trained transformer architecture based session-based recommendation models with Transformers4Rec 
# 3. deployed a trained model to Triton Inference Server, sent request and got responses from the server.

# ### Unload models and shut down the kernel

# In[ ]:


triton_client.unload_model(model_name="t4r_pytorch")
triton_client.unload_model(model_name="t4r_pytorch_nvt")
triton_client.unload_model(model_name="t4r_pytorch_pt")


# In[ ]:


import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)

