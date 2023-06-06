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


# <img src="https://developer.download.nvidia.com//notebooks/dlsw-notebooks/remtting-started-session-based-03-serving-session-based-model-torch-backend/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Serving a Session-based Recommendation model with Torch Backend

# This notebook is created using the latest stable [merlin-pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch/tags) container.
# 
# At this point, when you reach out to this notebook, we expect that you have already executed the `01-ETL-with-NVTabular.ipynb` and `02-session-based-XLNet-with-PyT.ipynb` notebooks, and saved the NVT workflow and the trained session-based model.
# 
# In this notebook, you are going to learn how you can serve a trained Transformer-based PyTorch model on NVIDIA [Triton Inference Server](https://github.com/triton-inference-server/server)  (TIS) with Torch backend using [Merlin systems](https://github.com/NVIDIA-Merlin/systems) library. One common way to do inference with a trained model is to use TorchScript, an intermediate representation of a PyTorch model that can be run in Python as well as in a high performance environment like C++. [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) is actually the recommended model format for scaled inference and deployment. TIS [PyTorch (LibTorch) backend](https://github.com/triton-inference-server/pytorch_backend) is designed to run TorchScript models using the PyTorch C++ API.
# 
# [Triton Inference Server](https://github.com/triton-inference-server/server) (TIS) simplifies the deployment of AI models at scale in production. TIS provides a cloud and edge inferencing solution optimized for both CPUs and GPUs. It supports a number of different machine learning frameworks such as TensorFlow and PyTorch.

# ### Import required libraries

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cudf
import glob
import numpy as np
import pandas as pd
import torch 

from transformers4rec import torch as tr
from merlin.io import Dataset

from merlin.core.dispatch import make_df 
from merlin.systems.dag import Ensemble  
from merlin.systems.dag.ops.pytorch import PredictPyTorch 
from merlin.systems.dag.ops.workflow import TransformWorkflow 


# We define the paths

# In[3]:


INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{INPUT_DATA_DIR}/sessions_by_day")
model_path= os.environ.get("model_path", f"{INPUT_DATA_DIR}/saved_model")


# ### Set the schema object

# We create the schema object by reading the processed train parquet file.

# In[4]:


from merlin.schema import Schema
from merlin.io import Dataset

train = Dataset(os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet"))
schema = train.schema


# We need to load the saved model to be able to serve it on TIS.

# In[5]:


import cloudpickle
loaded_model = cloudpickle.load(
                open(os.path.join(model_path, "t4rec_model_class.pkl"), "rb")
            )


# Switch the model to eval mode. We call `model.eval()` before tracing to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this might yield inconsistent inference results.

# In[6]:


model = loaded_model.cuda()
model.eval()


# ### Trace the model
# 
# We serve the model with the PyTorch backend that is used to execute TorchScript models. All models created in PyTorch using the python API must be traced/scripted to produce a TorchScript model. For tracing the model, we use [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) api that takes the model as a Python function or torch.nn.Module, and an example input  that will be passed to the function while tracing.

# In[7]:


train_paths = os.path.join(OUTPUT_DIR, f"{1}/train.parquet")
dataset = Dataset(train_paths)


# Create a dict of tensors to feed it as example inputs in the `torch.jit.trace()`.

# In[8]:


import pandas as pd
from merlin.table import TensorTable, TorchColumn
from merlin.table.conversions import convert_col

df = cudf.read_parquet(train_paths, columns=model.input_schema.column_names)
table = TensorTable.from_df(df.loc[:100])
for column in table.columns:
    table[column] = convert_col(table[column], TorchColumn)
model_input_dict = table.to_dict()


# In[9]:


model_input_dict


# In[10]:


traced_model = torch.jit.trace(model, model_input_dict, strict=True)


# Generate model input and output schemas to feed in the `PredictPyTorch` operator below.

# In[11]:


input_schema = model.input_schema
output_schema = model.output_schema


# In[12]:


input_schema


# Let's create a folder that we can store the exported models and the config files.

# In[13]:


import shutil
ens_model_path = os.environ.get("ens_model_path", f"{INPUT_DATA_DIR}/models")
# Make sure we have a clean stats space for Dask
if os.path.isdir(ens_model_path):
    shutil.rmtree(ens_model_path)
os.mkdir(ens_model_path)


# We want to serve NVT model and our trained session-based model together as an ensemble to the Triton Inference Server. That way we can send raw requests to Triton and return back item scores per session. For that we need to load our save workflow first.

# In[14]:


from nvtabular.workflow import Workflow
workflow = Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow_etl"))
print(workflow.input_schema.column_names)


# For transforming the raw input features during inference, we use [TransformWorkflow](https://github.com/NVIDIA-Merlin/systems/blob/main/merlin/systems/dag/ops/workflow.py) operator that ensures the workflow is correctly saved and packaged with the required config so the server will know how to load it. We use [PredictPyTorch](https://github.com/NVIDIA-Merlin/systems/blob/main/merlin/systems/dag/ops/pytorch.py) operator that takes a pytorch model and packages it correctly for tritonserver to run on the PyTorch backend.

# In[15]:


torch_op = workflow.input_schema.column_names >> TransformWorkflow(workflow) >> PredictPyTorch(
    traced_model, input_schema, output_schema
)

ensemble = Ensemble(torch_op, workflow.input_schema)


# The last step is to create the ensemble artifacts that Triton Inference Server can consume. To make these artifacts, we import the Ensemble class. The class is responsible for interpreting the graph and exporting the correct files for the server.
# 
# When we create an `Ensemble` object we supply the graph and a schema representing the starting input of the graph. The inputs to the ensemble graph are the inputs to the first operator of out graph. After we created the Ensemble we export the graph, supplying an export path for the `ensemble.export` function. This returns an ensemble config which represents the entire inference pipeline and a list of node-specific configs.

# In[16]:


ens_config, node_configs = ensemble.export(ens_model_path)


# In[17]:


ensemble.input_schema


# ## Starting Triton Server

# It is time to deploy all the models as an ensemble model to Triton Inference Serve TIS. After we export the ensemble, we are ready to start the TIS. You can start triton server by using the following command on your terminal:
# 
# `tritonserver --model-repository=<ensemble_export_path>`
# 
# For the `--model-repository` argument, specify the same path as the export_path that you specified previously in the `ensemble.export` method. This command will launch the server and load all the models to the server. Once all the models are loaded successfully, you should see READY status printed out in the terminal for each loaded model.

# In[18]:


import tritonclient.http as client

# Create a triton client
try:
    triton_client = client.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))


# After we create the client and verified it is connected to the server instance, we can communicate with the server and ensure all the models are loaded correctly.

# In[19]:


# ensure triton is in a good state
triton_client.is_server_live()
triton_client.get_model_repository_index()


# ### Send request to Triton and get the response

# The last step of a machine learning (ML)/deep learning (DL) pipeline is to deploy the model to production, and get responses for a given query or a set of queries.
# In this section, we generate a dataframe that we can serve as a request to TIS. We do serve the raw dataframe and in the production setting, we want to transform the input data as done during training (ETL). We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integer before we use the deployed DL model for a prediction.

# Let's generate a dataframe with raw input values. We can send this dataframe to Triton as a request.

# In[20]:


NUM_ROWS =1000
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

print(df.head(2))


# Once our models are successfully loaded to the TIS, we can now easily send a request to TIS and get a response for our query with send_triton_request utility function.

# In[21]:


from merlin.systems.triton.utils import send_triton_request
response = send_triton_request(workflow.input_schema, df, output_schema.column_names, endpoint="localhost:8001")
response


# In[22]:


response['next-item'].shape


# We return a response for each request in the df. Each row in the `response['next-item']` array corresponds to the logit values per item in the catalog, and one logit score corresponding to the null, OOV and padded items. The first score of each array in each row corresponds to the score for the padded item, OOV or null item. Note that we dont have OOV or null items in our syntheticall generated datasets.
# 
# This is the end of this suit of examples. You successfully performed feature engineering with NVTabular trained transformer architecture based session-based recommendation models with Transformers4Rec deployed the saved workflow and the trained model to Triton Inference Server with Torch backend, sent request and got responses from the server. If you would like to learn how to serve a TF4Rec model with Python backend please visit this [example](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/examples/end-to-end-session-based/02-End-to-end-session-based-with-Yoochoose-PyT.ipynb).
