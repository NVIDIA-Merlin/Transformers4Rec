#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # End-to-end session-based recommendations with PyTorch

# In recent years, several deep learning-based algorithms have been proposed for recommendation systems while its adoption in industry deployments have been steeply growing. In particular, NLP inspired approaches have been successfully adapted for sequential and session-based recommendation problems, which are important for many domains like e-commerce, news and streaming media. Session-Based Recommender Systems (SBRS) have been proposed to model the sequence of interactions within the current user session, where a session is a short sequence of user interactions typically bounded by user inactivity. They have recently gained popularity due to their ability to capture short-term or contextual user preferences towards items. 
# 
# The field of NLP has evolved significantly within the last decade, particularly due to the increased usage of deep learning. As a result, state of the art NLP approaches have inspired RecSys practitioners and researchers to adapt those architectures, especially for sequential and session-based recommendation problems. Here, we leverage one of the state-of-the-art Transformer-based architecture, [XLNet](https://arxiv.org/abs/1906.08237) with Masked Language Modeling (MLM) training technique (see our [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) for details) for training a session-based model.
# 
# In this end-to-end-session-based recommnender model example, we use `Transformers4Rec` library, which leverages the popular [HuggingFace’s Transformers](https://github.com/huggingface/transformers) NLP library and make it possible to experiment with cutting-edge implementation of such architectures for sequential and session-based recommendation problems. For detailed explanations of the building blocks of Transformers4Rec meta-architecture visit [getting-started-session-based](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/getting-started-session-based) and [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) example notebooks.

# ## 1. Model definition using Transformers4Rec

# In the previous notebook, we have created sequential features and saved our processed data frames as parquet files, and now we use these processed parquet files to train a session-based recommendation model with XLNet architecture.

# ### 1.1 Get the schema 

# The library uses a schema format to configure the input features and automatically creates the necessary layers. This *protobuf* text file contains the description of each input feature by defining: the name, the type, the number of elements of a list column,  the cardinality of a categorical feature and the min and max values of each feature. In addition, the annotation field contains the tags such as specifying `continuous` and `categorical` features, the `target` column or the `item_id` feature, among others.

# In[ ]:


from merlin_standard_lib import Schema
SCHEMA_PATH = "schema_demo.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)
get_ipython().system('cat $SCHEMA_PATH')


# We can select the subset of features we want to use for training the model by their tags or their names.

# In[ ]:


schema = schema.select_by_name(
   ['item_id-list_seq', 'category-list_seq', 'product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq']
)


# ### 3.2 Define the end-to-end Session-based Transformer-based recommendation model

# For session-based recommendation model definition, the end-to-end model definition requires four steps:
# 
# 1. Instantiate [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.features.html?highlight=tabularsequence#transformers4rec.tf.features.sequence.TabularSequenceFeatures) input-module from schema to prepare the embedding tables of categorical variables and project continuous features, if specified. In addition, the module provides different aggregation methods (e.g. 'concat', 'elementwise-sum') to merge input features and generate the sequence of interactions embeddings. The module also supports language modeling tasks to prepare masked labels for training and evaluation (e.g: 'mlm' for masked language modeling) 
# 
# 2. Next, we need to define one or multiple prediction tasks. For this demo, we are going to use [NextItemPredictionTask](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.model.html?highlight=nextitem#transformers4rec.tf.model.prediction_task.NextItemPredictionTask) with `Masked Language modeling`: during training randomly selected items are masked and predicted using the unmasked sequence items. For inference it is meant to always predict the next item to be interacted with.
# 
# 3. Then we construct a `transformer_config` based on the architectures provided by [Hugging Face Transformers](https://github.com/huggingface/transformers) framework. </a>
# 
# 4. Finally we link the transformer-body to the inputs and the prediction tasks to get the final pytorch `Model` class.
#     
# For more details about the features supported by each sub-module, please check out the library [documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/index.html) page.

# In[ ]:


from transformers4rec import torch as tr

max_sequence_length, d_model = 20, 320
# Define input module to process tabular input-features and to prepare masked inputs
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=max_sequence_length,
    continuous_projection=64,
    aggregation="concat",
    d_output=d_model,
    masking="mlm",
)

# Define Next item prediction-task 
prediction_task = tr.NextItemPredictionTask(hf_format=True, weight_tying=True)

# Define the config of the XLNet Transformer architecture
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
)

#Get the end-to-end model 
model = transformer_config.to_torch_model(input_module, prediction_task)


# In[ ]:


model


# ### 3.3. Daily Fine-Tuning: Training over a time window¶

# Now that the model is defined, we are going to launch training. For that, Transfromers4rec extends HF Transformers Trainer class to adapt the evaluation loop for session-based recommendation task and the calculation of ranking metrics. The original `train()` method is not modified meaning that we leverage the efficient training implementation from that library, which manages for example half-precision (FP16) training.

# #### Sets Training arguments

# An additional argument `data_loader_engine` is defined to automatically load the features needed for training using the schema. The default value is `nvtabular` for optimized GPU-based data-loading.  Optionally a `PyarrowDataLoader` (`pyarrow`) can also be used as a basic option, but it is slower and works only for small datasets, as the full data is loaded to CPU memory.

# In[ ]:


training_args = tr.trainer.T4RecTrainingArguments(
            output_dir="./tmp",
            max_sequence_length=20,
            data_loader_engine='nvtabular',
            num_train_epochs=10, 
            dataloader_drop_last=False,
            per_device_train_batch_size = 384,
            per_device_eval_batch_size = 512,
            learning_rate=0.0005,
            fp16=True,
            report_to = [],
            logging_steps=200
        )


# #### Instantiate the trainer

# In[ ]:


recsys_trainer = tr.Trainer(
    model=model,
    args=training_args,
    schema=schema,
    compute_metrics=True)


# #### Launches daily Training and Evaluation

# In this demo, we will use the `fit_and_evaluate` method that allows us to conduct a time-based finetuning by iteratively training and evaluating using a sliding time window: At each iteration, we use training data of a specific time index $t$ to train the model then we evaluate on the validation data of next index $t + 1$. Particularly, we set start time to 178 and end time to 180.

# In[ ]:


from transformers4rec.torch.utils.examples_utils import fit_and_evaluate
OT_results = fit_and_evaluate(recsys_trainer, start_time_index=178, end_time_index=180, input_dir='./preproc_sessions_by_day')


# #### Visualize the average over time metrics

# `OT_results` is a list of scores (accuracy metrics) for evaluation based on given start and end time_index. Since in this example we do evaluation on days 179, 180 and 181, we get three metrics in the list one for each day.

# In[ ]:


OT_results


# In[ ]:


import numpy as np
# take the average of metric values over time
avg_results = {k: np.mean(v) for k,v in OT_results.items()}
for key in sorted(avg_results.keys()): 
    print(" %s = %s" % (key, str(avg_results[key]))) 


# #### Saves the model

# In[ ]:


recsys_trainer._save_model_and_checkpoint(save_model_class=True)


# #### Exports the preprocessing workflow and model in the format required by Triton server:** 
# 
# NVTabular’s `export_pytorch_ensemble()` function enables us to create model files and config files to be served to Triton Inference Server. 

# In[ ]:


from nvtabular.inference.triton import export_pytorch_ensemble
from nvtabular.workflow import Workflow
workflow = Workflow.load("workflow_etl")

export_pytorch_ensemble(
    model,
    workflow,
    sparse_max=recsys_trainer.get_train_dataloader().dataset.sparse_max,
    name= "t4r_pytorch",
    model_path= "/workspace/TF4Rec/models/",
    label_columns =[],
)


# ## 4. Serving Ensemble Model to the Triton Inference Server

# NVIDIA [Triton Inference Server (TIS)](https://github.com/triton-inference-server/server) simplifies the deployment of AI models at scale in production. TIS provides a cloud and edge inferencing solution optimized for both CPUs and GPUs. It supports a number of different machine learning frameworks such as TensorFlow and PyTorch.
# 
# The last step of machine learning (ML)/deep learning (DL) pipeline is to deploy the ETL workflow and saved model to production. In the production setting, we want to transform the input data as done during training (ETL). We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integer before we use the DL model for a prediction. Therefore, we deploy the NVTabular workflow with the PyTorch model as an ensemble model to Triton Inference. The ensemble model guarantees that the same transformation is applied to the raw inputs.
# 
# 
# In this section, you will learn how to
# - to deploy saved NVTabular and PyTorch models to Triton Inference Server 
# - send requests for predictions and get responses.

# ### 4.1. Pull and Start Inference Container
# 
# At this point, before connecing to the Triton Server, we launch the inference docker container and then load the ensemble `t4r_pytorch` to the inference server. This is done with the scripts below:
# 
# **Launch the docker container**
# ```
# docker run -it --gpus device=0 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v <path_to_saved_models>:/workspace/models/ nvcr.io/nvidia/merlin/merlin-inference:21.09
# ```
# This script will mount your local model-repository folder that includes your saved models from the previous cell to `/workspace/models` directory in the merlin-inference docker container.
# 
# **Start triton server**<br>
# After you started the merlin-inference container, you can start triton server with the command below. You need to provide correct path of the models folder.
# 
# 
# ```
# tritonserver --model-repository=<path_to_models> --model-control-mode=explicit
# ```
# Note: The model-repository path for our example is `/workspace/models`. The models haven't been loaded, yet. Below, we will request the Triton server to load the saved ensemble model below.

# ### Connect to the Triton Inference Server and check if the server is alive

# In[ ]:


import tritonhttpclient
try:
    triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))
triton_client.is_server_live()


# ### Load raw data for inference
# We select the last 50 interactions and filter out sessions with less than 2 interactions. 

# In[ ]:


import pandas as pd
interactions_merged_df = pd.read_parquet("/raid/data/yoochoose/interactions_merged_df.parquet")
interactions_merged_df = interactions_merged_df.sort_values('timestamp')
batch = interactions_merged_df[-50:]
sessions_to_use = batch.session_id.value_counts()
filtered_batch = batch[batch.session_id.isin(sessions_to_use[sessions_to_use.values>1].index.values)]


# ### Send the request to triton server

# In[ ]:


triton_client.get_model_repository_index()


# ### Load the ensemble model to triton
# If all models are loaded successfully, you should be seeing `successfully loaded` status next to each model name on your terminal.

# In[ ]:


triton_client.load_model(model_name="t4r_pytorch")


# In[ ]:


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


# - Visualise top-k predictions

# In[ ]:


from transformers4rec.torch.utils.examples_utils import visualize_response
visualize_response(filtered_batch, response, top_k=5, session_col='session_id')


# As you noticed, we first got prediction results (logits) from the trained model head, and then by using a handy util function `visualize_response` we extracted top-k encoded item-ids from logits. Basically, we generated recommended items for a given session.
# 
# This is the end of the tutorial. You successfully
# 
# - performed feature engineering with NVTabular
# - trained transformer architecture based session-based recommendation models with Transformers4Rec
# - deployed a trained model to Triton Inference Server, sent request and got responses from the server.

# **Unload models**

# In[ ]:


triton_client.unload_model(model_name="t4r_pytorch")
triton_client.unload_model(model_name="t4r_pytorch_nvt")
triton_client.unload_model(model_name="t4r_pytorch_pt")


# ## References

# - Merlin Transformers4rec: https://github.com/NVIDIA-Merlin/Transformers4Rec
# 
# - Merlin NVTabular: https://github.com/NVIDIA-Merlin/NVTabular/tree/main/nvtabular
# 
# - Triton inference server: https://github.com/triton-inference-server
