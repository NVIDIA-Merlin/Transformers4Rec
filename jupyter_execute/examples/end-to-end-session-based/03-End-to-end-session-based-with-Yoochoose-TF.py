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


# # End-to-end session-based recommendation with TensorFlow

# In recent years, several deep learning-based algorithms have been proposed for recommendation systems while its adoption in industry deployments have been steeply growing. In particular, NLP inspired approaches have been successfully adapted for sequential and session-based recommendation problems, which are important for many domains like e-commerce, news and streaming media. Session-Based Recommender Systems (SBRS) have been proposed to model the sequence of interactions within the current user session, where a session is a short sequence of user interactions typically bounded by user inactivity. They have recently gained popularity due to their ability to capture short-term or contextual user preferences towards items. 
# 
# The field of NLP has evolved significantly within the last decade, particularly due to the increased usage of deep learning. As a result, state of the art NLP approaches have inspired RecSys practitioners and researchers to adapt those architectures, especially for sequential and session-based recommendation problems. Here, we leverage one of the state-of-the-art Transformer-based architecture, [XLNet](https://arxiv.org/abs/1906.08237) with `Causal Language Modeling (CLM)` training technique. Causal LM is the task of predicting the token following a sequence of tokens, where the model only attends to the left context, i.e. models the probability of a token given the previous tokens in a sentence (Lample and Conneau, 2019).
# 
# In this end-to-end-session-based recommnender model example, we use `Transformers4Rec` library, which leverages the popular [HuggingFace’s Transformers](https://github.com/huggingface/transformers) NLP library and make it possible to experiment with cutting-edge implementation of such architectures for sequential and session-based recommendation problems. For detailed explanations of the building blocks of Transformers4Rec meta-architecture visit [getting-started-session-based](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/getting-started-session-based) and [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) example notebooks.

# ## 1. Model definition using Transformers4Rec

# In the previous notebook, we have created sequential features and saved our processed data frames as parquet files, and now we use these processed parquet files to train a session-based recommendation model with XLNet architecture.

# ### 1.1 Import Libraries

# In[2]:


import os
import glob
import cudf
import numpy as np

from nvtabular.loader.tensorflow import KerasSequenceLoader

from transformers4rec import tf as tr
from transformers4rec.tf.ranking_metric import NDCGAt, RecallAt


# In[ ]:


# disable INFO and DEBUG logging everywhere
import logging
logging.disable(logging.WARNING)


# In[3]:


# avoid numba warnings
from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


# ### 1.2 Get the schema 

# The library uses a schema format to configure the input features and automatically creates the necessary layers. This *protobuf* text file contains the description of each input feature by defining: the name, the type, the number of elements of a list column,  the cardinality of a categorical feature and the min and max values of each feature. In addition, the annotation field contains the tags such as specifying `continuous` and `categorical` features, the `target` column or the `item_id` feature, among others.

# In[4]:


from merlin_standard_lib import Schema
SCHEMA_PATH = "schema_demo.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)
get_ipython().system('cat $SCHEMA_PATH')


# We can select the subset of features we want to use for training the model by their tags or their names.

# In[5]:


schema = schema.select_by_name(
    ['item_id-list_seq', 'category-list_seq', 'product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq']
)


# ### 3.2 Define the end-to-end Session-based Transformer-based recommendation model

# For session-based recommendation model definition, the end-to-end model definition requires four steps:
# 
# 1. Instantiate [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.features.html?highlight=tabularsequence#transformers4rec.tf.features.sequence.TabularSequenceFeatures) input-module from schema to prepare the embedding tables of categorical variables and project continuous features, if specified. In addition, the module provides different aggregation methods (e.g. 'concat', 'elementwise-sum') to merge input features and generate the sequence of interactions embeddings. The module also supports language modeling tasks to prepare masked labels for training and evaluation (e.g: 'clm' for causal language modeling).
# 
# 2. Next, we need to define one or multiple prediction tasks. For this demo, we are going to use [NextItemPredictionTask](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.model.html?highlight=nextitem#transformers4rec.tf.model.prediction_task.NextItemPredictionTask) with `Causal Language modeling (CLM)`.
# 
# 3. Then we construct a `transformer_config` based on the architectures provided by [Hugging Face Transformers](https://github.com/huggingface/transformers) framework. </a>
# 
# 4. Finally we link the transformer-body to the inputs and the prediction tasks to get the final Tensorflow `Model` class.
#     
# For more details about the features supported by each sub-module, please check out the library [documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/index.html) page.

# In[6]:


max_sequence_length, d_model = 20, 320

# Define the evaluation top-N metrics and the cut-offs
metrics = [
    NDCGAt(top_ks=[10, 20], labels_onehot=True), 
    RecallAt(top_ks=[10, 20], labels_onehot=True)
]

# Define input module to process tabular input-features and to prepare masked inputs
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=max_sequence_length,
    continuous_projection=64,
    aggregation="concat",
    d_output=d_model,
    masking="clm",
)

# Define Next item prediction-task 
prediction_task = tr.NextItemPredictionTask(weight_tying=True, metrics=metrics)

# Define the config of the XLNet Transformer architecture
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
)

# Get the end-to-end model
model = transformer_config.to_tf_model(input_module, prediction_task)


# In[7]:


model


# ### 3.3. Daily Fine-Tuning: Training over a time window¶

# Now that the model is defined, we are now going to launch training. In this example, we will conduct a time-based finetuning by iteratively training and evaluating using a sliding time window: At each iteration, we use training data of a specific time index $t$ to train the model then we evaluate on the validation data of next index $t + 1$. Particularly, we set start time to 178 and end time to 180. Note that, we are using tf.keras' `model.fit()` and `model.evaluate()` methods, where we train the model with model.fit(), and evaluate it with model.evaluate().

# #### Sets DataLoader

# We use the NVTabular `KerasSequenceLoader` Dataloader for optimized loading of multiple features from input parquet files. In our experiments, we see a speed-up by 9x of the same training workflow with NVTabular dataloader. You can learn more about this data loader [here](https://nvidia-merlin.github.io/NVTabular/main/training/tensorflow.html) and [here](https://medium.com/nvidia-merlin/training-deep-learning-based-recommender-systems-9x-faster-with-tensorflow-cc5a2572ea49).

# In[8]:


# Define categorical and continuous columns
x_cat_names = ['item_id-list_seq', 'category-list_seq']
x_cont_names = ['product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq']

# dictionary representing max sequence length for each column
sparse_features_max = {
    fname: 20
    for fname in x_cat_names + x_cont_names
}


# In[ ]:


def get_dataloader(paths_or_dataset, batch_size=384):
    dataloader = KerasSequenceLoader(
        paths_or_dataset,
        batch_size=batch_size,
        label_names=None,
        cat_names=x_cat_names,
        cont_names=x_cont_names,
        sparse_names=list(sparse_features_max.keys()),
        sparse_max=sparse_features_max,
        sparse_as_dense=True,
    )
    return dataloader.map(lambda X, y: (X, []))


# The reason we set the targets to [] in the data-loader because the true item labels are computed internally by the `MaskSequence` class.

# In[9]:


import tensorflow as tf

opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
# set it to True if to run the model eagerly
model.compile(optimizer=opt, run_eagerly=False) 


# In[10]:


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", './preproc_sessions_by_day')


# In[11]:


start_time_window_index = 178
final_time_window_index = 180
# Iterating over days of one week
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))

    # Train on day related to time_index 
    print('*' * 20)
    print("Launch training for day %s are:" %time_index)
    print('*' * 20 + '\n')
    train_loader = get_dataloader(train_paths, batch_size=384)
    losses = model.fit(train_loader, epochs=5)
    model.reset_metrics()
    # Evaluate on the following day
    eval_loader = get_dataloader(eval_paths, batch_size=512)
    eval_metrics = model.evaluate(eval_loader, return_dict=True)
    print('*'*20)
    print("Eval results for day %s are:\t" %time_index_eval)
    print('\n' + '*' * 20 + '\n')
    for key in sorted(eval_metrics.keys()):
        print(" %s = %s" % (key, str(eval_metrics[key])))


# #### Exports the preprocessing workflow and model in the format required by Triton server:** 
# 
# NVTabular’s `export_tensorflow_ensemble()` function enables us to create model files and config files to be served to Triton Inference Server. 

# In[12]:


import nvtabular as nvt
workflow = nvt.Workflow.load('workflow_etl')


# In[14]:


from nvtabular.inference.triton import export_tensorflow_ensemble
export_tensorflow_ensemble(
    model,
    workflow,
    name="t4r_tf",
    model_path='/workspace/TF4Rec/models/tf/',
    label_columns=[],
    sparse_max=sparse_features_max
)


# ## 4. Serving Ensemble Model to the Triton Inference Server

# NVIDIA [Triton Inference Server (TIS)](https://github.com/triton-inference-server/server) simplifies the deployment of AI models at scale in production. TIS provides a cloud and edge inferencing solution optimized for both CPUs and GPUs. It supports a number of different machine learning frameworks such as TensorFlow and PyTorch.
# 
# The last step of machine learning (ML)/deep learning (DL) pipeline is to deploy the ETL workflow and saved model to production. In the production setting, we want to transform the input data as done during training (ETL). We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integer before we use the DL model for a prediction. Therefore, we deploy the NVTabular workflow with the Tensorflow model as an ensemble model to Triton Inference. The ensemble model guarantees that the same transformation is applied to the raw inputs.
# 
# 
# In this section, you will learn how to
# - to deploy saved NVTabular and Tensorflow models to Triton Inference Server 
# - send requests for predictions and get responses.

# ### 4.1. Pull and Start Inference Container
# 
# At this point, before connecing to the Triton Server, we launch the inference docker container and then load the ensemble `t4r_tf` to the inference server. This is done with the scripts below:
# 
# **Launch the docker container**
# ```
# docker run -it --gpus device=0 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v <path_to_saved_models>:/workspace/models/ nvcr.io/nvidia/merlin/merlin-inference:21.11
# ```
# This script will mount your local model-repository folder that includes your saved models from the previous cell to `/workspace/models` directory in the merlin-inference docker container.
# 
# **Start triton server**<br>
# After you started the merlin-inference container, you can start triton server with the command below. You need to provide correct path of the models folder.
# 
# 
# ```
# tritonserver --model-repository=<path_to_models> --backend-config=tensorflow,version=2 --model-control-mode=explicit
# ```
# Note: The model-repository path for our example is `/workspace/models`. The models haven't been loaded, yet. Below, we will request the Triton server to load the saved ensemble model below.

# ### Connect to the Triton Inference Server and check if the server is alive

# In[15]:


import tritonhttpclient
try:
    triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))
triton_client.is_server_live()


# ### Load raw data for inference
# We select the last 50 interactions and filter out sessions with less than 2 interactions. 

# In[16]:


interactions_merged_df = cudf.read_parquet('/workspace/data/interactions_merged_df.parquet')
interactions_merged_df = interactions_merged_df.sort_values('timestamp')
batch = interactions_merged_df[-50:]
sessions_to_use = batch.session_id.value_counts()
# ignore sessions with less than 2 interactions
filtered_batch = batch[batch.session_id.isin(sessions_to_use[sessions_to_use.values > 1].index.values)]


# ### Send the request to triton server

# In[17]:


triton_client.get_model_repository_index()


# ### Load the ensemble model to triton
# If all models are loaded successfully, you should be seeing `successfully loaded` status next to each model name on your terminal.

# In[18]:


triton_client.load_model(model_name="t4r_tf")


# In[19]:


import nvtabular.inference.triton as nvt_triton
import tritonclient.grpc as grpcclient

inputs = nvt_triton.convert_df_to_triton_input(filtered_batch.columns, filtered_batch, grpcclient.InferInput)

output_names = ["output_1"]

outputs = []
for col in output_names:
    outputs.append(grpcclient.InferRequestedOutput(col))

MODEL_NAME_NVT = "t4r_tf"

with grpcclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer(MODEL_NAME_NVT, inputs)
    print(col, ':\n', response.as_numpy(col))


# In[20]:


def visualize_response(batch, response, top_k, session_col="session_id"):
    """
    Util function to extract top-k encoded item-ids from logits
    Parameters
    """
    sessions = batch[session_col].drop_duplicates().values
    predictions = response.as_numpy("output_1")
    top_preds = np.argpartition(predictions, -top_k, axis=1)[:, -top_k:]
    for session, next_items in zip(sessions, top_preds):
        print(
            "- Top-%s predictions for session `%s`: %s\n"
            % (top_k, session, " || ".join([str(e) for e in next_items]))
        )


# - Visualise top-k predictions

# In[21]:


visualize_response(filtered_batch, response, top_k=5, session_col='session_id')


# As you noticed, we first got prediction results (logits) from the trained model head, and then by using a handy util function `visualize_response` we extracted top-k encoded item-ids from logits. Basically, we generated recommended items for a given session.
# 
# This is the end of these example notebooks. You successfully
# 
# - performed feature engineering with NVTabular
# - trained transformer architecture based session-based recommendation models with Transformers4Rec
# - deployed a trained model to Triton Inference Server, sent request and got responses from the server.

# **Unload models**

# In[ ]:


triton_client.unload_model(model_name="t4r_tf")
triton_client.unload_model(model_name="t4r_tf_nvt")
triton_client.unload_model(model_name="t4r_tf_tf")


# ## References

# - Merlin Transformers4rec: https://github.com/NVIDIA-Merlin/Transformers4Rec
# - Merlin NVTabular: https://github.com/NVIDIA-Merlin/NVTabular/tree/main/nvtabular
# - Triton inference server: https://github.com/triton-inference-server
# - Guillaume Lample, and Alexis Conneau. "Cross-lingual language model pretraining." arXiv preprint arXiv:1901.07291
