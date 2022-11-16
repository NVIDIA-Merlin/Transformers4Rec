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


# # Session-based Recommendation with XLNET

# In this notebook we introduce the [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) library for sequential and session-based recommendation. This notebook uses the Tensorflow API built with TensorFlow 2.x, but a PyTorch API is also available (see [example](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/examples/getting-started-session-based/02-session-based-XLNet-with-PyT.ipynb)). Transformers4Rec integrates with the popular [HuggingFace’s Transformers](https://github.com/huggingface/transformers) and make it possible to experiment with cutting-edge implementation of the latest NLP Transformer architectures.  
# 
# We demonstrate how to build a session-based recommendation model with the [XLNET](https://arxiv.org/abs/1906.08237) Transformer architecture. The XLNet architecture was designed to leverage the best of both auto-regressive language modeling and auto-encoding with its Permutation Language Modeling training method. In this example we will use XLNET with causal language modeling (CLM) training method. 

# In the previous notebook we went through our ETL pipeline with NVTabular library, and created sequential features to be used in training a session-based recommendation model. In this notebook we will learn:
# 
# - Accelerating data loading of parquet files with multiple features on Tensorflow using NVTabular library
# - Training and evaluating a Transformer-based (XLNET-CLM) session-based recommendation model with multiple features

# ## Build a DL model with Transformers4Rec library  

# Transformers4Rec supports multiple input features and provides configurable building blocks that can be easily combined for custom architectures:
# 
# - [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.html#transformers4rec.torch.TabularSequenceFeatures) class that reads from schema and creates an input block. This input module combines different types of features (continuous, categorical & text) to a sequence.
# -  [MaskSequence](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/tf/masking.py) to define masking schema and prepare the masked inputs and labels for the selected LM task.
# -  [TransformerBlock](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.html#transformers4rec.torch.TransformerBlock) class that supports HuggingFace Transformers for session-based and sequential-based recommendation models.
# -  [SequentialBlock](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.html#transformers4rec.torch.SequentialBlock) creates the body by mimicking [tf.keras.Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) class. It is designed to define our model as a sequence of layers.
# -  [Head](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.html#transformers4rec.tf.Head) where we define the prediction task of the model.
# -  [NextItemPredictionTask](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.html#transformers4rec.tf.NextItemPredictionTask) is the class to support next item prediction task.
# 
# You can check the [full documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/index.html) of Transformers4Rec if needed.

# Figure 1 illustrates Transformers4Rec meta-architecture and how each module/block interacts with each other.

# ![tf4rec_meta](images/tf4rec_meta2.png)

# ## Import required libraries

# In[2]:


import os
import glob

from nvtabular.loader.tensorflow import KerasSequenceLoader

from transformers4rec import tf as tr
from transformers4rec.tf.ranking_metric import NDCGAt, RecallAt

import logging
logging.disable(logging.WARNING) # disable INFO and DEBUG logging everywhere


# Transformers4Rec library relies on a schema object to automatically build all necessary layers to represent, normalize and aggregate input features. As you can see below, `schema.pb` is a protobuf file that contains metadata including statistics about features such as cardinality, min and max values and also tags features based on their characteristics and dtypes (e.g., categorical, continuous, list, integer).

# In[3]:


# avoid numba warnings
from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


# ## Set the schema 

# In[4]:


from merlin_standard_lib import Schema
SCHEMA_PATH = "schema.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)
get_ipython().system('cat $SCHEMA_PATH')


# In[5]:


# You can select a subset of features for training
schema = schema.select_by_name(['item_id-list_trim', 
                                'category-list_trim', 
                                'timestamp/weekday/sin-list_trim',
                                'timestamp/age_days-list_trim'])


# ## Define the sequential input module

# Below we define our `input` block using the `TabularSequenceFeatures` [class](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/tf/features/sequence.py#L121). The `from_schema()` method processes the schema and creates the necessary layers to represent features and aggregate them. It keeps only features tagged as `categorical` and `continuous` and supports data aggregation methods like `concat` and `elementwise-sum` techniques. It also support data augmentation techniques like stochastic swap noise. It outputs an interaction representation after combining all features and also the input mask according to the training task (more on this later).
# 

# The `max_sequence_length` argument defines the maximum sequence length of our sequential input, and if `continuous_projection` argument is set, all numerical features are concatenated and projected by an MLP block so that continuous features are represented by a vector of size defined by user, which is `64` in this example.

# In[6]:


inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=64,
        masking="clm",
)


# The output of the `TabularSequenceFeatures` module is the sequence of interactions embeddings vectors defined in the following steps:
# - 1. Create sequence inputs: If the schema contains non sequential features, expand each feature to a sequence by repeating the value as many as the `max_sequence_length` value.  
# - 2. Get a representation vector of categorical features: Project each sequential categorical feature using the related embedding table. The resulting tensor is of shape (bs, max_sequence_length, embed_dim).
# - 3. Project scalar values if `continuous_projection` is set : Apply an MLP layer with hidden size equal to `continuous_projection` vector size value. The resulting tensor is of shape (batch_size, max_sequence_length, continuous_projection).
# - 4. Aggregate the list of features vectors to represent each interaction in the sequence with one vector: For example, `concat` will concat all vectors based on the last dimension `-1` and the resulting tensor will be of shape (batch_size, max_sequence_length, D) where D is the sum over all embedding dimensions and the value of continuous_projection. 
# - 5. If masking schema is set (needed only for the NextItemPredictionTask training), the masked labels are derived from the sequence of raw item-ids and the sequence of interactions embeddings are processed to mask information about the masked positions.

# ## Define the Transformer Block

# In the next cell, the whole model is build with a few lines of code. 
# Here is a brief explanation of the main classes:  
# - [XLNetConfig](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/config/transformer.py#L261) - We have injected in the HF transformers config classes like `XLNetConfig` the `build()` method, that provides default configuration to Transformer architectures for session-based recommendation. Here we use it to instantiate and configure an XLNET architecture.  
# - [TransformerBlock](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/tf/block/transformer.py#L42) class integrates with HF Transformers, which are made available as a sequence processing module for session-based and sequential-based recommendation models.  
# - [NextItemPredictionTask](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/405e3142f1274b1b0d642f4834ac437f2549cd33/transformers4rec/tf/model/prediction_task.py#82) supports the next-item prediction task. We also support other predictions [tasks](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/tf/model/prediction_task.py), like classification and regression for the whole sequence. 

# In[7]:


# Define XLNetConfig class and set default parameters for HF XLNet config  
transformer_config = tr.XLNetConfig.build(
    d_model=64, n_head=4, n_layer=2, total_seq_length=20
)
# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    [inputs, tr.TransformerBlock(transformer_config, masking=inputs.masking)]
)

# Defines the evaluation top-N metrics and the cut-offs
metrics = [
    NDCGAt(top_ks=[20, 40], labels_onehot=True),  
    RecallAt(top_ks=[20, 40], labels_onehot=True)
]

# link task to body and generate the end-to-end keras model
task = tr.NextItemPredictionTask(weight_tying=True, metrics=metrics)

model = task.to_model(body=body)


# ## Set DataLoader

# We use the NVTabular `KerasSequenceLoader` Dataloader for optimized loading of multiple features from input parquet files. In our experiments, we see a speed-up by 9x of the same training workflow with NVTabular dataloader. NVTabular dataloader’s features are:
# 
# - removing bottleneck of item-by-item dataloading
# - enabling larger than memory dataset by streaming from disk
# - reading data directly into GPU memory and remove CPU-GPU communication
# - preparing batch asynchronously in GPU to avoid CPU-GPU communication
# - supporting commonly used .parquet format
# - easy integration into existing TensorFlow pipelines by using similar API - works with tf.keras models
# 
# You can learn more about this data loader [here](https://nvidia-merlin.github.io/NVTabular/main/training/tensorflow.html) and [here](https://medium.com/nvidia-merlin/training-deep-learning-based-recommender-systems-9x-faster-with-tensorflow-cc5a2572ea49).

# In[8]:


# Define categorical and continuous columns
x_cat_names, x_cont_names = ['category-list_trim', 'item_id-list_trim'], ['timestamp/age_days-list_trim', 'timestamp/weekday/sin-list_trim']

# dictionary representing max sequence length for each column
sparse_features_max = {
    fname: 20
    for fname in x_cat_names + x_cont_names
}

def get_dataloader(paths_or_dataset, batch_size=128):
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


# ## Daily Fine-Tuning: Training over a time window

# Here we do daily fine-tuning meaning that we use the first day to train and second day to evaluate, then we use the second day data to train the model by resuming from the first step, and evaluate on the third day, so on so forth.

# Define the output folder of the processed parquet files

# In[9]:


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/sessions_by_day")


# ### Train the model

# In[10]:


import tensorflow as tf

model.compile(optimizer="adam", run_eagerly=True)


# In[11]:


start_time_window_index = 1
final_time_window_index = 3
# Iterating over days of one week
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data 
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
    
    # Train on day related to time_index 
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    train_loader = get_dataloader(train_paths) 
    losses = model.fit(train_loader, epochs=5)
    model.reset_metrics()
    # Evaluate on the following day
    eval_loader = get_dataloader(eval_paths) 
    eval_metrics = model.evaluate(eval_loader, return_dict=True)
    print('*'*20)
    print("Eval results for day %s are:\t" %time_index_eval)
    print('\n' + '*'*20 + '\n')
    for key in sorted(eval_metrics.keys()):
        print(" %s = %s" % (key, str(eval_metrics[key]))) 


# ## Save the model

# In[12]:


model.save('./tmp/tensorflow')


# ## Reload the model

# In[13]:


model = tf.keras.models.load_model('./tmp/tensorflow')


# In[14]:


batch = next(iter(eval_loader))


# In[15]:


# Generate predictions (logits) from reloaded model with a batch
model(batch[0])


# That's it!  
# You have just trained your session-based recommendation model using Transformers4Rec Tensorflow API.
