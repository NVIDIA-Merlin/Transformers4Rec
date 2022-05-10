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


# # Session-based Recommendation with XLNET

# In this notebook we introduce the [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) library for sequential and session-based recommendation. This notebook uses the PyTorch API, but a TensorFlow API is also available. Transformers4Rec integrates with the popular [HuggingFace’s Transformers](https://github.com/huggingface/transformers) and make it possible to experiment with cutting-edge implementation of the latest NLP Transformer architectures.  
# 
# We demonstrate how to build a session-based recommendation model with the [XLNET](https://arxiv.org/abs/1906.08237) Transformer architecture. The XLNet architecture was designed to leverage the best of both auto-regressive language modeling and auto-encoding with its Permutation Language Modeling training method. In this example we will use XLNET with masked language modeling (MLM) training method, which showed very promising results in the experiments conducted in our [ACM RecSys'21 paper](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf).

# In the previous notebook we went through our ETL pipeline with NVTabular library, and created sequential features to be used in training a session-based recommendation model. In this notebook we will learn:
# 
# - Accelerating data loading of parquet files with multiple features on PyTorch using NVTabular library
# - Training and evaluating a Transformer-based (XLNET-MLM) session-based recommendation model with multiple features

# ## Build a DL model with Transformers4Rec library  

# Transformers4Rec supports multiple input features and provides configurable building blocks that can be easily combined for custom architectures:
# 
# - [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.TabularSequenceFeatures) class that reads from schema and creates an input block. This input module combines different types of features (continuous, categorical & text) to a sequence.
# -  [MaskSequence](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/masking.py) to define masking schema and prepare the masked inputs and labels for the selected LM task.
# -  [TransformerBlock](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.TransformerBlock) class that supports HuggingFace Transformers for session-based and sequential-based recommendation models.
# -  [SequentialBlock](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.SequentialBlock) creates the body by mimicking [torch.nn.sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) class. It is designed to define our model as a sequence of layers.
# -  [Head](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.Head) where we define the prediction task of the model.
# -  [NextItemPredictionTask](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.NextItemPredictionTask) is the class to support next item prediction task.
# - [Trainer](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.Trainer) extends the `Trainer` class from HF transformers and manages the model training and evaluation.
# 
# You can check the [full documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/index.html) of Transformers4Rec if needed.

# Figure 1 illustrates Transformers4Rec meta-architecture and how each module/block interacts with each other.

# ![tf4rec_meta](images/tf4rec_meta2.png)

# ### Imports required libraries

# In[8]:


import os
import glob
import torch 

from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt
from transformers4rec.torch.utils.examples_utils import wipe_memory


# Transformers4Rec library relies on a schema object to automatically build all necessary layers to represent, normalize and aggregate input features. As you can see below, `schema.pb` is a protobuf file that contains metadata including statistics about features such as cardinality, min and max values and also tags features based on their characteristics and dtypes (e.g., categorical, continuous, list, integer).

# ### Manually set the schema 

# In[9]:


from merlin_standard_lib import Schema
SCHEMA_PATH = os.environ.get("INPUT_SCHEMA_PATH", "schema.pb")
schema = Schema().from_proto_text(SCHEMA_PATH)
get_ipython().system('cat $SCHEMA_PATH')


# In[10]:


# You can select a subset of features for training
schema = schema.select_by_name(['item_id-list_trim', 
                                'category-list_trim', 
                                'timestamp/weekday/sin-list_trim',
                                'timestamp/age_days-list_trim'])


# ### Define the sequential input module

# Below we define our `input` block using the `TabularSequenceFeatures` [class](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/features/sequence.py#L97). The `from_schema()` method processes the schema and creates the necessary layers to represent features and aggregate them. It keeps only features tagged as `categorical` and `continuous` and supports data aggregation methods like `concat` and `elementwise-sum` techniques. It also support data augmentation techniques like stochastic swap noise. It outputs an interaction representation after combining all features and also the input mask according to the training task (more on this later).
# 

# The `max_sequence_length` argument defines the maximum sequence length of our sequential input, and if `continuous_projection` argument is set, all numerical features are concatenated and projected by an MLP block so that continuous features are represented by a vector of size defined by user, which is `64` in this example.

# In[11]:


inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="mlm",
)


# The output of the `TabularSequenceFeatures` module is the sequence of interactions embeddings vectors defined in the following steps:
# - 1. Create sequence inputs: If the schema contains non sequential features, expand each feature to a sequence by repeating the value as many as the `max_sequence_length` value.  
# - 2. Get a representation vector of categorical features: Project each sequential categorical feature using the related embedding table. The resulting tensor is of shape (bs, max_sequence_length, embed_dim).
# - 3. Project scalar values if `continuous_projection` is set : Apply an MLP layer with hidden size equal to `continuous_projection` vector size value. The resulting tensor is of shape (batch_size, max_sequence_length, continuous_projection).
# - 4. Aggregate the list of features vectors to represent each interaction in the sequence with one vector: For example, `concat` will concat all vectors based on the last dimension `-1` and the resulting tensor will be of shape (batch_size, max_sequence_length, D) where D is the sum over all embedding dimensions and the value of continuous_projection. 
# - 5. If masking schema is set (needed only for the NextItemPredictionTask training), the masked labels are derived from the sequence of raw item-ids and the sequence of interactions embeddings are processed to mask information about the masked positions.

# ### Define the Transformer Block

# In the next cell, the whole model is build with a few lines of code. 
# Here is a brief explanation of the main classes:  
# - [XLNetConfig](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/config/transformer.py#L261) - We have injected in the HF transformers config classes like `XLNetConfig`the `build()` method, that provides default configuration to Transformer architectures for session-based recommendation. Here we use it to instantiate and configure an XLNET architecture.  
# - [TransformerBlock](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/block/transformer.py#L57) class integrates with HF Transformers, which are made available as a sequence processing module for session-based and sequential-based recommendation models.  
# - [NextItemPredictionTask](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/model/prediction_task.py#L110) supports the next-item prediction task. We also support other predictions [tasks](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/model/prediction_task.py), like classification and regression for the whole sequence. 

# In[12]:


# Define XLNetConfig class and set default parameters for HF XLNet config  
transformer_config = tr.XLNetConfig.build(
    d_model=64, n_head=4, n_layer=2, total_seq_length=20
)
# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
)

# Defines the evaluation top-N metrics and the cut-offs
metrics = [NDCGAt(top_ks=[20, 40], labels_onehot=True),  
           RecallAt(top_ks=[20, 40], labels_onehot=True)]

# Define a head related to next item prediction task 
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                              metrics=metrics),
    inputs=inputs,
)

# Get the end-to-end Model class 
model = tr.Model(head)


# Note that we can easily define an RNN-based model inside the `SequentialBlock` instead of a Transformer-based model. You can explore this [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) for a GRU-based model example.

# ### Train the model 

# We use the NVTabular PyTorch Dataloader for optimized loading of multiple features from input parquet files. You can learn more about this data loader [here](https://nvidia-merlin.github.io/NVTabular/main/training/pytorch.html).

# ### **Set Training arguments**

# In[13]:


from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer
# Set hyperparameters for training 
train_args = T4RecTrainingArguments(data_loader_engine='nvtabular', 
                                    dataloader_drop_last = True,
                                    report_to = [], 
                                    gradient_accumulation_steps = 1,
                                    per_device_train_batch_size = 256, 
                                    per_device_eval_batch_size = 32,
                                    output_dir = "./tmp", 
                                    learning_rate=0.0005,
                                    lr_scheduler_type='cosine', 
                                    learning_rate_num_cosine_cycles_by_epoch=1.5,
                                    num_train_epochs=5,
                                    max_sequence_length=20, 
                                    no_cuda=False)


# Note that we add an argument `data_loader_engine='nvtabular'` to automatically load the features needed for training using the schema. The default value is nvtabular for optimized GPU-based data-loading. Optionally a PyarrowDataLoader (pyarrow) can also be used as a basic option, but it is slower and works only for small datasets, as the full data is loaded to CPU memory.

# ## Daily Fine-Tuning: Training over a time window

# Here we do daily fine-tuning meaning that we use the first day to train and second day to evaluate, then we use the second day data to train the model by resuming from the first step, and evaluate on the third day, so on so forth.

# We have extended the HuggingFace transformers `Trainer` class (PyTorch only) to support evaluation of RecSys metrics. In this example, the evaluation of the session-based recommendation model is performed using traditional Top-N ranking metrics such as Normalized Discounted Cumulative Gain (NDCG@20) and Hit Rate (HR@20). NDCG accounts for rank of the relevant item in the recommendation list and is a more fine-grained metric than HR, which only verifies whether the relevant item is among the top-n items. HR@n is equivalent to Recall@n when there is only one relevant item in the recommendation list.

# In[14]:


# Instantiate the T4Rec Trainer, which manages training and evaluation for the PyTorch API
trainer = Trainer(
    model=model,
    args=train_args,
    schema=schema,
    compute_metrics=True,
)


# - Define the output folder of the processed parquet files

# In[15]:


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/sessions_by_day")


# In[16]:


start_time_window_index = 1
final_time_window_index = 7
#Iterating over days of one week
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data 
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
    print(train_paths)
    
    # Train on day related to time_index 
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    trainer.train_dataset_or_path = train_paths
    trainer.reset_lr_scheduler()
    trainer.train()
    trainer.state.global_step +=1
    print('finished')
    
    # Evaluate on the following day
    trainer.eval_dataset_or_path = eval_paths
    train_metrics = trainer.evaluate(metric_key_prefix='eval')
    print('*'*20)
    print("Eval results for day %s are:\t" %time_index_eval)
    print('\n' + '*'*20 + '\n')
    for key in sorted(train_metrics.keys()):
        print(" %s = %s" % (key, str(train_metrics[key]))) 
    wipe_memory()


# ### Saves the model

# In[17]:


trainer._save_model_and_checkpoint(save_model_class=True)


# ### Reloads the model

# In[18]:


trainer.load_model_trainer_states_from_checkpoint('./tmp/checkpoint-%s'%trainer.state.global_step)


# ### Re-compute eval metrics of validation data

# In[19]:


eval_data_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))


# In[20]:


# set new data from day 7
eval_metrics = trainer.evaluate(eval_dataset=eval_data_paths, metric_key_prefix='eval')
for key in sorted(eval_metrics.keys()):
    print("  %s = %s" % (key, str(eval_metrics[key])))


# That's it!  
# You have just trained your session-based recommendation model using Transformers4Rec.

# Tip: We can easily log and visualize model training and evaluation on [Weights & Biases (W&B)](https://wandb.ai/home), [Tensorboard](https://www.tensorflow.org/tensorboard) and [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger). By default, the HuggingFace transformers `Trainer` (which we extend) uses Weights & Biases (W&B) to log training and evaluation metrics, which provides nice results visualization and comparison between different runs.
