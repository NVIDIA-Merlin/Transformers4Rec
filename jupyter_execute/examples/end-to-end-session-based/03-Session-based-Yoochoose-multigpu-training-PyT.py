#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_transformers4rec_end-to-end-session-based-02-end-to-end-session-based-with-yoochoose-pyt/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Multi-GPU training for session-based recommendations with PyTorch
# 
# This notebook was prepared by using the latest [merlin-pytorch:22.XX](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch/tags) container.

# In the previous two notebooks, we have first created sequential features and saved our processed data frames as parquet files. Then we used these processed parquet files in training a session-based recommendation model with XLNet architecture on a single GPU. We will now expand this exercise to perform a multi-GPU training with the same dataset using PyTorch.
# 
# There are multiple ways to scale a training pipeline to multiple GPUs: 
# 
# - <b>Model Parallel</b>: If the model is too large to fit on a single GPU, the parameters are distributed over multiple GPUs. This is usually the case for the RecSys domain since the embedding tables can be exceptionally large and memory intensive.
# - <b>Data Parallel</b>: Every GPU has a copy of all model parameters and runs the forward/backward pass for its batch. Data parallel is useful when you want to speed-up the training/evaluation of data leveraging multiple GPUs in parallel (as typically data won't fit into GPU memory, that is why models are trained on batches).
# 
# In this example, we demonstrate how to scale a training pipeline to multi-GPU, single node. The goal is to maximize throughput and reduce training time. In that way, models can be trained more frequently and researches can run more experiments in a shorter time duration. 
# 
# This is equivalent to training with a larger batch-size. As we are using more GPUs, we have more computational resources and can achieve higher throughput. In data parallel training, it is often required that all model parameters fit into a single GPU. Every worker (each GPU) has a copy of the model parameters and runs the forward pass on their local batch. The workers synchronize the gradients with each other, which can introduce an overhead.
# 
# <b>Learning objectives</b> 
# - Scaling training pipeline to multiple GPUs
# 
# <b>Prerequisites</b>
# - Run the <b>01-ETL-with-NVTabular.ipynb</b> notebook first to generate the dataset and directories that are also needed by this notebook.

# ### 1. Creating a multi-gpu training python script 

# In this example, we will be using PyTorch, which expects all code including importing libraries, loading data, building the model and training it, to be in a single python script (.py file). We will then spawn torch.distributed.launch, which will distribute the training and evaluation tasks to 2 GPUs with the default <b>DistributedDataParallel</b> configuration.
# 
# The following cell exports all related code to a <b>pyt_trainer.py</b> file to be created in the same working directory as this notebook. The code is structured as follows:
# 
# - importing required libraries
# - specifying and processing command line arguments
# - specifying the schema file to load and filtering the features that will be used in training
# - defining the input module
# - specifying the prediction task
# - defining the XLNet Transformer architecture configuration
# - defining the model object and its training arguments
# - creating a trainer object and running the training loop (over multiple days) with the trainer object
# 
# All of these steps were covered in the previous two notebooks; feel free to visit the two notebooks to refresh the concepts for each step of the script.
# 

# Please note these <b>important points</b> that are relevant to multi-gpu training:
# - <b>specifying multiple GPUs</b>: PyTorch distributed launch environment will recognize that we have two GPUs, since the `--nproc_per_node` arg of `torch.distributed.launch` takes care of assigning one GPU per process and performs the training loop on multiple GPUs (2 in this case) using different batches of the data in a data-parallel fashion.
# - <b>data repartitioning</b>: when training on multiple GPUs, data must be re-partitioned into >1 partitions where the number of partitions must be at least equal to the number of GPUs. The torch utility library in Transformers4Rec does this automatically and outputs a UserWarning message. If you would like to avoid this warning message, you may choose to manually re-partition your data files before you launch the training loop or function. See [this document](https://nvidia-merlin.github.io/Transformers4Rec/main/multi_gpu_train.html#distributeddataparallel) for further information on how to do manual re-partitioning.
# - <b>training and evaluation batch sizes</b>: in the default DistributedDataParallel mode we will be running, keeping the batch size unchanged means each worker will receive the same-size batch despite the fact that you are now using multiple GPUs. If you would like to keep the total batch size constant, you may want to divide the training and evaluation batch sizes by the number of GPUs you are running on, which is expected to reduce time it takes train and evaluate on each batch.

# In[ ]:


get_ipython().run_cell_magic('writefile', "'./pyt_trainer.py'", '\nimport argparse\nimport os\nimport glob\nimport torch \n\nimport cupy\n\nfrom transformers4rec import torch as tr\nfrom transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt\nfrom transformers4rec.torch.utils.examples_utils import wipe_memory\nfrom merlin.schema import Schema\nfrom merlin.io import Dataset\n\n\ncupy.cuda.Device(int(os.environ["LOCAL_RANK"])).use()\n\n# define arguments that can be passed to this python script\nparser = argparse.ArgumentParser(description=\'Hyperparameters for model training\')\nparser.add_argument(\'--path\', type=str, help=\'Directory with training and validation data\')\nparser.add_argument(\'--learning-rate\', type=float, default=0.0005, help=\'Learning rate for training\')\nparser.add_argument(\'--per-device-train-batch-size\', type=int, default=384, help=\'Per device batch size for training\')\nparser.add_argument(\'--per-device-eval-batch-size\', type=int, default=512, help=\'Per device batch size for evaluation\')\nsh_args = parser.parse_args()\n\n# create the schema object by reading the processed train set generated in the previous 01-ETL-with-NVTabular notebook\n\nINPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data")\ntrain = Dataset(os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet"))\nschema = train.schema\n\n# select the subset of features we want to use for training the model by their tags or their names.\nschema = schema.select_by_name(\n   [\'item_id-list\', \'category-list\', \'product_recency_days_log_norm-list\', \'et_dayofweek_sin-list\']\n)\n\nmax_sequence_length, d_model = 20, 320\n# Define input module to process tabular input-features and to prepare masked inputs\ninput_module = tr.TabularSequenceFeatures.from_schema(\n    schema,\n    max_sequence_length=max_sequence_length,\n    continuous_projection=64,\n    aggregation="concat",\n    d_output=d_model,\n    masking="mlm",\n)\n\n# Define Next item prediction-task \nprediction_task = tr.NextItemPredictionTask(weight_tying=True)\n\n# Define the config of the XLNet Transformer architecture\ntransformer_config = tr.XLNetConfig.build(\n    d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length\n)\n\n# Get the end-to-end model \nmodel = transformer_config.to_torch_model(input_module, prediction_task)\n\n# Set training arguments \ntraining_args = tr.trainer.T4RecTrainingArguments(\n            output_dir="./tmp",\n            max_sequence_length=20,\n            data_loader_engine=\'merlin\',\n            num_train_epochs=10, \n            dataloader_drop_last=True,\n            per_device_train_batch_size = sh_args.per_device_train_batch_size,\n            per_device_eval_batch_size = sh_args.per_device_eval_batch_size,\n            learning_rate=sh_args.learning_rate,\n            report_to = [],\n            logging_steps=200,\n        )\n\n# Instantiate the trainer\nrecsys_trainer = tr.Trainer(\n    model=model,\n    args=training_args,\n    schema=schema,\n    compute_metrics=True)\n\n# Set input and output directories\nINPUT_DIR=sh_args.path\nOUTPUT_DIR=sh_args.path\n\nimport time\nstart = time.time()\n\n# main loop for training\nstart_time_window_index = 178\nfinal_time_window_index = 181\n# Iterating over days from 178 to 181\nfor time_index in range(start_time_window_index, final_time_window_index):\n    # Set data \n    time_index_train = time_index\n    time_index_eval = time_index + 1\n    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))\n    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))\n    \n    # Train on day related to time_index \n    print(\'*\'*20)\n    print("Launch training for day %s are:" %time_index)\n    print(\'*\'*20 + \'\\n\')\n    recsys_trainer.train_dataset_or_path = train_paths\n    recsys_trainer.reset_lr_scheduler()\n    recsys_trainer.train()\n    recsys_trainer.state.global_step +=1\n    print(\'finished\')\n\n    # Evaluate on the following day\n    recsys_trainer.eval_dataset_or_path = eval_paths\n    train_metrics = recsys_trainer.evaluate(metric_key_prefix=\'eval\')\n    print(\'*\'*20)\n    print("Eval results for day %s are:\\t" %time_index_eval)\n    print(\'\\n\' + \'*\'*20 + \'\\n\')\n    for key in sorted(train_metrics.keys()):\n        print(" %s = %s" % (key, str(train_metrics[key]))) \n    wipe_memory()\n\nend = time.time()\nprint(\'Total training time:\',end-start)\n')


# ### 2. Executing the multi-gpu training

# You are now ready to execute the python script you created. Run the following shell command which will execute the script and perform data loading, model building and training using 2 GPUs.
# 
# Note that there are four arguments you can pass to your python script, and only the first one is required:
# - <b>path</b>: this argument specifies the directory in which to find the multi-day train and validation files to work on
# - <b>learning rate</b>: you can experiment with different learning rates and see the effect of this hyperparameter when multiple GPUs are used in training (versus 1 GPU). Typically, increasing the learning rate (up to a certain level) as the number of GPUs is increased helps with the accuracy metrics.
# - <b>per device batch size for training</b>: when using multiple GPUs in DistributedDataParallel mode, you may choose to reduce the batch size in order to keep the total batch size constant. This should help reduce training times.
# - <b>per device batch size for evaluation</b>: see above

# In[ ]:


get_ipython().system(' torchrun --nproc_per_node 2 pyt_trainer.py --path "/workspace/data/preproc_sessions_by_day" --learning-rate 0.0005')


# <b>Congratulations!!!</b> You successfully trained your model using 2 GPUs with a `distributed data parallel` approach. If you choose, you may now go back and experiment with some of the hyperparameters (eg. learning rate, batch sizes, number of GPUs) to collect information on various accuracy metrics as well as total training time, to see what fits best into your workflow.

# ## References

# - Multi-GPU data-parallel training using the Trainer class https://nvidia-merlin.github.io/Transformers4Rec/main/multi_gpu_train.html
# 
# - Merlin Transformers4rec: https://github.com/NVIDIA-Merlin/Transformers4Rec
# 
# - Merlin NVTabular: https://github.com/NVIDIA-Merlin/NVTabular/tree/main/nvtabular
# - Merlin Dataloader: https://github.com/NVIDIA-Merlin/dataloader
