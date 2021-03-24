# NVIDIA Transformers4RecSys - How to setup and run

The main code for preprocessing, training and evaluating Transformer-based architectures for sequencial, session-based, and session-aware recommendation is in this folder

## Dependencies

The current code uses following ingredients to implement RecSys-Transformer models:
- Trainer and Transformer models implementation for PyTorch: [HuggingFace (HF) transformers library](https://huggingface.co/transformers/)
- Data Loaders to read input Parquet files: [NVIDIA NVTabular](https://github.com/NVIDIA/NVTabular/), [PyArrow](https://github.com/apache/arrow/), or [Petastorm](https://petastorm.readthedocs.io)
- RecSys evaluation metrics: [karlhigley's implementation](https://github.com/karlhigley/ranking-metrics-torch)
- Logging: [Weights & Biases (W&B)](https://wandb.ai/), [Tensorboard](https://www.tensorflow.org/tensorboard) and [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger)


## Enviroment Setup
You can choose Conda or Docker to setup your local environment.


### Conda
You might prefer the Conda environment for easy debugging with [Visual Studio Code](https://code.visualstudio.com/).

```
conda create -n env_name python=3.6
conda activate conda

# Run this line only if GPUs are available (to install NVTabular and its dependency RAPIDS)
conda install -c rapidsai -c nvidia -c numba -c conda-forge -c defaults cudf=0.16 nvtabular=0.3

pip install -r requirements.txt
```

#### **Visual Studio Code setup**
To be able to run and debug the pipeline using Visual Studio Code:
- Click in *File / Open Folder*, selecting the `recsys/transformers4recsys` folder
- Create a hidden folder `.vscode` within that root folder and copy there an example of the VS Code launch config, available in `resources/dev_env/vscode/launch.json`
- Now, in the Run tab on the left VS Code bar, you can select one of the example configurations (defined in `launch.json`) and run/debug the code.

### Docker

Here are the commmands to create a local container for development. 

Available containers:
- `container/Dockerfile.dev_nvt` (recommended, GPU-only) - Supports the GPU-accelerated [NVTabular](https://github.com/NVIDIA/NVTabular/) dataloader (`--data_loader_engine nvtabular`) for parquet files. 
- `container/Dockerfile.dev` (GPU or CPU) - Supports only [PyArrow](https://github.com/apache/arrow/) (`--data_loader_engine pyarrow`) or [Petastorm](https://petastorm.readthedocs.io) (`--data_loader_engine petastorm`) data loaders. Will use GPU if available, otherwise CPU.


*Example commands:*

```bash
#Build the image
docker build --no-cache -t transf4rec_dev -f container/Dockerfile.dev_nvt .

#Run the image in the interactive mode
docker run --gpus all --ipc=host -it --rm --cap-add=SYS_ADMIN --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  \
 -p 6006:6006 -p 8888:8888 -v ~/projects/nvidia/recsys:/workspace -v ~/dataset/:/data --workdir /workspace/transformers4recsys/codes transf4rec_dev /bin/bash 
```

For more advanced instructions with Docker check the `containers` folder.



### Weights & Biases logging setup

By default, Huggingface uses [Weights & Biases (W&B)](https://wandb.ai/) to log training and evaluation metrics.
It allows a nice management of experiments, including config logging, and provides plots with the evolution of losses and metrics over time.

1. Create account if you don't have and obtain API key: https://www.wandb.com

2. Run the following command and follow the instructions to get an API key.
```bash
wandb login
```

After you get the API key, you can set it as an environment variable with

```bash
export WANDB_API_KEY=<YOUR API KEY HERE>
```

## Run training and evaluation scripts

### 1. Get the preprocessed e-commerce large dataset

You can download a sample of the preprocessed ecommerce dataset from Google Drive by running the followin script:

```bash
bash script/dowload_dataset_from_gdrive.bash
```

The pipeline follows a incremental training and evaluation protocol, that emulates a production setting where the model is fine-tuned with only the most recent data (e.g. daily or hourly), and evaluated on predictions for the next time period unit (e.g. day or hour). 
The pipeline expects three parquet files for each time period unit (e.g. date or hour), for example:

- session_start_date=2019-10-01-train.parquet
- session_start_date=2019-10-01-valid.parquet
- session_start_date=2019-10-01-test.parquet

### 2. **Run the training & evaluation script**

#### **DataParallel (default)**

The following command runs the pipeline using `torch.nn.DataParallel()` (default) to make the pipeline data parallel. In this setting, the dataloader loads a batch from a dataset and splits it to the different GPUs using multi-threading that processes those chuncks of data in parallel.  

```bash
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false python recsys_main.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset_path/ --start_date 2019-10-01 --end_date 2019-10-15 --data_loader_engine nvtabular --per_device_train_batch_size 320 --per_device_eval_batch_size 512 --model_type gpt2 --loss_type cross_entropy --logging_steps 10 --d_model 256 --n_layer 2 --n_head 8 --dropout 0.1 --learning_rate 0.001 --similarity_type concat_mlp --num_train_epochs 1 --all_rescale_factor 1 --neg_rescale_factor 0 --feature_config ../datasets/ecommerce-large/config/features/session_based_features_pid.yaml --inp_merge mlp --tf_out_activation tanh --experiments_group local_test --weight_decay 1.3e-05 --learning_rate_schedule constant_with_warmup --learning_rate_warmup_steps 0 --learning_rate_num_cosine_cycles 1.25 --dataloader_drop_last --compute_metrics_each_n_steps 1 --hidden_act gelu_new --save_steps 0 --eval_on_last_item_seq_only --fp16 --overwrite_output_dir --session_seq_length_max 20 --predict_top_k 1000 --eval_accumulation_steps 10
```

**Notes:**
- The following Transformer architectures do not work when using multiple GPUs with `torch.nn.DataParallel()` (because they use `model.parameters()`, which is not available with `DataParallel()`): 
  - **TransfoXLModel()**

#### **DistributedDataParallel**


Another option is to run the pipeline using `torch.nn.parallel.DistributedDataParallel()`, which is more optimized than `torch.nn.DataParallel()` for using multi-processing instead of multi-threading (more info [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)). The `DistributedDataParallel()` can be used by providing `--model_parallel` argument and launch it using `torch.distributed.launch`, as shown in the following command (set `--nproc_per_node` with the number of available GPUs):

```bash
python -m torch.distributed.launch --nproc_per_node 2 recsys_main.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset_path/ --start_date 2019-10-01 --end_date 2019-10-15 --data_loader_engine nvtabular --per_device_train_batch_size 320 --per_device_eval_batch_size 512 --model_type gpt2 --loss_type cross_entropy --logging_steps 10 --d_model 256 --n_layer 2 --n_head 8 --dropout 0.1 --learning_rate 0.001 --similarity_type concat_mlp --num_train_epochs 1 --all_rescale_factor 1 --neg_rescale_factor 0 --feature_config ../datasets/ecommerce-large/config/features/session_based_features_pid.yaml --inp_merge mlp --tf_out_activation tanh --experiments_group local_test --weight_decay 1.3e-05 --learning_rate_schedule constant_with_warmup --learning_rate_warmup_steps 0 --learning_rate_num_cosine_cycles 1.25 --dataloader_drop_last --compute_metrics_each_n_steps 1 --hidden_act gelu_new --save_steps 0 --eval_on_last_item_seq_only --fp16 --overwrite_output_dir --session_seq_length_max 20 --predict_top_k 1000 --eval_accumulation_steps 10 --model_parallel
```

**Notes:**
- When using `torch.nn.parallel.DistributedDataParallel()`, currently all GPUs receive the exact same data (batch), which is not interesting for training. It needs further investigation on how to enable shuffling for NVTabular and PyArrow data loaders to circumvent that problem.



### **Notes on command line arguments**

#### **Data loading**
- The date range considered for the incremental training and evaluation loop are defined using `--start_date` and `--end_date`. For each day, the model will train for as many epochs as defined in `--num_train_epochs`, before moving to the next day. To train for only a few steps each day before moving to evaluation for the next day, use the argument `--max_steps`.
- If using NVIDIA GPU(s) with Volta or later architectures, enable `--fp16` for Automatic Mixed Precision (AMP), leading to faster processing.
- If using `--data_loader_engine pyarrow` or `--data_loader_engine petastorm`, you should also set `--workers_count` to a number greater than 0, for async data loading with prefetching on CPU (much better performance than `--workers_count 0`).
- The `--feature_config` argument takes the path of a JSON config file with the input features definition corresponding to the dataset (parquet files), including the cardinality of categorical features, shared embeddings configuration and a flag to set which feature is the label for the next-click prediction (usually the item id). You can see some examples under the `datasets` folder.
- The `--dataloader_drop_last` argument ignores the last batch of the dataset, which is usually incomplete (less number of examples than the batch size). This is important when using more than 1 GPU, to avoid parallelization issues with `torch.nn.DataParallel(model)`, because if GPUs receive a batch with a different size, it causes errors when aggregating results.
- For more info about the arguments, check the args help comments in `recsys_args.py`.

#### **Architectures and Tasks**
- Some Transformer architecures follow the [Causal Language Modeling (LM)](https://huggingface.co/transformers/task_summary.html#causal-language-modeling) task, which predicts the next "word" (i.e., next item in our case) using only the items before in the sequence (unidirectional), like GPT-2 (`--model_type gpt2`). Other task is the [Masked Language Modeling (LM)](https://huggingface.co/transformers/task_summary.html#masked-language-modeling), like BERT or XLNet `--model_type xlnet`, in which items in the sequence are randomly masked and the unmasked items are used for prediction of the masked items, even those on the right side (i.e. future interactions). The default masking task for this framework is Causal LM, and for Masked LM you need to set the bool arg`--mlm` and `--mlm_probability 0.5` with a masking probability.  
- This framework support session-based recommendation (default) and session-aware recommendation (`--session_aware`) tasks. For both tasks, you need to specify the maximum length of the session sequence features for padding using `--session_seq_length_max`. For session-aware recommnedation, you also need to provide, for every sequential feature you have for the current session, another sequential feature corresponding to users past interactions (before the current session). Those additional past feature names might be equal to the session feature name prefixed as specified by the `--session_aware_features_prefix` argument (e.g. "**product_id_seq**" for a feature of the current session and "**bef_product_id_seq**" for the corresponding past interactions feature, if `--session_aware_features_prefix "bef_"`), so that features of the current session and past interactions can be concatenated before feeding to the transformers layers. The length of the past sequences for padding must be defined using `--session_aware_past_seq_length_max`.

#### **Training, Evaluating and Logging**
- When using the Masked LM task, any item of the sequence can be masked during training, but during evaluation it is ensured that only the last item is masked (so that future interactions are not leaked). With Causal LM task, we never run into this issue because the model can only use past interactions for predicting the next item, thus, is possible to evaluate for every next-item in the sequence. But to be able to compare metrics between models using the Causal LM and Masked LM tasks, you need to set `--eval_on_last_item_seq_only` so that metrics are only computed for the last item of the sequence.
- For every execution of the pipeline, it is logged a new experiment to Weights & Biases online service. The experiments can be organized in W&B by setting a common `--experiments_group`, so that they can be filtered later. For local debugging, if you want to disable online logging when debugging, you can set the following environment variable: `WANDB_MODE=dryrun`.



### 3. **Check results**
The following output files will be saved in the path defined by the `--output_dir` argument:
- `eval_results_avg_over_days.txt` - Final metrics Averaged Over Days (AOD) in human-readable format
- `train_results_dates.txt` and `train_results_dates.txt` - Metric results for each training and evaluation day in human-readable format
- `eval_train_results_dates.csv` - Metric results for each training and evaluation day in a machine-readable format (CSV)
- `log.json` - Output metrics in the DLLoger format. Used to retrieve the final metrics by an NVIDIA internal hypertuning tool (AutoBench).
- `pred_logs/` - Folder created with logged predictions and metrics during evaluation when `--log_predictions` is enabled
- `attention_weights/` - Folder created with logged Transformer''s attention weights during evaluation when `--log_attention_weights` is enabled

Additionally, the pipeline generates the following folders in the current folder:
- `runs/` - Tensorboard event logs, which you can visualize with ```tensorboard --logdir runs/```
- `wandb/` - Weights & Biases local logs, which are automatically sync to update W&B online service. 

You can also check plots of the timeseries and metrics in the [Weights & Biases (W&B) app](https://wandb.ai/)

## Code overview
- `recsys_main.py`: Main training and evaluation pipeline
- `recsys_models.py`: Definition of various sequence models (Huggingface Transformers and PyTorch GRU,RNN,LSTMs)
- `recsys_meta_model.py`: RecSys wrapper model that gets embeddings for discrete input tokens and merges multiple sequences of input features. Then, it runs forward function of defined sequence model and computes loss.
- `recsys_trainer.py`: Extends Huggingface's Trainer class to enable customized dataset in training and evaluation loops.
- `recsys_data.py`: setup for dataloader and necessaties to read Parquet files
- `recsys_metrics.py`: defines various evaluation metric computation function (e.g. Recall@k, Precision@k, NDCG, etc.) Any additional metric computation functions can be added and executed here.
- `recsys_args.py`: defines input args for the code.

