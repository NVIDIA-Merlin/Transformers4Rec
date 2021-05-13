# Transformers4Rec - How to setup and run

This folder has the code for training and evaluating Transformer-based architectures for sequencial, session-based, and session-aware recommendation.

## Dependencies

We use the follow ingredients to implement RecSys-Transformer pipelines:
- Trainer and Transformer models implementation for PyTorch: [HuggingFace (HF) transformers library](https://huggingface.co/transformers/)
- Data Loaders to read input Parquet files: [NVIDIA NVTabular](https://github.com/NVIDIA/NVTabular/), [PyArrow](https://github.com/apache/arrow/), or [Petastorm](https://petastorm.readthedocs.io)
- RecSys evaluation metrics: [karlhigley's implementation](https://github.com/karlhigley/ranking-metrics-torch)
- Logging: [Weights & Biases (W&B)](https://wandb.ai/), [Tensorboard](https://www.tensorflow.org/tensorboard) and [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger)


## Enviroment Setup
You can choose Conda or Docker to setup your local environment.


### Conda
You might prefer the Conda environment for easy debugging with [Visual Studio Code](https://code.visualstudio.com/).

```
conda create -n env_name python=3.7
conda activate conda

# Run this line only if GPUs are available (to install NVTabular and its dependency RAPIDS)
conda install -c rapidsai -c nvidia -c numba -c conda-forge -c defaults cudf=0.16 nvtabular=0.3

pip install -r requirements.txt
```

#### **Visual Studio Code setup**
To be able to run and debug the pipeline using Visual Studio Code:
- Click in *File / Open Folder*, selecting the root `hf4rec` folder
- Create a hidden folder `.vscode` within that root folder and copy there an example of the VS Code launch config, available in `resources/dev_env/vscode/launch.json`
- Now, in the Run tab on the left VS Code bar, you can select one of the example configurations (defined in `launch.json`) and run/debug the code.

### Docker

Here are the commmands to create a local container for development. 

Available containers:
- `containers/Dockerfile.dev_nvt` (recommended, GPU-only) - Supports the GPU-accelerated [NVTabular](https://github.com/NVIDIA/NVTabular/) dataloader (`--data_loader_engine nvtabular`) for parquet files. 
- `containers/Dockerfile.dev` (GPU or CPU) - Supports only [PyArrow](https://github.com/apache/arrow/) (`--data_loader_engine pyarrow`) or [Petastorm](https://petastorm.readthedocs.io) (`--data_loader_engine petastorm`) data loaders. Will use GPU for training if available, otherwise CPU.


*Example commands:*

```bash
#Build the image
docker build --no-cache -t hf4rec_dev -f container/Dockerfile.dev_nvt .

#Run the image in the interactive mode
docker run --gpus all --ipc=host -it --rm --cap-add=SYS_ADMIN --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  \
 -p 6006:6006 -p 8888:8888 -v ~/projects/rapidsai/hf4rec:/workspace -v ~/dataset/:/data --workdir /workspace/ hf4rec_dev /bin/bash 
```

For more advanced instructions with Docker check the [`containers`](../containers/README.md) folder.



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

### 1. **Get the preprocessed datasets**

You can find information on the example available datasets, and instructions on their preprocessing or download of preprocessed datasets [here](../datasets/README.md).


### 2. **Run the training & evaluation script**

The following command runs the pipeline using `torch.nn.DataParallel()` (default) to make the pipeline data parallel. In this setting, the dataloader loads a batch from a dataset and splits it to the different GPUs using multi-threading that processes those chuncks of data in parallel.  

```bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --data_path "/DATA_PATH/" \
    --feature_config datasets/ecommerce_rees46/config/features/session_based_features_pid.yaml \
    --data_loader_engine nvtabular \
    --workers_count 2 \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
    --start_time_window_index 1 \
    --final_time_window_index 15 \
    --time_window_folder_pad_digits 4 \
    --model_type gpt2 \
    --loss_type cross_entropy \
    --similarity_type concat_mlp \
    --tf_out_activation tanh \
    --inp_merge mlp \
    --hidden_act gelu_new \
    --dataloader_drop_last \
    --compute_metrics_each_n_steps 1 \
    --session_seq_length_max 20 \
    --eval_on_last_item_seq_only \
    --num_train_epochs 10 \
    --per_device_train_batch_size 192 \
    --per_device_eval_batch_size 128 \
    --learning_rate 0.00014969647714359603 \
    --learning_rate_schedule linear_with_warmup \
    --learning_rate_warmup_steps 0 \
    --mf_constrained_embeddings \
    --dropout 0.1 \
    --weight_decay 6.211639773976265e-05 \
    --d_model 320 \
    --n_layer 1 \
    --n_head 2
```


**Notes:**
- The following Transformer architectures do not work when using multiple GPUs with `torch.nn.DataParallel()` (because they use `model.parameters()`, which is not available with `DataParallel()`). For this models, you should set CUDA_VISIBLE_DEVICES to just a single GPU: 
  - **TransfoXLModel()**



### **Notes on command line arguments**

#### **Data loading**
- The path provided in the `--data_path` argument must contain a subfolder for each time window, named as contiguous numbers (e.g. **0001**, **0002**, **0003**). Each subfolder must contain three files - **train.parquet**, **valid.parquet** and **test.parquet** -- with sessions of that time window. The time window unit can be a day, an hour, or any other time period.
- Use the `--start_time_window_index` and `--final_time_window_index` to define the range of time windows that will be used for incremental training and evaluation. The subfolder names must be integers with padded zeros on the left, up to the number of digits defined in `--time_window_folder_pad_digits`. 
- For each day, the model will train for as many epochs as defined in `--num_train_epochs`, before moving to the next day. To train for only a few steps each day before moving to evaluation for the next day, use the argument `--max_steps`.
- If using NVIDIA GPU(s) with Volta or later architectures, enable `--fp16` for Automatic Mixed Precision (AMP), leading to faster execution.
- when GPUs are available, use `--data_loader_engine nvtabular` for faster data loading. If using `--data_loader_engine pyarrow` or `--data_loader_engine petastorm`, you should also set `--workers_count` to a number greater than 0, for async data loading with prefetching on CPU (much better performance than `--workers_count 0`).
- The `--dataloader_drop_last` argument ignores the last batch of the dataset, which is usually incomplete (less number of examples than the batch size). This is important when using more than 1 GPU, to avoid parallelization issues with `torch.nn.DataParallel(model)`, because if GPUs receive a batch with a different size, it causes errors when aggregating results.
- The `--feature_config` argument takes the path of a JSON config file with the input features definition corresponding to the dataset (parquet files), including the cardinality of categorical features, shared embeddings configuration and a flag to set which feature is the label for the next-click prediction (usually the item id). You can find more infor and see some examples of features config files in the [datasets](../dataset/README.md) folder.
- For more info about the arguments, check the args help comments in [recsys_args.py](recsys_args.py).

#### **Architectures and Tasks**
- Some Transformer architecures follow the [Causal Language Modeling (LM)](https://huggingface.co/transformers/task_summary.html#causal-language-modeling) task, which predicts the next "word" (i.e., next item in our case) using only the items before in the sequence (unidirectional), like GPT-2 (`--model_type gpt2`) and TransformerXL (`--model_type transfoxl`). Other task is the [Masked Language Modeling (LM)](https://huggingface.co/transformers/task_summary.html#masked-language-modeling), in which items in the sequence are randomly masked and the unmasked items are used for prediction of the masked items, even those on the right side (i.e. future interactions). Currently only `--model_type xlnet` supports Masked LM. The default masking task for this framework is Causal LM, and for Masked LM you need to set the bool arg`--mlm` and `--mlm_probability 0.5` with a masking probability.  

#### **Session-based and session-aware recommendation**
- This framework supports session-based recommendation (default) and session-aware recommendation (`--session_aware`) tasks. For both tasks, you need to specify the maximum length of the session sequence features for padding using `--session_seq_length_max`. For session-aware recommendation, you also need to provide, for every sequential feature you have for the current session, another sequential feature corresponding to users past interactions (before the current session). Those additional past feature names might be equal to the session feature name prefixed as specified by the `--session_aware_features_prefix` argument (e.g. "**product_id_seq**" for a feature of the current session and "**bef_product_id_seq**" for the corresponding past interactions feature, if `--session_aware_features_prefix "bef_"`), so that features of the current session and past interactions can be concatenated before feeding to the transformers layers. The length of the past sequences for padding must be defined using `--session_aware_past_seq_length_max`.

#### **Training, Evaluating and Logging**
- When using the Masked LM task, any item of the sequence can be masked during training, but during evaluation it is ensured that only the last item is masked (so that future interactions are not leaked). With Causal LM task, we never run into this issue because the model can only use past interactions for predicting the next item, thus, is possible to evaluate for every next-item in the sequence. But to be able to compare metrics between models using the Causal LM and Masked LM tasks, you need to set `--eval_on_last_item_seq_only` so that metrics are only computed for the last item of the sequence.
- For every execution of the pipeline, it is logged a new experiment to Weights & Biases online service. The experiments can be organized in W&B by setting a common value to `--experiments_group`, so that they can be filtered later. For local debugging, if you want to disable online logging when debugging, you can set the following environment variable: `WANDB_MODE=dryrun`.

#### **Best practices**
From some extensive experiments on the example datasets we found out that:
- After some hypertuning sessions we found that, differently from NLP, usually small transformer architectures (with few layers and head) provide the best evaluation accuracy.
- In general, models trained with Masked LM (e.g. XLNet) are more accurate than those trained with Casusal LM.
- The usage of the [tying embeddings](https://arxiv.org/pdf/1611.01462.pdf) technique, which shares the weights of the item id embedding table weights in the last layer of the network that provides the predictions over all items, considerably improves the accuracy. This option is available by the `--mf_constrained_embeddings` argument and we recommend enabling it by default.
- Next-click prediciton can be treated as a classification problem, where the labels are the next items clicked by the user. Those labels might not be very reliable as users browsing is a stochastic process. We found that using Label Smoothing (e.g. `--label smoothing 0.8`) can improve both train and evaluation accuracy.
- Providing additional features (i.e., other than the item id) usually improves the model accuracy. The categorical features are encoded as embeddings, with their size defined by a function based on their cardinality (obtained from the features config file) and on the `--embedding_dim_from_cardinality_multiplier` argument. The best way we have found to represent numerical features was the `Soft One-Hot Encoding Embedding` technique (Section 3.2 of this [paper](https://arxiv.org/pdf/1708.00065.pdf)), which you can use by setting the `----numeric_features_soft_one_hot_encoding_num_embeddings` and `--numeric_features_project_to_embedding_dim` arguments.

### 3. **Check results**
The following output files will be saved in the path defined by the `--output_dir` argument:
- `eval_results_avg_over_days.txt` - Final metrics Averaged Over Time (AOT) in human-readable format
- `train_results_over_time.txt` and `eval_results_over_time.txt` - Metric results for each training and evaluation day in human-readable format.
- `eval_train_results.csv` - Metric results for each training and evaluation day in a machine-readable format (CSV)
- `log.json` - Output metrics in the DLLoger format. Used to retrieve the final metrics by an NVIDIA internal hypertuning tool (AutoBench).
- `pred_logs/` - Folder created with logged predictions and metrics during evaluation when `--log_predictions` is enabled
- `attention_weights/` - Folder created with logged Transformer''s attention weights during evaluation when `--log_attention_weights` is enabled

Additionally, the pipeline generates the following folders in the current folder:
- `runs/` - Tensorboard event logs, which you can visualize with ```tensorboard --logdir runs/```
- `wandb/` - Weights & Biases local logs, which are automatically sync with the W&B online service. 

You can also check plots of the timeseries and metrics in the [Weights & Biases (W&B) app](https://wandb.ai/)

## Code overview
- `recsys_main.py`: Main script to run the train and evaluation pipeline of the Transformers4Rec framework
- `recsys_models.py`: Definition of various sequence models (Huggingface Transformers and PyTorch GRU)
- `recsys_meta_model.py`: RecSys module that generates interaction embeddings from the available input features and feeds to Transformer blocks. Then it takes the output of the last Transformer block and predicts the scores for all items.
- `recsys_trainer.py`: Extends Huggingface's Trainer class to enable customized dataset in training and evaluation loops.
- `recsys_data.py`: Data loading functinos
- `recsys_args.py`: Defines the command line arguments for the pipeline.
- `recsys_outputs.py`: Manages the logging and output files with metrics results, predicted items and attention weights
- `evaluation/recsys_metrics.py`: Manages streaming recommendation accuracy metrics: NDCG@k, MAP@k, Recall@k, Precision@k
- `baselines/recsys_baselines_main.py`: Main script to run the pipeline for the baseline methods (e.g. V-SkNN).