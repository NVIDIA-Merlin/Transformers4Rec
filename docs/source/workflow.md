
# Training and Deploying a Model
Many examples of data preparation, training and deployment of models using Transformers4Rec are available in our [examples](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples) directory.

## Data loading
Transformers4Rec leverages by default the NVTabular dataloader for GPU-accelerated loading of preprocessed data stored in Parquet format, which is a suitable format for being structured and queryable. 
The data in Parquet files are directly loaded to GPU memory as feature tensors. CPUs are also supported when GPUs are not available.

The following example uses the NVTabular data loader, wrapped by the `DataLoader` that automatically sets some options from the dataset schema. Optionally the `PyarrowDataLoader` can also be used as a basic option, but it is slower and works only for small datasets, as the full data is loaded to CPU memory.

```python
train_loader = transformers4rec.torch.utils.data_utils.DataLoader.from_schema(
        schema,
        paths_or_dataset=train_path,
        batch_size=TrainingArguments.train_batch_size,
        drop_last=True,
        shuffle=True,
    )
```


## PyTorch Training
For PyTorch we extend the HF Transformers `Trainer` class, but keep its `train()` method. That means that we leverage the efficient training implementation from that library, which manages for example half-precision (FP16) and multi-GPU training.

Two [approaches](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) are available for PyTorch multi-GPU training: `DataParallel` and `DistributedDataParallel`. `DataParallel` uses a single process and multiple threads on a single machine. `DistributedDataParallel` is more efficient for assigning separate processes for each GPU. Transformers4Rec supports both training approaches when using the NVTabular Dataloader.

**TODO: Update the previous statement if we cannot have `DistributedDataParallel` working completely with our Data loader.**


```python
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

training_args = T4RecTrainingArguments(
            output_dir="./tmp",
            avg_session_length=20,
            num_train_epochs=3,            
            fp16=True,
        )

recsys_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
)     

recsys_trainer.train()
```


You can optionally get the data loaders instantiated by the `Trainer` when the following arguments are provided.

```python
training_args = T4RecTrainingArguments(
            ...,
            data_loader_engine="nvtabular",
            per_device_train_batch_size=256,
            per_device_eval_batch_size=512,            
        )

# Instantiates the train and eval dataloader
Trainer(
    model=model,
    args=training_args,
    train_dataset_or_path=train_path,
    eval_dataset_or_path=eval_path,   
)     
```            

##  Tensorflow Training

**TODO: Describe the training options for TF**

**TODO: Include code snippets for training with TF**

### Evaluation
For the Item Prediction head, top-N metrics comonly used in [Information Retrieval](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) and RecSys are supported for evaluation:

Top-N metrics
- **Precision@n** - Computes the percentage of the top-N recommended items which are relevant (labels)
- **Recall@n** - Computes the percentage of elevant items (labels) are present among the top-N recommended items

Ranking metrics
- **NDCG@n** - Normalized Discounted Cumulative Gain at cut-off N of the recommendation list
- **MAP@n** - Mean Average Precision at cut-off N of the recommendation list


During training, the metrics are computed each N steps for both training and evaluation sets. During evaluation, the metrics are computed for all evaluation batches and averaged.

#### Incremental Evaluation
You can implement incremental evaluation by splitting your data into time windows (e.g. week, day or hour). Then you can have a loop that trains (or fine-tune a pre-trained model) with session of time window T and evaluates on sessions of time window T+1.

Here is an example which assumes daily data is split in folders. There is a loop that iterates over the days, trains the model (or fine-tunes the model pre-trained in the previous day) and evaluates with data of the next day.

```python
# Iterates over parquet files with daily data
for time_index in range(1, 7):    
    train_paths = glob.glob(f"./data/day-{time_index}/data.parquet")
    eval_paths = glob.glob(f"./data/day-{time_index+1}/data.parquet")
    
    print('Training with day {}'.format(time_index))
    trainer.train_dataset = train_paths
    trainer.reset_lr_scheduler()
    trainer.train()

    print('Evaluating with day {}'.format(time_index+1))
    trainer.eval_dataset = eval_paths
    train_metrics = trainer.evaluate(metric_key_prefix='eval')
    print(train_metrics)
    trainer.wipe_memory()

```


## End-to-end pipeline with Merlin

Transformers4Rec has a first-class integration with NVIDIA Merlin components, to build end-to-end GPU accelerated pipelines for sequential and session-based recommendation.

<div style="text-align: center; margin: 20pt"><img src="_images/pipeline.png" alt="Pipeline for Sequential and Session-based recommendation using NVIDIA Merlin components" style="width:600px;"/><br><figcaption style="font-style: italic;">Fig.3 -cPipeline for Sequential and Session-based recommendation using NVIDIA Merlin components</figcaption></div>

### Integration with NVTabular

[NVTabular](https://github.com/NVIDIA/NVTabular/) is a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems. 

It has some popular [techniques](https://nvidia.github.io/NVTabular/main/api/index.html) to deal with categorical and numerical features like `Categorify`, `Normalize`, `Bucketize`, `TargetEncoding`, `DifferenceLag`, to name a few supported, and also allow for the definition of custom transformations (`LambdaOp`) using cuDF data frame operations.

Usually the input RecSys datasets contains one example per user interaction. For sequential recommendation, the training example is a sequence of user interactions, and for session-based recommendation it is a sequence of session interactions. In practice, each interaction-level feature needs to be converted to a sequence grouped by user/session and their sequence length must match, as each position of the sequence correspond to one interaction. You can see in Fig. 4 how the preprocessed parquet should look like.

<div style="text-align: center; margin: 20pt"><img src="_images/preproc_data_example.png" alt="Example of preprocessed parquet file" style="width:800px;"/><br><figcaption style="font-style: italic;">Example of preprocessed parquet file</figcaption></div>

NVTabular can easily prepare such data with the [Groupby](https://nvidia.github.io/NVTabular/main/api/ops/groupby.html) op, which allows grouping by a categorical column (e.g. user id, session id), sorting by another column (e.g. timestamp) and aggregating other columns as sequences (`list`) or by taking the `first` or `last` element of the sequence, as exemplified below. 

```python
groupby_features = [
    'user_id', 'session_id', 'product_id', 'category_id', 'timestamp'
] >> ops.Groupby(
    groupby_cols=['session_id'],
    sort_cols=['timestamp'],
    aggs={
        'product_id': 'list',
        'category_id': 'list',
        'timestamp': ['first', 'last'],
    },
)
```

#### Outputs

NVTabular outputs parquet files with the preprocessed data. The parquet files can be (Hive) partitioned by a categorical column (e.g. day, company), as in the following example.

```python
nvt_output_path ='./output'
partition_col = ['day']
nvt.Dataset(dataset).to_parquet(nvt_output_path, partition_on=[partition_col])
```

NVTabular also outputs a schema of the parquet columns in Profobuf Text format, e.g. including the cardinality of categorical features, the max squence length for sequential features and tags that can be associated to features (e.g. to indicate what is the item id, what are item and user features, what are categorical or continuous features). You can see [here](../../tests/assets/data_schema/schema.pbtxt) an example of such schema in Protobuf Text format.
P.s. If you don't use NVTabular to preprocess your data, you can generate the Schema via code.  

**TODO: Include code snippet of how to define the Schema manualy using code**

The NVTabular workflow can be saved after `workflow.fit()` is called, so that the same preproc workflow can be applied to new input data, either in batch or online (via integration with Triton Inference Server), described in the next section.

```python
# Instantiates an NVTabular dataset
dataset = nvt.Dataset([os.path.join(INPUT_PATH, "*.parquet")], part_size="100MB")
# Perform a single pass over the dataset to collect columns statistics
workflow.fit(dataset)
# Applies the transform ops to the dataset
new_dataset = workflow.transform(dataset)
# Saves the "fitted" preprocessing workflow
workflow.save(os.path.join(OUTPUT_PATH, "workflow"))
```

### Integration with Triton Inference Server

**TODO: Describe the integration with Triton**
