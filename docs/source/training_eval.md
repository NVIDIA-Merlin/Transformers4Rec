# Training and Evaluation

By default, Transformers4Rec leverages the NVTabular dataloader for GPU-accelerated loading of preprocessed data that's stored as parquet files. Parquet allows this preprocessed data to be easily structured and queryable. The data in these parquet files are directly loaded to the GPU memory as feature tensors. CPUs are also supported when GPUs are not available.

The following example uses the NVTabular dataloader that is wrapped by the `DataLoader`, which automatically sets some options from the dataset schema. Optionally, the `PyarrowDataLoader` can also be used as a basic option, but it is slower and works only for small datasets since the full data is loaded to CPU memory.

```python
train_loader = transformers4rec.torch.utils.data_utils.DataLoader.from_schema(
        schema,
        paths_or_dataset=train_path,
        batch_size=TrainingArguments.train_batch_size,
        drop_last=True,
        shuffle=True,
    )
```

## PyTorch
To leverage our Transformers4Rec example notebooks that demonstrate how to use Transformers4Rec with PyTorch, refer to [Transformers4Rec Example Notebooks](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples).

### Training 

For PyTorch, the HF transformers `Trainer` class is extended while retaining its `train()` method. Essentially, this means the efficient training implementation from that library is leveraged, which manages half-precision (FP16) and multi-GPU training.

Two [approaches](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) are available for PyTorch multi-GPU training: `DataParallel` and `DistributedDataParallel`. `DataParallel` uses a single process and multiple threads on a single machine. `DistributedDataParallel` is more efficient for assigning separate processes for each GPU. Transformers4Rec supports both training approaches when using the NVTabular Dataloader.

For example:

```python
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

training_args = T4RecTrainingArguments(
            output_dir="./tmp",
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

You can optionally get the dataloaders instantiated by the `Trainer` when the following arguments are provided:

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

### Evaluation
For the Item Prediction head, top-N metrics and ranking metrics commonly used in [Information Retrieval](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) and RecSys are supported for evaluation:

**Top-N metrics**
- **Precision@n** - Computes the percentage of the top-N recommended items, which are relevant (labels).
- **Recall@n** - Computes the percentage of elevant items (labels) that are present among the top-N recommended items.

**Ranking metrics**
- **NDCG@n** - Normalized Discounted Cumulative Gain at cut-off N of the recommendation list.
- **MAP@n** - Mean Average Precision at cut-off N of the recommendation list.

During training, the metrics are computed for each N step for both training and evaluation sets. During evaluation, the metrics are computed for all evaluation batches and averaged. 

You can implement incremental evaluation by splitting your data into time windows such as week, day, or hour. A loop, which trains a model or fine-tunes a pre-trained model, can be used with sessions of time window T. This loop evaluates on sessions of time window T+1.

The following example contains a loop that iterates over the days, trains the model or fine-tunes the model pre-trained in the previous day, evaluates with data of the next day, and assumes daily data is split in folders:

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

## TensorFlow
Training and evaluation with the Tensorflow API is coming soon!
