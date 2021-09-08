# Core Features

## The relationship between NLP and RecSys

Over the past decade there has been a trend toward leveraging and adapting approaches proposed by Natural Language Processing (NLP) research like Word2Vec, GRU, and Attention for recommender systems (RecSys). The phenomena is especially noticeable for sequential and session-based recommendation where the sequential processing of users interactions is analogous to the language modeling (LM) task and many key RecSys architectures have been adapted from NLP, like GRU4Rec -- the seminal Recurrent Neural Network (RNN)-based architecture for session-based recommendation.

More recently, Transformer architectures have become the dominant technique over convolutional and recurrent neural networks for language modeling tasks. Because of their efficient parallel training, these architectures scale well with training data and model size, and are effective at modeling long-range sequences. 

Transformers have similarly been applied to sequential recommendation in architectures like [SASRec](https://arxiv.org/abs/1808.09781), [BERT4Rec](https://arxiv.org/abs/1904.06690) and [BST](https://arxiv.org/pdf/1905.06874.pdf%C2%A0), providing higher accuracy than architectures based on CNN and RNNs, as can be seen in their reported experiments and also in our [ACM RecSys'21 paper](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf). 

You can read more about this relationship between NLP and RecSys and the evolution of the architectures for sequential and session-based recommendation towards Transformers in our [paper](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf) too.

<div style="text-align: center; margin: 20pt"><img src="_images/nlp_x_recsys.png" alt="A timeline illustrating the influence of NLP research in Recommender Systems" style="width:800px;"/><br><figcaption style="font-style: italic;">Fig. 1 - A timeline illustrating the influence of NLP research in Recommender Systems, from the <a href="https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf)">Transformers4Rec paper</a></figcaption></div>




## Integration with HuggingFace Transformers

Transformers4Rec integrates with the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) library, allowing RecSys researchers and practitioners to easily experiment with the latest and state-of-the-art NLP Transformer architectures for sequential and session-based recommendation tasks and deploy those models into production.

The HF Transformers was *"established with the goal of opening up advancements in NLP to the wider machine learning community"*. It has become very popular among NLP researchers and practitioners (more than 900 contributors), providing standardized implementations of the state-of-the-art Transformer architectures (more than 68 and counting) produced by the research community, often within days or weeks of their publication. 

HF Transformers is designed for both research and production. Models are composed of three building blocks: (a) a tokenizer, which converts raw text to sparse index encodings; (b) a transformer architecture; and (c) a head for NLP tasks, like Text Classification, Generation, Sentiment Analysis, Translation, Summarization, among others. 

In Transformers4Rec we leverage from HF Transformers only the transformer architectures building block (b) and their configuration classes. Transformers4Rec provides additional blocks necessary for recommendation, e.g., input features normalization and aggregation, and heads for recommendation and sequence classification/prediction. We also extend their `Trainer` class to allow for the evaluation with RecSys metrics.


## Flexibility in Model Architecture
Transformers4Rec provides modularized building blocks that can be combined with plain PyTorch modules and Keras layers. This provides a great flexibility in the model definition, as you can use the blocks to build custom architectures, e.g., with multiple towers, multiple heads and losses (multi-task).

In Fig. 2, we provide a reference architecture for next-item prediction with Transformers, that can be used for both sequential and session-based recommendation. We can divide that reference architecture in four conceptual layers, described next.

<div style="text-align: center; margin: 20pt"><img src="_images/transformers4rec_metaarchitecture.png" alt="Transformers4Rec meta-architecture" style="width:600px;"/><br><figcaption style="font-style: italic;">Fig. 2 - Transformers4Rec meta-architecture</figcaption></div>


### Features Processing 
Here the input features are processed. Categorical features are represented by embeddings. Numerical features can be represented as a scalar, projected by a fully-connected (FC) layer to multiple dimensions, or represented as a weighted average of embeddings by the technique Soft One-Hot embeddings (more info in our [paper online appendix](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/Appendices/Appendix_A-Techniques_used_in_Transformers4Rec_Meta-Architecture.md)).

The features are optionally normalized (with layer normalization) and then aggregated. The current feature aggregation options are:
- **Concat** - Concatenation of the features
- **Element-wise sum** - Features are summed. For that, all features must have the same dimension, i.e. categorical embeddings must have the same dim and continuous features are projected to that dim.
- **Element-wise sum & item multiplication** - Similar to *Element-wise sum*, as all features are summed. except for the item id embedding, which is multiplied by the other features sum. The aggregation formula is available in our [paper](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf).

The core class of this module is the `TabularSequenceFeatures`, which is responsible to process and aggregate all features and outputs *interaction embeddings*. It can be instantiated from a dataset schema (`from_schema()`), which directly creates all the necessary layers to represent categorical and continuous features. In addition, it has options to aggregate the sequential features, to prepare masked labels, depending on the chosen sequence masking approach (see next section)).

```python
from transformers4rec.torch import TabularSequenceFeatures
tabular_inputs = TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        d_output=100,
        aggregation="sequential-concat",
        masking="clm"
    )
```


### Sequence Masking
Transformer architectures can be trained in different ways. Depending of the training method, there is a specific masking schema. The masking schema sets the items to be predicted (labels) and mask (hide) some positions of the sequence that cannot be used by the Transformer layers for prediction. Currently supports the following training approaches, inspired by NLP:

- **Causal LM (`masking="clm"`)** - Predicts the next item based on past positions of the sequence. Future positions are masked.
- **Masked LM (`masking="mlm"`)** - Randomly select some positions of the sequence to be predicted, which are masked. The Transformer layer is allowed to use positions on the right (future information) during training. During inference, all past items are visible for the Transformer layer, which tries to predict the next item.
- **Permutation LM (`masking="plm"`)** - Uses a permutation factorization at the level of the self-attention layer to define the accessible bidirectional context
- **Replacement Token Detection (`masking="rtd"`)** - Uses MLM to randomly select some items, but replaces them by random tokens. Then, a discriminator model (that can share the weights with the generator or not), is asked to classify whether the item at each position belongs or not to the original sequence. The generator-discriminator architecture was jointly trained using Masked LM and RTD tasks. 


### Sequence Processing
Processes the input sequences of interaction vectors. It can the `RNNBlock` for RNN architectures (e.g. LSTM or GRU) or the `TransformerBlock` for supported Transformer architectures.

In the following example, a `SequentialBlock` module is defined connecting the output of the `TabularSequenceFeatures` (inputs), with a MLP projection to 64 dim (to match the Transformer `d_model`) with an XLNet transformer block with 2 layers (4 heads each).


```python
from transformers4rec.config import transformer
from transformers4rec.torch import MLPBlock, SequentialBlock, TransformerBlock

# Configures the XLNet Transformer architecture
transformer_config = transformer.XLNetConfig.build(
    d_model=64, n_head=4, n_layer=2, total_seq_length=20
)

# Defines the model body including: inputs, masking, projection and transformer block.
model_body = SequentialBlock(
    tabular_inputs, 
    torch4rec.MLPBlock([64]), 
    torch4rec.TransformerBlock(transformer_config, masking=inputs.masking)
)
```


### Prediction head
The library supports the following prediction heads. They can have multiple losses, that can be combined for multi-task learning and multiple metrics.

- **Item Prediction** - Predicts items for a given sequence of interactions. During training it can be the next item or randomly selected items, depending on the masking scheme. For inference it is meant to always predict the next interacted item. Currently cross-entropy and some pairwise losses are supported. 
- **Classification** - Predicts a categorical feature using the whole sequence. In the context of recommendation, it can be used to predict for example if the user is going to abandon a product added to cart or proceed to its purchase.
- **Regression** - Predicts a continuous feature using the whole sequence. The label could be for example the elapsed time until the user returns to a service.

In the following example, it is instantiated a head with the pre-defined `model_body` for the `NextItemPredictionTask`. That head enables the `weight_tying` option, which is described in the next section.  
Decoupling model bodies and heads allow for a flexible model architecture definition, as it allows for multiple towers and/or heads. Finally, the `Model` class combines the heads and wraps the whole model.

```python
from transformers4rec.torch import Head, Model
from transformers4rec.torch.model.head import NextItemPredictionTask

# Defines the head related to next item prediction task 
head = Head(
    model_body,
    NextItemPredictionTask(weight_tying=True, hf_format=True),
    inputs=inputs,
)

# Get the end-to-end Model class 
model = Model(head)
```

### Tying embeddings
For `NextItemPredictionTask`, it is available an option called **Tying Embeddings**, proposed originally by the NLP community to tie the weights of the input (item id) embedding matrix with the output projection layer. **Tying Embeddings** showed to be a very effective technique in extensive experimentation for competitions and empirical analysis (more details in our [paper](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf) and its [online appendix](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/Appendices/Appendix_A-Techniques_used_in_Transformers4Rec_Meta-Architecture.md)). You can enable this option as follows.


### Regularization

The library supports a number of regularization techniques like Dropout, Weight Decay, Softmax Temperature Scaling, Stochastic Shared Embeddings, and Label Smoothing. In our extensive experimentation hypertuning all regularization techniques for different dataset we found that the Label Smoothing was particularly useful at improving both train and validation accuracy and better calibrating the predictions. 


More details of the options available for each building block can be found in our **API Documentation**.

## Training and evaluation

### Data loading
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


### PyTorch Training
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

NVTabular also outputs a schema of the parquet columns in Profobuf Text format, e.g. including the cardinality of categorical features, the max squence length for sequential features and tags that can be associated to features (e.g. to indicate what is the item id, what are item and user features, what are categorical or continuous features). You can see [here](../../tests/assets/yoochoose/schema.pbtxt) an example of such schema in Protobuf Text format.
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