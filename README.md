# [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec/)

[![PyPI](https://img.shields.io/pypi/v/Transformers4Rec?color=orange&label=version)](https://pypi.python.org/pypi/Transformers4Rec)
[![LICENSE](https://img.shields.io/github/license/NVIDIA-Merlin/Transformers4Rec)](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/Transformers4Rec/main/README.html)

Transformers4Rec is a flexible and efficient library for sequential and session-based recommendation. It's available for both PyTorch and Tensorflow. It works as a bridge between Natural Language Processing (NLP) and recommender systems by integrating with one of the most popular NLP frameworks, Hugging Face Transformers (HF), making state-of-the-art transformer architectures available for RecSys researchers and industry practitioners. For more information about Hugging Face Transformers, refer to the [HuggingFace Transformers Documentation](https://github.com/huggingface/transformers) on GitHub.

<div align=center><img src="_images/sequential_rec.png" alt="Sequential and Session-based recommendation with Transformers4Rec" style="width:800px;"/><br>
<figcaption font-style: italic; align: center>Fig. 1: Sequential and Session-Based Recommendation with Transformers4Rec</figcaption></div>

<br><br/>

Over the past decade, proposed approaches based on NLP research, such as Word2Vec, GRU, and Attention for recommender systems (RecSys), have gained popularity with RecSys researchers and industry practitioners. This phenomena is especially noticeable for sequential and session-based recommendation where the sequential processing of user interactions is analogous to the language modeling (LM) task. Many key RecSys architectures have been adopted based on NLP research such as GRU4Rec, which is the seminal recurrent neural network (RNN) based architecture for session-based recommendation.

Transformer architectures have recently become the dominant technique over convolutional neural networks (CNNs) and recurrent neural networks for language modeling tasks. Because of their efficient parallel training, these architectures can scale training data and model sizes. Transformer architectures are also effective at modeling long-range sequences. 

Transformers have also been applied to sequential recommendation in architectures such as [SASRec](https://arxiv.org/abs/1808.09781), [BERT4Rec](https://arxiv.org/abs/1904.06690), and [BST](https://arxiv.org/pdf/1905.06874.pdf%C2%A0), providing higher accuracy than CNN and RNN based architectures. For more information, refer to our [ACM RecSys'21 paper](https://dl.acm.org/doi/10.1145/3460231.3474255). For more information about the evolution of Transformer architectures and bridging the gap between NLP and sequential/session-based recommendation, refer to our [Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation paper](https://dl.acm.org/doi/10.1145/3460231.3474255).

<div align=center><img src="_images/nlp_x_recsys.png" alt="timeline illustrating the influence of NLP research in Recommender Systems" style="width:800px;"/><br><figcaption style="font-style: italic;">Fig. 2: Timeline Illustrating the Influence of NLP Research in Recommender Systems as Referenced in the <a href="https://dl.acm.org/doi/10.1145/3460231.3474255)">Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation Paper</a></figcaption></div>

## Benefits

Transformers4Rec recently won two session-based recommendation competitions: [WSDM WebTour Workshop Challenge 2021 (organized by Booking.com)](https://developer.nvidia.com/blog/how-to-build-a-winning-deep-learning-powered-recommender-system-part-3/) and [SIGIR eCommerce Workshop Data Challenge 2021 (organized by Coveo)](https://medium.com/nvidia-merlin/winning-the-sigir-ecommerce-challenge-on-session-based-recommendation-with-transformers-v2-793f6fac2994). The usage of Transformers4Rec for session-based recommendation has also undergone extensive empirical evaluation, which was able to provide higher accuracy than baseline algorithms. These observations have been published in our [ACM RecSys'21 paper](https://dl.acm.org/doi/10.1145/3460231.3474255).

Transformers4Rec offers the following benefits:

- **Flexibility**: Transformers4Rec provides modularized building blocks that are configurable and compatible with vanilla PyTorch modules and TF Keras layers, making it easy for custom architectures to be created with multiple towers, multiple heads/tasks, and losses.

- **Accessibility to HF transformers**: More than 64 different Transformer architectures can be used to evaluate your sequential and session-based recommendation task as a result of the [Hugging Face Transformers](https://github.com/huggingface/transformers) integration.

- **Support for multiple input features**: HF transformers only support sequences of token IDs as input since it was originally designed for NLP. Transformers4Rec makes it possible for other types of sequential tabular data to be used as input with HF transformers due to the rich features that are available in RecSys datasets. Transformers4Rec uses a schema format to configure the input features and automatically creates the necessary layers, such as embedding tables, projection layers, and output layers based on the target, without requiring code changes to include new features. Interaction and sequence-level input features can be normalized and combined in configurable ways.

- **Seamless preprocessing and feature engineering**: The integration with [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) and [Triton Inference Server](https://github.com/triton-inference-server/server) allows you to build a fully GPU-accelerated pipeline for sequential and session-based recommendation. NVTabular has common preprocessing ops for session-based recommendation and exports a dataset schema, which is compatible with Transformers4Rec so that input features can be configured automatically. Exports have trained models to serve with Triton Inference Server in a single pipeline that includes online feature preprocessing and model inference. For more information, refer to [End-to-end pipeline with NVIDIA Merlin](pipeline.md).

<div align=center><img src="_images/pipeline.png" alt="GPU-accelerated sequential and session-based recommendation" style="width:600px;"/><br><figcaption style="font-style: italic;">Fig. 3: GPU-Accelerated Pipeline for Sequential and Session-Based Recommendation Using NVIDIA Merlin Components</figcaption></div>

## Sequential and Session-Based Recommendation Usage

Traditional recommendation algorithms usually ignore the temporal dynamics and the sequence of interactions when trying to model user behavior. Generally, the next user interaction is related to the sequence of the user's previous choices. In some cases, it might be even a repeated purchase or song play. User interests might also suffer from the interest drift since preferences might change over time. Those challenges are addressed by the **sequential recommendation** task.

A special use case of sequential-recommendation is the **session-based recommendation** task where you only have access to the short sequence of interactions within the current session. This is very common in online services like e-commerce, news, and media portals where the user might choose to browse anonymously due to GDPR compliance in which no cookies are collected, or because it's a new user. This task is also relevant for scenarios where the users' interests change a lot over time depending on the user context or intent, so leveraging the current session interactions is more promising than old interactions to provide relevant recommendations.

To deal with sequential and session-based recommendation, many sequence learning algorithms previously applied in machine learning and NLP research have been explored for RecSys based on k-Nearest Neighbors, Frequent Pattern Mining, Hidden Markov Models, Recurrent Neural Networks, and more recently neural architectures using the Self-Attention Mechanism and transformer architectures. Unlike Transformers4Rec, these existing frameworks only accept sequences of item IDs as input and don't provide a modularized, scalable implementation for production usage.

## Installation

Trasformers4Rec can be installed using one of the following methods:

- [pip](#installing-transformers4rec-using-pip)
- [conda](#installing-transformers4rec-using-conda)
- [Docker](#installing-transformers4rec-using-docker)

### Installing Transformers4Rec Using pip

Transformers4Rec comes in two flavors: PyTorch and Tensorflow. It can optionally use the GPU-accelerated NVTabular dataloader, which is highly recommended.
PyTorch and TensorFlow can be installed as optional args for the pip install package. 

**NOTE**: Installing Transformers4Rec with `pip` only supports the CPU version of NVTabular for now.

To install Transformers4Rec using pip, run one of the following commands:

- **All**: `pip install transformers4rec[all]`
- **PyTorch**: `pip install transformers4rec[pytorch,nvtabular]`
- **Tensorflow**: `pip install transformers4rec[tensorflow,nvtabular]`

### Installing Transformers4Rec Using conda

To install Transformers4Rec using Conda, run the following command:

`conda install -c nvidia transformers4rec`

### Installing Transformers4Rec Using Docker

Transformers4Rec is pre-installed within the following NVIDIA Merlin Docker containers:

| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-tensorflow-training | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training) | Transformers4Rec, NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin |
| merlin-pytorch-training    | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training)    | Transformers4Rec, NVTabular, and PyTorch
| merlin-inference           | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference)           | Transformers4Rec, NVTabular, PyTorch, and Triton Inference |  |

These Docker containers are available in the [NVIDIA container repository](https://ngc.nvidia.com/catalog/containers/nvidia:merlin). To use these Docker containers, you must first install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker. You can use the NGC links referenced in the table above to obtain more information about how to launch and run these containers.

## Defining and Training the Model

To define and train any model on a dataset:

1. Provide the [schema](https://nvidia-merlin.github.io/Transformers4Rec/main/api/merlin_standard_lib.schema.html#merlin_standard_lib.schema.schema.Schema) and            construct an input-module.

   If you encounter session-based recommendation issues, you typically want to use [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.features.html#transformers4rec.torch.features.sequence.TabularSequenceFeatures),   
   which merges context features with sequential features. 

2. Provide the prediction-tasks.

   The tasks that are provided right out of the box are available from our [API documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.model.html#module-transformers4rec.torch.model.prediction_task).

3. Construct a transformer-body and convert this into a model.

Here's a PyTorch example:
```python
from transformers4rec import torch as tr

schema: tr.Schema = tr.data.tabular_sequence_testing_data.schema
# Or read schema from disk: tr.Schema().from_json(SCHEMA_PATH)
max_sequence_length, d_model = 20, 64

# Define the input module to process tabular input-features
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=max_sequence_length,
    continuous_projection=d_model,
    aggregation="concat",
    masking="causal",
)
# Define one or multiple prediction-tasks
prediction_tasks = tr.NextItemPredictionTask()

# Define a transformer-config like the XLNet architecture
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
)
model: tr.Model = transformer_config.to_torch_model(input_module, prediction_tasks)
```

Here's a TensorFlow example:
```python
from transformers4rec import tf as tr

schema: tr.Schema = tr.data.tabular_sequence_testing_data.schema
# Or read schema from disk: tr.Schema().from_json(SCHEMA_PATH)
max_sequence_length, d_model = 20, 64

# Define the input module to process tabular input-features
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=max_sequence_length,
    continuous_projection=d_model,
    aggregation="concat",
    masking="causal",
)
# Define one or multiple prediction-tasks
prediction_tasks = tr.NextItemPredictionTask()

# Define a transformer-config like the XLNet architecture
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
)
model: tr.Model = transformer_config.to_tf_model(input_module, prediction_tasks)
```

### Feedback and Support

If you'd like to make direct contributions to Transformers4Rec, refer to [Contributing to Transformers4Rec](CONTRIBUTING.md). We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations. To further advance our Merlin roadmap, we encourage you to share all the details regarding your recommender system pipeline by going to https://developer.nvidia.com/merlin-devzone-survey.

If you're interested in learning more about how Transformers4Rec works, refer to our
[Transformers4Rec documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/getting_started.html). We also have [API documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/api/modules.html) that outlines the specifics of the available modules and classes within Transformers4Rec.
