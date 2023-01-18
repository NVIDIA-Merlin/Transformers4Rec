# [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec/)

[![PyPI](https://img.shields.io/pypi/v/Transformers4Rec?color=orange&label=version)](https://pypi.python.org/pypi/Transformers4Rec)
[![LICENSE](https://img.shields.io/github/license/NVIDIA-Merlin/Transformers4Rec)](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/Transformers4Rec/main/README.html)

Transformers4Rec is a flexible and efficient library for sequential and session-based recommendation and can work with PyTorch.

The library works as a bridge between natural language processing (NLP) and recommender systems (RecSys) by integrating with one of the most popular NLP frameworks, [Hugging Face Transformers](https://github.com/huggingface/transformers) (HF).
Transformers4Rec makes state-of-the-art transformer architectures available for RecSys researchers and industry practitioners.

The following figure shows the use of the library in a recommender system.
Input data is typically a sequence of interactions such as items that are browsed in a web session or items put in a cart.
The library helps you process and model the interactions so that you can output better recommendations for the next item.

<img src="_images/sequential_rec.png" alt="Sequential and Session-based recommendation with Transformers4Rec" style="width:800px;display:block;margin-left:auto;margin-right:auto;"/><br>
<div style="text-align:center;margin:20pt">
  <figcaption style="font-style:italic;">Sequential and Session-based recommendation with Transformers4Rec</figcaption>
</div>

Traditional recommendation algorithms usually ignore the temporal dynamics and the sequence of interactions when trying to model user behavior.
Generally, the next user interaction is related to the sequence of the user's previous choices.
In some cases, it might be a repeated purchase or song play.
User interests can also suffer from interest drift because preferences can change over time.
Those challenges are addressed by the **sequential recommendation** task.

A special use case of sequential-recommendation is the **session-based recommendation** task where you only have access to the short sequence of interactions within the current session.
This is very common in online services like e-commerce, news, and media portals where the user might choose to browse anonymously due to GDPR compliance that restricts collecting cookies or because the user is new to the site.
This task is also relevant for scenarios where the users' interests change a lot over time depending on the user context or intent.
In this case, leveraging the interactions for the current session is more promising than old interactions to provide relevant recommendations.

To deal with sequential and session-based recommendation, many sequence learning algorithms previously applied in machine learning and NLP research have been explored for RecSys based on k-Nearest Neighbors, Frequent Pattern Mining, Hidden Markov Models, Recurrent Neural Networks, and more recently neural architectures using the Self-Attention Mechanism and transformer architectures.
Unlike Transformers4Rec, these frameworks only accept sequences of item IDs as input and do not provide a modularized, scalable implementation for production usage.

## Benefits of Transformers4Rec

Transformers4Rec offers the following benefits:

- **Flexibility**: Transformers4Rec provides modularized building blocks that are configurable and compatible with standard PyTorch modules.
This building-block design enables you to create custom architectures with multiple towers, multiple heads/tasks, and losses.

- **Access to HF Transformers**: More than 64 different Transformer architectures can be used to evaluate your sequential and session-based recommendation task as a result of the [Hugging Face Transformers](https://github.com/huggingface/transformers) integration.

- **Support for multiple input features**: HF Transformers only support sequences of token IDs as input because it was originally designed for NLP.
Transformers4Rec enables you to use other types of sequential tabular data as input with HF transformers due to the rich features that are available in RecSys datasets.
Transformers4Rec uses a schema to configure the input features and automatically creates the necessary layers, such as embedding tables, projection layers, and output layers based on the target without requiring code changes to include new features.
You can normalize and combine interaction and sequence-level input features in configurable ways.

- **Seamless preprocessing and feature engineering**: As part of the Merlin ecosystem, Transformers4Rec is integrated with [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) and [Triton Inference Server](https://github.com/triton-inference-server/server).
These components enable you to build a fully GPU-accelerated pipeline for sequential and session-based recommendation.
NVTabular has common preprocessing operations for session-based recommendation and exports a dataset schema.
The schema is compatible with Transformers4Rec so that input features can be configured automatically.
You can export your trained models to serve with Triton Inference Server in a single pipeline that includes online feature preprocessing and model inference.
For more information, refer to [End-to-end pipeline with NVIDIA Merlin](https://nvidia-merlin.github.io/Transformers4Rec/main/pipeline.html).

<img src="_images/pipeline.png" alt="GPU-accelerated Sequential and Session-based recommendation" style="width:600px;display:block;margin-left:auto;margin-right:auto;"/><br>
<div style="text-align: center; margin: 20pt">
  <figcaption style="font-style: italic;">GPU-accelerated pipeline for Sequential and Session-based recommendation using NVIDIA Merlin components</figcaption>
</div>

## Transformers4Rec Achievements

Transformers4Rec recently won two session-based recommendation competitions: [WSDM WebTour Workshop Challenge 2021 (organized by Booking.com)](https://developer.nvidia.com/blog/how-to-build-a-winning-deep-learning-powered-recommender-system-part-3/) and [SIGIR eCommerce Workshop Data Challenge 2021 (organized by Coveo)](https://medium.com/nvidia-merlin/winning-the-sigir-ecommerce-challenge-on-session-based-recommendation-with-transformers-v2-793f6fac2994).
The library provides higher accuracy for session-based recommendation than baseline algorithms and we performed extensive empirical analysis about the accuracy.
These observations are published in our [ACM RecSys'21 paper](https://dl.acm.org/doi/10.1145/3460231.3474255).

## Sample Code: Defining and Training the Model

Training a model with Transformers4Rec typically requires performing the following high-level steps:

1. Provide the [schema](https://nvidia-merlin.github.io/Transformers4Rec/main/api/merlin_standard_lib.schema.html#merlin_standard_lib.schema.schema.Schema) and construct an input-module.

   If you encounter session-based recommendation issues, you typically want to use the
   [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.features.html#transformers4rec.torch.features.sequence.TabularSequenceFeatures)
   class because it merges context features with sequential features.

2. Provide the prediction-tasks.

   The tasks that are provided right out of the box are available from our [API documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.model.html#module-transformers4rec.torch.model.prediction_task).

3. Construct a transformer-body and convert this into a model.

The following code sample shows how to define and train an XLNet model with PyTorch for next-item prediction task:

```python
from transformers4rec import torch as tr

# Create a schema or read one from disk: tr.Schema().from_json(SCHEMA_PATH).
schema: tr.Schema = tr.data.tabular_sequence_testing_data.schema

max_sequence_length, d_model = 20, 64

# Define the input module to process the tabular input features.
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=max_sequence_length,
    continuous_projection=d_model,
    aggregation="concat",
    masking="causal",
)

# Define a transformer-config like the XLNet architecture.
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
)

# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    input_module,
    tr.MLPBlock([d_model]),
    tr.TransformerBlock(transformer_config, masking=input_module.masking)
)

# Define the evaluation top-N metrics and the cut-offs
metrics = [NDCGAt(top_ks=[20, 40], labels_onehot=True),
           RecallAt(top_ks=[20, 40], labels_onehot=True)]

# Define a head with NextItemPredictionTask.
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, metrics=metrics),
    inputs=input_module,
)

# Get the end-to-end Model class.
model = tr.Model(head)
```

> You can modify the preceding code to perform binary classification.
> The masking in the input module can be set to `None` instead of `causal`.
> When you define the head, you can replace the `NextItemPredictionTask`
> with an instance of `BinaryClassificationTask`.
> See the sample code in the [API documentation for the class](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.BinaryClassificationTask).

## Installation

You can install Transformers4Rec with Pip, Conda, or run a Docker container.

### Installing Transformers4Rec Using Pip

You can install Transformers4Rec with the functionality to use the GPU-accelerated Merlin dataloader.
Installation with the dataloader is highly recommended for better performance.
Those components can be installed as optional arguments for the `pip install` command.

To install Transformers4Rec using Pip, run the following command:

```shell
pip install transformers4rec[pytorch,nvtabular,dataloader]
```

> Be aware that installing Transformers4Rec with `pip` only supports the CPU version of Merlin Dataloader because `pip` does not install cuDF.
> The GPU capabilities of the dataloader are available by using the Docker container or by installing
> the dataloader with Conda first and then performing the `pip` installation within the Conda environment.

### Installing Transformers4Rec Using Conda

To install Transformers4Rec using Conda, run the following command:

```shell
conda install -c nvidia transformers4rec
```

### Installing Transformers4Rec Using Docker

Transformers4Rec is pre-installed in the `merlin-pytorch` container that is available from the NVIDIA GPU Cloud (NGC) catalog.

Refer to the [Merlin Containers](https://nvidia-merlin.github.io/Merlin/main/containers.html) documentation page for information about the Merlin container names, URLs to container images in the catalog, and key Merlin components.

## Notebook Examples and Tutorials

The [End-to-end pipeline with NVIDIA Merlin](https://nvidia-merlin.github.io/Transformers4Rec/main/pipeline.html) page
shows how to use Transformers4Rec and other Merlin libraries like NVTabular to build a complete recommender system.

We have several [example](./examples) notebooks to help you build a recommender system or integrate Transformers4Rec into your system:

- A getting started example that includes training a session-based model with an XLNET transformer architecture.
- An end-to-end example that trains a model and takes the next step to serve inference with Triton Inference Server.
- Another end-to-end example that trains and evaluates a session-based model on RNN and also serves inference with Triton Inference Server.
- A notebook and scripts that reproduce the experiments presented in a paper for RecSys 2021.

## Feedback and Support

If you'd like to make direct contributions to Transformers4Rec, refer to [Contributing to Transformers4Rec](CONTRIBUTING.md). We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations. To further advance our Merlin roadmap, we encourage you to share all the details regarding your recommender system pipeline by going to https://developer.nvidia.com/merlin-devzone-survey.

If you're interested in learning more about how Transformers4Rec works, refer to our
[Transformers4Rec documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/getting_started.html). We also have [API documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/api/modules.html) that outlines the specifics of the available modules and classes within Transformers4Rec.
