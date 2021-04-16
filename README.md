# Transformers4Rec
## Transformer-based neural architectures and pipelines for sequential, session-based and session-aware recommendations


## Objective
The objective of this project is to be a **bridge between NLP and recommender systems research**, by leveraging the most popular framework for Transformer architectures -- the [HuggingFace Transformers project](https://github.com/huggingface/transformers) -- and making those SOTA NLP building blocks available for researchers and industry practitioners working on sequential, session-based, and session-aware recommendation tasks.


## Context
Traditional recommendation algorithms usually ignore the temporal dynamics and the sequence of interactions when trying to model user behaviour. Generally, the next user interaction is related to the sequence of his previous choices. Inclusive, it can be a repeated purchase or song play. But his interests might also suffer from the interest drift, as his preferences might change over time. Those challenges are addressed by the problem of **sequential recommendation**. 
A special case of sequential-recommendation is the **session-based recommendation** task, where you have only access to the short sequence of interactions within the current session. This is very common in online services like e-commerce, news and media portals where the user might choose to browse anonymously (and due to GDPR compliance no cookies are collected), or because it is a new user. In the **session-aware recommendation** task,  you can leverage both information about the current session and past user sessions. 

To deal with sequential recommendation, many sequence learning techniques previously applied in NLP research have been explored for RecSys, like Frequent Pattern Mining, Hidden Markov Models, Recurrent Neural Networks, and more recently neural architectures using the Self-Attention Mechanism and the Transformer architectures.

## Project Overview
As many of the inspirations for sequential recommendation come from NLP, this research project aims to build a bridge between those research areas.
The [HuggingFace transformers project](https://github.com/huggingface/transformers) is by far the most popular framework on the Transformers Architecture, where top researchers have been contributing with their new fancy Transformers architectures for NLP.

The Transformers4Rec framework uses the HuggingFace Transformers library to **leverage the increasing number of Transformer architectures for sequential recommendation**.
With such an approach, the RecSys community is now able to easily compare different neural architectures for sequential / session-based / session-aware recommendation and investigate those that perform better for different use cases and datasets.

Differently from most implementations of Transformers for RecSys, which use only the sequence of interacted item ids to model users preferences over time, this framework allows to use **multiple input features** (side information) to represent the item (e.g. product category and price, news content embedding) and the user context (e.g. time, location, device), for more accurate recommendations.

# Project Organization

- [`hf4rec`](hf4rec/README.md) - Here are the main scripts for train and evaluating of Transformer-based RecSys models. The train and evaluation pipelines are PyTorch-based. 
- [`containers`](containers/README.md) - Dockerfiles to get the development and deployment enviroments setup
- [`datasets`](https://github.com/rapidsai/hf4rec/tree/main/datasets) - Resources for each dataset used in the experiments, including preprocessing scripts and config files of the available features for the models.
- [`resources`](https://github.com/rapidsai/hf4rec/tree/main/resources) - Additional resources, like examples of Visual Studio Code config files (launch.json, settings.json).



