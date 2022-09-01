# Why Transformers4Rec?

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## The relationship between NLP and RecSys

Over the past decade, proposed approaches based on NLP research, such as Word2Vec, GRU, and Attention for RecSys, have gained popularity with RecSys researchers and industry practitioners.
This phenomena is especially noticeable for sequential and session-based recommendation where the sequential processing of user interactions is analogous to the language modeling (LM) task.
Many key RecSys architectures have been adopted based on NLP research such as GRU4Rec.
GRU4Rec is the seminal recurrent neural network (RNN) based architecture for session-based recommendation.

Transformer architectures have become the dominant technique over convolutional neural networks (CNNs) and recurrent neural networks (RNNs) for language modeling tasks.
Because of their efficient parallel training, these architectures can scale training data and model sizes.
Transformer architectures are also effective at modeling long-range sequences.

Transformers have also been applied to sequential recommendation in architectures such as [SASRec](https://arxiv.org/abs/1808.09781), [BERT4Rec](https://arxiv.org/abs/1904.06690), and [BST](https://arxiv.org/pdf/1905.06874.pdf%C2%A0).
These architectures can provide higher accuracy than CNN and RNN-based architectures.
For more information, see our [ACM RecSys 2021 paper](https://dl.acm.org/doi/10.1145/3460231.3474255).
For more information about the evolution of Transformer architectures and bridging the gap between NLP and sequential and session-based recommendation, see our [Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation paper](https://dl.acm.org/doi/10.1145/3460231.3474255).

<img src="_images/nlp_x_recsys.png" alt="A timeline illustrating the influence of NLP research in Recommender Systems" style="width:800px;display:block;margin-left:auto;margin-right:auto;"/><br>

<div style="text-align: center; margin: 20pt">
<figcaption style="font-style: italic;">A timeline illustrating the influence of NLP research in Recommender Systems, from the <a href="https://dl.acm.org/doi/10.1145/3460231.3474255)">Transformers4Rec paper</a></figcaption>
</div>

## Integration with HuggingFace Transformers

Transformers4Rec integrates with the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) library, allowing RecSys researchers and practitioners to easily experiment with the latest and state-of-the-art NLP Transformer architectures for sequential and session-based recommendation tasks and deploy those models into production.

The HF Transformers was *"established with the goal of opening up advancements in NLP to the wider machine learning community"*. It has become very popular among NLP researchers and practitioners (more than 900 contributors), providing standardized implementations of the state-of-the-art Transformer architectures (more than 68 and counting) produced by the research community, often within days or weeks of their publication.

HF Transformers is designed for both research and production. Models are composed of three building blocks:

* A *tokenizer* that converts raw text to sparse index encodings.
* A *Transformer architecture*.
* A *head* for NLP tasks such as text classification, generation, sentiment analysis, translation, summarization, among others.

Transformers4Rec leverages the Transformer architectures building block and configuration classes from Hugging Face.
Transformers4Rec provides additional blocks that are necessary for recommendation, such as input features normalization and aggregation, and heads for recommendation and sequence classification and prediction.
The library also extends the {func}`Trainer <transformers4rec.torch.trainer.Trainer>` class to allow for the evaluation with RecSys metrics.
