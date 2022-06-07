# Additional Resources

## Transformers4Rec and Session-Based Recommendation
- [Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation](https://dl.acm.org/doi/10.1145/3460231.3474255) - Paper presented at the [ACM RecSys'21](https://recsys.acm.org/recsys21/) where we discuss the relationship between NLP and RecSys and introduce Transformers4Rec along with its core features. We also provide a comprehensive empirical analysis comparing Transformer architectures with session-based recommendation algorithms. To obtain the **Online Appendices of the Paper** and **Experiments Reproducibility**, refer to this [README](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/README.md).
- [Blog post](https://medium.com/nvidia-merlin/transformers4rec-4523cc7d8fa8) - Briefly introduces Transformers4Rec.
- [End-to-end session based recommendation demo](https://www.youtube.com/watch?v=ajegb0W-JbU) - Recorded demo presented at ACM RecSys'21 about end-to-end session-based recommendation using NVTabular, Transformers4Rec, and Triton.
- [Session-based recommenders](https://developer.nvidia.com/session-based-recommenders?ncid=so-medi-151539#cid=dl19_so-medi_en-us) - Provides session-based recommendation resources.

## Competitions
- **SIGIR eCommerce Workshop Data Challenge 2021 (organized by Coveo)** - The NVIDIA Merlin team won this competition by using Transformer architectures to predict the next interacted products for user sessions in an e-commerce. For more information about our solution, refer to our [blog post](https://medium.com/nvidia-merlin/winning-the-sigir-ecommerce-challenge-on-session-based-recommendation-with-transformers-v2-793f6fac2994) and [paper](https://arxiv.org/abs/2107.05124).
- **WSDM WebTour Challenge 2021 (organized by Booking. com)** - The NVIDIA Merlin team won this competition by leveraging a Transformers4Rec model in the final ensemble. For more information about our solution, refer to our [blog post](https://developer.nvidia.com/blog/how-to-build-a-winning-deep-learning-powered-recommender-system-part-3/) and [paper](http://ceur-ws.org/Vol-2855/challenge_short_2.pdf).

## NVIDIA Merlin
Transformers4Rec is part of the NVIDIA Merlin ecosystem for recommender systems, which includes the following components:
- [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular/) - NVTabular is a feature engineering and preprocessing library for tabular data that is designed to easily manipulate datasets at terabyte scale and train deep learning (DL) based recommender systems.
- [Triton Inference Server](https://github.com/triton-inference-server/server) - Provides a cloud and edge inferencing solution that is optimized for both CPUs and GPUs. Transformers4Rec models can be exported and served with Triton.
- [HugeCTR](https://github.com/NVIDIA/HugeCTR) - A GPU-accelerated recommender framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs).

## Supported Hugging Face Architectures and Pre-Training Approaches
Transformers4Rec supports the following [masking tasks](https://nvidia-merlin.github.io/Transformers4Rec/main/model_definition.html#sequence-masking):

- Causal Language Modeling (CLM)
- Masked Language Modeling (MLM)
- Permutation Language Modeling (PLM)
- Replacement Token Detection (RTD)

In Transformers4Rec, we decouple the pre-training approaches from transformers architectures and provide a `TransformerBlock` module that links the config class of the transformer architecture to the masking task. Transformers4Rec also defines a `transformer_registry`, which includes pre-defined [`T4RecConfig`](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.config.html#transformers4rec.config.transformer.T4RecConfig) constructors that automatically set the arguments of the related Hugging Face Transformers' configuration classes.

The table below represents the architectures that are currently supported in Transformers4Rec and links them to the possible masking tasks. It also lists the pre-registered `T4RecConfig` classes in the `Registered` column.

**Tip**: Consider registering HF Transformers config classes into Transformers4Rec as this can be a great first contribution.


|   Model     | CLM |  MLM  |  PLM  |  RTD  | Registered |
| ----------- |--------|-------|-------|-------|-------|
|    [AlBERT](https://huggingface.co/transformers/model_doc/albert.html#bertconfig) |   ❌   |  ✅    |   ❌   |  ✅  |   ✅   |
|    [BERT](https://huggingface.co/transformers/model_doc/bert.html#bertconfig)     |   ❌   |  ✅    |   ❌   |  ✅  |   ✅   |
|  [ConvBERT](https://huggingface.co/transformers/model_doc/convbert.html#convbertconfig)   |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|   [DeBERTa](https://huggingface.co/transformers/model_doc/deberta.html#debertaconfig)   |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|  [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html#distilbertmodel) |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|   [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2config)     |   ✅   | ❌     |   ❌   |  ❌  |   ✅   |
|  [Longformer](https://huggingface.co/transformers/model_doc/longformer.html#longformerconfig) |   ✅   | ✅     |   ❌   |  ❌  |   ✅   |
| [MegatronBert](https://huggingface.co/transformers/model_doc/megatron_bert.html#megatronbertconfig) |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|   [MPNet](https://huggingface.co/transformers/model_doc/mpnet.html#mpnetconfig)     |   ❌    |  ✅   |   ❌   |  ✅  |   ❌   |
|   [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html#robertaconfig)   |   ❌    |  ✅   |   ❌   |  ✅  |   ✅   |
|   [RoFormer](https://huggingface.co/transformers/model_doc/roformer.html#roformerconfig)  |   ✅    |  ✅   |   ❌   |  ✅  |   ❌   |
| [Transformer-XL](https://huggingface.co/transformers/model_doc/transformerxl.html#transfoxlconfig)|   ✅    | ❌     |   ❌   |  ❌    |   ✅   |
|   [XLNet](https://huggingface.co/transformers/model_doc/xlnet.html#xlnetconfig)    |   ✅    | ✅     |   ✅   |  ✅    |   ✅   |

 **Note**: The following HF architectures will eventually be supported: `Reformer`, `Funnel Transformer`, and `ELECTRA`.

## Other Resources
- [NVIDIA Merlin engineering blog](https://medium.com/nvidia-merlin)
- NVIDIA developer blogs:
    - [Technical walkthrough on how to build a winning recommendation system](https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-1/)
    - [Using Neural Networks for Your Recommender System](https://developer.nvidia.com/blog/using-neural-networks-for-your-recommender-system/)
