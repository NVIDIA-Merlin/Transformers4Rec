# Additional Resources

## Papers
- [Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf) - Paper presented at the [ACM RecSys'21](https://recsys.acm.org/recsys21/) where we discuss the relationship between NLP and RecSys and introduce theTransformers4Rec library, describing its focus and core features. We also provide a comprehensive empirical analysis comparing Transformer architectures with session-based recommendation algorithms, which are outperformed by the former.

## Competitions
- **SIGIR eCommerce Workshop Data Challenge 2021, organized by Coveo** - NVIDIA Merlin team was one of the winners of this competition on predicting the next interacted products for user sessions in an e-commerce. In our solution we used only Transformer architectures. Check our [**post**](https://medium.com/nvidia-merlin/winning-the-sigir-ecommerce-challenge-on-session-based-recommendation-with-transformers-v2-793f6fac2994) and [**paper**](https://arxiv.org/abs/2107.05124).  
- **WSDM WebTour Challenge 2021 , organized by Booking. com** - Competition on next destination prediction for multi-city trips won by NVIDIA. We leveraged a model from the Transformers4Rec library in the final ensemble. Here is our solution [**post**](https://developer.nvidia.com/blog/how-to-build-a-winning-deep-learning-powered-recommender-system-part-3/) and [**paper**](http://ceur-ws.org/Vol-2855/challenge_short_2.pdf).


## NVIDIA Merlin
Transformers4Rec is part of the NVIDIA Merlin ecosystem for Recommender Systems. Check our other libraries:
- [NVTabular](https://github.com/NVIDIA/NVTabular/) - NVTabular is a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems. 
- [Triton Inference Server](https://github.com/triton-inference-server/server). - Provides a cloud and edge inferencing solution optimized for both CPUs and GPUs. Transformers4Rec models can be exported and served with Triton.
- [HugeCTR](https://github.com/NVIDIA/HugeCTR) - A GPU-accelerated recommender framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs). 


## Supported HuggingFace architectures and Pre-training approaches

Transformers4Rec supports the four following masking tasks: 

|Accronyms| Definition|
|---------|--------------|
| CLM    | Causal Language Modelling|
| MLM    | Masked Language Modelling|
| PLM    | Permutation Language Modelling|
| RTD    | Replacement Token Detection|


In Transformers4Rec, we decouple the pre-training approach from transformers architectures and provide `TransformerBlock` module that links the config class of transformers architecture to the masking task. Transformers4Rec also defines [`transformer_registry`](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/2c29411fe8c87c72d0a32314788c3ed0dbddb0ef/transformers4rec/config/transformer.py#L21) including pre-defined `T4RecConfig` constructors that automatically set the arguments of the related HuggingFace configuration classes. 
The table below represents the current supported architectures in Transformers4Rec and links them to the possible masking tasks. It also lists the registered `T4RecConfig` classes in `Included` column.


|   Model     | CLM |  MLM  |  PLM  |  RTD  | Included |
| ----------- |--------|-------|-------|-------|-------|
|    [BERT](https://huggingface.co/transformers/model_doc/bert.html#bertconfig)     |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|  [ConvBERT](https://huggingface.co/transformers/model_doc/convbert.html#convbertconfig)   |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|   [DeBERTa](https://huggingface.co/transformers/model_doc/deberta.html#debertaconfig)   |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|  [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html#distilbertmodel) |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|   [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2config)     |   ✅   | ❌     |   ❌   |  ❌  |   ✅   |
|  [Longformer](https://huggingface.co/transformers/model_doc/longformer.html#longformerconfig) |   ✅   | ✅     |   ❌   |  ❌  |   ✅   |
| [MegatronBert](https://huggingface.co/transformers/model_doc/megatron_bert.html#megatronbertconfig) |   ❌   |  ✅    |   ❌   |  ✅  |   ❌   |
|   [MPNet](https://huggingface.co/transformers/model_doc/mpnet.html#mpnetconfig)     |   ❌    |  ✅   |   ❌   |  ✅  |   ❌   |
|   [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html#robertaconfig)   |   ❌    |  ✅   |   ❌   |  ✅  |   ❌   |
|   [RoFormer](https://huggingface.co/transformers/model_doc/roformer.html#roformerconfig)  |   ✅    |  ✅   |   ❌   |  ✅  |   ❌   |
| [Transformer-XL](https://huggingface.co/transformers/model_doc/transformerxl.html#transfoxlconfig)|   ✅    | ❌     |   ❌   |  ❌    |   ❌   |
|   [XLNet](https://huggingface.co/transformers/model_doc/xlnet.html#xlnetconfig)    |   ✅    | ✅     |   ✅   |  ✅    |   ✅   |


 **Note**: The following HF architectures will be supported in future release: `Reformer`, `Funnel Transformer`, `ELECTRA` and `ALBERT`. 




## Other Resources
- [NVIDIA Merlin engineering blog](https://medium.com/nvidia-merlin)
- NVIDIA Developer blog
    - Post series on [how to build winning RecSys](https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-1/)
    - [Using Neural Networks for Your Recommender System](https://developer.nvidia.com/blog/using-neural-networks-for-your-recommender-system/)
