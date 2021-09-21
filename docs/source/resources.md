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

## Other Resources
- [NVIDIA Merlin engineering blog](https://medium.com/nvidia-merlin)
- NVIDIA Developer blog
    - Post series on [how to build winning RecSys](https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-1/)
    - [Using Neural Networks for Your Recommender System](https://developer.nvidia.com/blog/using-neural-networks-for-your-recommender-system/)
