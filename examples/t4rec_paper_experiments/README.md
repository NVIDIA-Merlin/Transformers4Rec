# Transformers4Rec paper - Experiments reproducibility

## Context
The Transformers4Rec library was introduced in a [paper](https://dl.acm.org/doi/10.1145/3460231.3474255) at RecSys'21, which reports experiments on session-based recommendation for two e-commerce and two news datasets.

The original experiments for that paper were performed in a former pre-release version of the Transformers4Rec library tagged as [recsys2021](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/recsys2021), which can be used for the full reproducibility of paper experiments, as detailed [here](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/experiments_reproducibility_commands.md) in the paper online appendices.

## Paper experiments reproducibility with the released Transformers4Rec API

In this example, we demonstrate how to reproduce the most of the paper experiments (only Transformers, not the baselines algorithms) with the PyTorch API of the released Transformers4Rec library.

For researchers and practitioners aiming to perform experiments similar to the ones presented in our paper (e.g. incremental training and evaluation of session-based recommendation with Transformers), we strongly encourage the usage of our released PyTorch API (like in this example), because it is more modularized and documented than the original scripts, and is supported by the NVIDIA Merlin team.

A few warnings:
- It is natural to find some differences in evaluation metrics results, as the library was completely refactored after the paper experiments and even the same random seeds won't initialize the model weights identically when layers are build in different order.
- *WIP*: We are still working to add a few missing components in our PyTorch API necessary to reproduce some experiments: (1) Including ALBERT and ELECTRA and (2) including the Replacement Token Detection (RTD) training task. You can track the progress of those issues in [#262](https://github.com/NVIDIA-Merlin/Transformers4Rec/issues/262) and [#263](https://github.com/NVIDIA-Merlin/Transformers4Rec/issues/263).

### Datasets

We have used four datasets for the paper experiments with session-based recommendation:
- REES46 ecommerce
- YOOCHOOSE ecommerce
- G1 news
- ADRESSA news

We provide links to download the original datasets and preprocessing scripts [here](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/recsys2021/datasets), i.e., for creating features and grouping interactions features by sessions.

But for your convenience we also provide the [pre-processed version of the datasets](https://drive.google.com/drive/folders/1fxZozQuwd4fieoD0lmcD3mQ2Siu62ilD?usp=sharing) for download, so that you can directly jump into running experiments with Transformers4Rec.

### Requirements and Setup

To run the experiments you need:

- Install Transformers4Rec for PyTorch API and NVTabular (more instructions [here](https://github.com/NVIDIA-Merlin/Transformers4Rec)): `pip install transformers4rec[pytorch,nvtabular]`
- Install the example additional requirements (available in this folder): `pip install -r requirements.txt`

### Weights & Biases logging setup

By default, Huggingface uses [Weights & Biases (W&B)](https://wandb.ai/) to log training and evaluation metrics.
It allows a nice management of experiments, including config logging, and provides plots with the evolution of losses and metrics over time.
To see the experiment metrics reported in W&B you can follow the following steps. Otherwise you need to disable wandb sync to the online service by setting this environment variable: `WANDB_MODE="dryrun"`.

1. Create account if you don't have and obtain API key: https://www.wandb.com
2. Run the following command and follow the instructions to get an API key.
```bash
wandb login
```
After you get the API key, you can set it as an environment variable with
```bash
export WANDB_API_KEY=<YOUR API KEY HERE>
```

### Training and evaluation commands
In our paper we have performed hyperparameter tuning for each experiment group (dataset and algorithm pair), whose search space and best hyperparameters can be found in the paper [Online Appendix C](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/Appendices/Appendix_C-Hyperparameters.md). The command lines to run each experiment group with the best hyperparameters using the original scripts can be found [here](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/experiments_reproducibility_commands.md).

This example script was implemented using the released PyTorch API of the Transformers4Rec library, keeping compatibility with the command line arguments used for the [original scripts](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/recsys2021).

To reproduce the paper experiments with this example, you just need to perform two replacements in the [original scripts command lines](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/experiments_reproducibility_commands.md):
1. Replace the Python package name and script  `hf4rec.recsys_main` by `t4r_paper_repro.transf_exp_main`
2. Replace the argument `--feature_config [.yaml file path]` by `--features_schema_path [schema file path]`, as previously we used an YAML file to configure dataset features and now we use a features schema protobuf text file for the same purpose.
3. For experiments using multiple features (RQ3), include the `--use_side_information_features` argument

Below is the updated command to reproduce the experiment [TRANSFORMERS WITH MULTIPLE FEATURES - XLNet (MLM)](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/experiments_reproducibility_commands.md#xlnet-mlm) for the REES46 ECOMMERCE DATASET.

```bash
DATA_PATH=~/transformers4rec_paper_preproc_datasets/ecom_rees46/
FEATURE_SCHEMA_PATH=datasets_configs/ecom_rees46/rees46_schema.pbtxt
CUDA_VISIBLE_DEVICES=0 python3 -m t4r_paper_repro.transf_exp_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_PATH --features_schema_path $FEATURE_SCHEMA_PATH --fp16 --data_loader_engine merlin --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation concat --per_device_train_batch_size 256 --learning_rate 0.00020171456712823088 --dropout 0.0 --input_dropout 0.0 --weight_decay 2.747484129693843e-05 --d_model 448 --item_embedding_dim 448 --n_layer 2 --n_head 8 --label_smoothing 0.5 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.015 --mlm_probability 0.1 --embedding_dim_from_cardinality_multiplier 3.0 --eval_on_test_set --seed 100 --use_side_information_features
```

#### Remarks
For the experiments with multiple input features and element-wise aggregation of features (`--input_features_aggregation elementwise_sum_multiply_item_embedding`), it is necessary that all features have the same dimension. In the original implementation of the paper experiments we used a linear layer to project the continuous features to the the same dimension of the categorical embeddings. But that option is not available in the API, as the [soft-one hot embedding technique](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/Appendices/Appendix_A-Techniques_used_in_Transformers4Rec_Meta-Architecture.md) is more effective to represent continuous features. So reproducing exactly the experiment results for the element-wise aggregation will not be possible with the new API, but instead we recommend enabling Soft-One Hot Embeddings to represent continuous features, by setting the arguments `--numeric_features_project_to_embedding_dim` to be equal to the `--item_embedding_dim` value and also `--numeric_features_soft_one_hot_encoding_num_embeddings` to the number of desired embeddings (generally a value between 10-20 is a good default choice).

### Code organization
The main script of this example is the `transf_exp_main.py` script that is available from the [`t4r_paper_repro`](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/t4rec_paper_experiments/t4r_paper_repro) directory of the GitHub repository.
This script parses the command line arguments and use the Transformers4Rec PyTorch API to build a model for session-based recommendation according to the arguments and perform incremental training and evaluation over time.

The available command-line arguments are configured in the `transf_exp_args.py` script and logic for logging and saving results is available in the `exp_outputs.py` script.
