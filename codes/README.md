# RecSys-Transformer based on Huggingface's implementation

Current code uses following ingredients to implement RecSys-Transformer models:
- DataLoader to read from Parquet file: Petastorm (https://petastorm.readthedocs.io)
- Trainer and Transformer models implementation: Huggingface (https://huggingface.co/transformers/)
- Evaluation metrics: karlhigley's implementation (https://github.com/karlhigley/ranking-metrics-torch)
- Training and evaluation logging: Weights & Biases (wandb.com)

## Installation guide

Step 1. Get this repo (e.g., `git clone https://github.com/rapidsai/recsys.git`)

Step 2. Install PyTorch

Visit https://pytorch.org and follow their guideline. 

E.g., For Linux with cuda=10.2 use this:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Step 3. Install Huggingface and Petastorm

```
pip install -r requirements.txt
```
or
```
pip install transformers==3.0.0
pip install petastorm==0.9.2
pip install wandb
```

Step 3. Setup wandb for experiment logging

By default, Huggingface uses Weights & Biases (wandb) to log training and evaluation metrics. Let's keep use it.

1) Create account if you don't have and obtain API
https://www.wandb.com

2) Insert API
```
wandb login
```
Follow instruction to insert API key

## How to Run?

**Step 1. Get preprocessed e-commerce dataset**

We prepared the sample preprocessed dataset and uploaded at Google drive. You can get it by running following scripts:
```
bash script/get_dataset.bash
```

**Step 2. Run the training & evaluation code**
```
bash script/run_transformer.bash
```

## CODE
- `recsys_main.py`: main experiment-running (train+eval) code
- `recsys_models.py`: definition of various sequence models (Huggingface Transformers and PyTorch GRU,RNN,LSTMs)
- `recsys_meta_model.py`: RecSys wrapper model that gets embeddings for discrete input tokens and merges multiple sequences of product id, category id, etc. Then, it runs forward function of defined sequence model and computes loss.
- `recsys_trainer.py`: Extends Huggingface's trainer.py code to enable customized dataset in training and evaluation loops.
- `recsys_data.py`: setup for dataloader and necessaties to read Parquet file (currently use Petastorm)
- `recsys_metrics.py`: defines various evaluation metric computation function (e.g. Recall@k, Precision@k, NDCG, etc.) Any additional metric computation functions can be added and executed here.
- `recsys_args.py`: defines input args for the code.
