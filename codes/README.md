# RecSys-Transformer based on Huggingface's implementation

Current code uses following ingredients to implement RecSys-Transformer models:
- DataLoader to read from Parquet file: Petastorm (https://petastorm.readthedocs.io)
- Trainer and Transformer models implementation: Huggingface (https://huggingface.co/transformers/)
- Evaluation metrics: karlhigley's implementation (https://github.com/karlhigley/ranking-metrics-torch)   


## Installation guide

Step 1. Install PyTorch
Visit https://pytorch.org and follow their guideline.

Step 2. Install Huggingface and Petastorm
```
pip install -r requirements.txt
```

Step 3. Get this repo and run following:
```
CUDA_VISIBLE_DEVICES=0 python main_runner.py --output_dir ./tmp/ --do_eval --data_path ~/dataset/ecommerce_preproc_2019-10/ecommerce_preproc.parquet/ --per_device_train_batch_size 128
```

## NOTE
- Current version does not support multi-gpu. It shows error when try to use multiple gpus. (Error is caused when the number of elements in each batch at different GPUs are different. Seems not easy to fix so far.)

- Evaluation metrics part is still under-construction. 

