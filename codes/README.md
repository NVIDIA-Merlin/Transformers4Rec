# RecSys-Transformer based on Huggingface's implementation

Current code uses following ingredients to implement RecSys-Transformer models:
- DataLoader to read from Parquet file: Petastorm (https://petastorm.readthedocs.io)
- Trainer and Transformer models implementation: Huggingface (https://huggingface.co/transformers/)
- Evaluation metrics: karlhigley's implementation (https://github.com/karlhigley/ranking-metrics-torch)   


## Installation guide

Step 1. Get this repo (e.g., git clone ..)

Step 2. Install PyTorch
Visit https://pytorch.org and follow their guideline.

Step 3. Install Huggingface and Petastorm
```
pip install -r requirements.txt
```
or
```
pip install transformers==3.0.0
pip install petastorm==0.9.2
```



## How to Run?

Step 1. Get preprocessed e-commerce dataset 
We prepared the sample preprocessed dataset and uploaded at Google drive. You can get it with the instruction on this notebook:
https://github.com/rapidsai/recsys/blob/master/transformers4recsys/notebooks/fetch_preprocessed_dataset_google_drive.ipynb


Step 2. Run the training & evaluation code:
```
CUDA_VISIBLE_DEVICES=0 python main_runner.py --output_dir ./tmp/ --do_eval --data_path ~/dataset/ecommerce_preproc_2019-10/ecommerce_preproc.parquet/ --per_device_train_batch_size 128
```

## NOTE
- Current version does not support multi-gpu. It shows error when try to use multiple gpus. (Error is caused when the number of elements in each batch at different GPUs are different. Seems not easy to fix so far.)

- Evaluation metrics part is still under-construction. 
