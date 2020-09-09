## Build the image

```
cd recsys/transformers4recsys/
docker build --tag transf4rec_exp:0.1.0 container/
```

## Try locally

```
docker run --gpus all -it --rm transf4rec_exp:0.1.0
```

Run inside the container
```
#Download the eCommerce preprocessed dataset
bash script/get_dataset.bash

#Login into Weights&Biases
wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12

#Run training script
cd /workspace/recsys/transformers4recsys/codes
bash scripts/run_transformer_v2.bash full session_cooccurrence \
  --start_date "2019-10-01" \
  --end_date "2019-10-15" \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 128 \
  --model_type gpt2 \
  --loss_type cross_entropy_neg \
  --margin_loss 0.0 \
  --d_model 64 \
  --n_layer 4 \
  --n_head 2 \
  --hidden_act gelu \
  --dropout 0.2 \
  --learning_rate 1e-03  \
  --similarity_type concat_mlp \
  --num_train_epochs 15 \
  --all_rescale_factor 1.0 \
  --neg_rescale_factor 0.0 \
  --inp_merge mlp \
  --tf_out_activation tanh
```