## Build the image

```
cd recsys/transformers4recsys/
docker build --tag transf4rec_exp:0.1.0 --tag nvcr.io/nvidian/prj-recsys/transf4rec_exp:0.1.0 --no-cache container/
```

## Try locally

```
docker run --gpus all -it --rm transf4rec_exp:0.1.0
```

Run inside the container
```
cd /workspace/recsys/transformers4recsys/codes

git pull origin experimentation

#Download the eCommerce preprocessed dataset
bash script/get_dataset.bash

#Login into Weights&Biases
wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12

#Run training script
bash script/run_transformer_v2.bash full session_cooccurrence \
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

## Push image to NGC
```
docker push nvcr.io/nvidian/prj-recsys/transf4rec_exp:0.1.0
```

## Run a Job on NGC
```
ngc batch run --name "tranf4rec-job-$(date +%Y%m%d%H%M%S)" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --port 8888 \
--commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/get_dataset.bash && date && bash script/run_transformer_v2.bash full session_cooccurrence \
  --start_date 2019-10-01 \
  --end_date 2019-10-15 \
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
  --tf_out_activation tanh \
  && date"
```