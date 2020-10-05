GPT2

python main.py --htune_study_name gpt2_hypertune_round_08 --concurrent_jobs 5 --htune_num_trials 100 --verbose --htune_log_dir /home/gmoreira/projects/nvidia/recsys_fork/recsys/transformers4recsys/resources/autobench_config/gpt2/2020-09-10/htune_log "/home/gmoreira/projects/nvidia/recsys_fork/recsys/transformers4recsys/resources/autobench_config/gpt2/2020-09-10/config_ngc_tranf4rec_hypertune.yaml"



------------------------------------



GRU


python main.py --htune_study_name gru_hypertune_round_09 --concurrent_jobs 5 --htune_num_trials 80 --verbose --htune_log_dir /home/gmoreira/projects/nvidia/recsys_fork/recsys/transformers4recsys/resources/autobench_config/gru/2020-09-12/htune_log "/home/gmoreira/projects/nvidia/recsys_fork/recsys/transformers4recsys/resources/autobench_config/gru/2020-09-12/config_ngc_tranf4rec_gru_hypertune.yaml"

model_cls = nn.GRU(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,



------------------------------------------------------------------------------------------------------------------------------



Tests with regularization (Weight Decay and learning rate schedule)

#Original in the hypertunning  (--per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.3)

ngc batch run --name "tranf4rec-job-autobench" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/run_transformer_v2.bash gpt2_hypertune_round_08 full session_cooccurrence --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --num_train_epochs 15 --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --neg_rescale_factor 0.0 --inp_merge mlp --per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.3 --d_model 128 --n_layer 1 --n_head 1 --hidden_act gelu && date" --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --datasetid 66173:/data


####AOD_all_Test_ndcg@1000_all
####0.319


--------------------------------------------------------


#(--per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.0 --weight_decay 1e-4)

ngc batch run --name "tranf4rec-job-autobench-drop-0-decay-1e-4" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/run_transformer_v2.bash gpt2_hypertune_round_08 full session_cooccurrence --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --num_train_epochs 15 --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --neg_rescale_factor 0.0 --inp_merge mlp --per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.0 --weight_decay 1e-4 --d_model 128 --n_layer 1 --n_head 1 --hidden_act gelu && date" --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --datasetid 66173:/data


#W&B:  glamorous-rain-502

---------------------------------------------------------

#(--per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.0 --weight_decay 1e-3)

ngc batch run --name "tranf4rec-job-autobench-drop-0-decay-1e-3" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/run_transformer_v2.bash gpt2_hypertune_round_08 full session_cooccurrence --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --num_train_epochs 15 --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --neg_rescale_factor 0.0 --inp_merge mlp --per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.0 --weight_decay 1e-3 --d_model 128 --n_layer 1 --n_head 1 --hidden_act gelu && date" --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --datasetid 66173:/data


#W&B:  mild-snow-503


---------------------------------------------------------

#(--per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.3 --weight_decay 1e-4)

ngc batch run --name "tranf4rec-job-autobench-drop-0.3-decay-1e-4" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/run_transformer_v2.bash gpt2_hypertune_round_08 full session_cooccurrence --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --num_train_epochs 15 --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --neg_rescale_factor 0.0 --inp_merge mlp --per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.3 --weight_decay 1e-4 --d_model 128 --n_layer 1 --n_head 1 --hidden_act gelu && date" --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --datasetid 66173:/data


#W&B:  olive-sky-504



---------------------------------------------------------

#(--per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.3 --weight_decay 1e-4 --learning_rate_schedule linear_with_warmup) with hardcoded L2 of 1e-4 on embedding tables


ngc batch run --name "tranf4rec-job-autobench-drop-0.3-decay-1e-4-embl2-1e-3" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/run_transformer_v2.bash gpt2_hypertune_round_08 full session_cooccurrence --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --num_train_epochs 15 --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --neg_rescale_factor 0.0 --inp_merge mlp --per_device_train_batch_size 64 --learning_rate 0.0021770576069357664 --dropout 0.3 --weight_decay 1e-4 --learning_rate_schedule linear_with_warmup --d_model 128 --n_layer 1 --n_head 1 --hidden_act gelu && date" --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --datasetid 66173:/data


#W&B:  crimson-flower-514



---------------------------------------------------------


#--per_device_train_batch_size 128 --learning_rate 1e-4 --dropout 0.1 --weight_decay 1e-4 --learning_rate_schedule constant_with_warmup 
#--learning_rate_warmup_steps 10 --d_model 64 --n_layer 1 --n_head 8 --hidden_act gelu

ngc batch run --name "tranf4rec-job-batch-128-constant-lr-1e-4" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/run_transformer_v2.bash gpt2_single_epoch full session_cooccurrence --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --num_train_epochs 1 --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --neg_rescale_factor 0.0 --inp_merge mlp --per_device_train_batch_size 128 --learning_rate 1e-4 --dropout 0.1 --weight_decay 1e-4 --learning_rate_schedule constant_with_warmup --learning_rate_warmup_steps 10 --d_model 64 --n_layer 1 --n_head 8 --hidden_act gelu && date" --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --datasetid 66173:/data



# dry-brook-516



---------------------------------------------------------


#--per_device_train_batch_size 128 --learning_rate 1e-3 --dropout 0.0 --weight_decay 1e-5 
#--learning_rate_schedule constant_with_warmup --learning_rate_warmup_steps 10 --d_model 128 --n_layer 1 --n_head 8 --hidden_act gelu

ngc batch run --name "tranf4rec-job-batch-128-constant-lr-1e-4" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline "nvidia-smi && wandb login 76eea90114bb1cdcbafe151b262e4a5d4ff60f12 && date && git pull origin experimentation && date && bash script/run_transformer_v2.bash gpt2_single_epoch full session_cooccurrence --start_date 2019-10-01 --end_date 2019-10-15 --model_type gpt2 --num_train_epochs 1 --loss_type cross_entropy --per_device_eval_batch_size 128 --similarity_type concat_mlp --tf_out_activation tanh --all_rescale_factor 1.0 --neg_rescale_factor 0.0 --inp_merge mlp --per_device_train_batch_size 128 --learning_rate 1e-3 --dropout 0.0 --weight_decay 1e-5 --learning_rate_schedule constant_with_warmup --learning_rate_warmup_steps 10 --d_model 128 --n_layer 1 --n_head 8 --hidden_act gelu && date" --result /results --image "nvidian/prj-recsys/transf4rec_exp:0.1.0" --org nvidian --team prj-recsys --datasetid 66173:/data


#W&B:  tough-terrain-517






