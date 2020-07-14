cd ..

CUDA_VISIBLE_DEVICES=0 python recsys_main.py --output_dir ./tmp/ --do_train --do_eval --data_path ~/dataset/sessions_with_neg_samples_example/ --per_device_train_batch_size 128 --model_type xlnet --fast-test