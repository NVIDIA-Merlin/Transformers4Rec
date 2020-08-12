#!/usr/bin/env

python3 -m codes.preprocessing.append_neg_samples_to_parquet \
--input_parquet_path_pattern "/home/gmoreira/dataset/ecommerce_preproc_2019-*/ecommerce_preproc.parquet/session_start_date=*" \
--output_parquet_root_path "/home/gmoreira/dataset/ecommerce_preproc_with_neg_samples/" \
--subsample_first_n_sessions_by_day 10000 \
--perc_valid_set 0.9 \
--sampling_strategy "session_cooccurrence" \
--num_neg_samples 50 \
--batch_size 1000 \
--update_stats_each_n_batches 3 \
--save_each_n_batches 3 \
--sliding_windows_last_n_days 3.0 \
--recent_temporal_decay_exp_factor 0.002 # (83% of relevance in one quarter, 70% in one semester, 50% in one year and 23% in two years)