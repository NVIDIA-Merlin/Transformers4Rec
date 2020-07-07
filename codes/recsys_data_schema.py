"""
Set data-specific schema, vocab sizes, and feature extract function.
"""


import numpy as np
from petastorm.unischema import UnischemaField


# set vocabulary sizes for discrete input seqs. 
# NOTE: First one is the output (target) size
product_vocab_size = 300000
category_vocab_size = 60000
vocab_sizes = [product_vocab_size, category_vocab_size]

def f_feature_extract(inputs):
    """
    This function will be used inside of trainer.py (_training_step) right before being 
    passed inputs to a model. 
    For negative sampling (NS) approach
    """
    product_seq = inputs["sess_pid_seq"].long()
    category_seq = inputs["sess_ccid_seq"].long()
    neg_prod_seq = inputs["sess_neg_pids"].long()
    neg_category_seq = inputs["sess_neg_ccid"].long()
    
    return product_seq, category_seq, neg_prod_seq, neg_category_seq

# A schema that we use to read specific columns from parquet data file
recsys_schema_small = [
    UnischemaField('sess_pid_seq', np.int64, (None,), None, True),
    UnischemaField('sess_ccid_seq', np.int64, (None,), None, True),
    UnischemaField('sess_neg_pids', np.int64, (None,), None, True),
    UnischemaField('sess_neg_ccid', np.int64, (None,), None, True),
]

# Full Schema
# recsys_schema_full = [
#     UnischemaField('user_idx', np.int, (), None, True),
#     #   UnischemaField('user_session', str_, (), None, True),
#     UnischemaField('sess_seq_len', np.int, (), None, False),
#     UnischemaField('session_start_ts', np.int64, (), None, True),
#     UnischemaField('user_seq_length_bef_sess', np.int, (), None, False),
#     UnischemaField('user_elapsed_days_bef_sess', np.float, (), None, True),
#     UnischemaField('user_elapsed_days_log_bef_sess_norm', np.double, (), None, True),
#     UnischemaField('sess_pid_seq', np.int64, (None,), None, True),
#     UnischemaField('sess_etime_seq', np.int64, (None,), None, True),
#     UnischemaField('sess_etype_seq', np.int, (None,), None, True),
#     UnischemaField('sess_csid_seq', np.int, (None,), None, True),
#     UnischemaField('sess_ccid_seq', np.int, (None,), None, True),
#     UnischemaField('sess_bid_seq', np.int, (None,), None, True),
#     UnischemaField('sess_price_seq', np.float, (None,), None, True),
#     UnischemaField('sess_dtime_seq', np.float, (None,), None, True),
#     UnischemaField('sess_product_recency_seq', np.float, (None,), None, True),
#     UnischemaField('sess_relative_price_to_avg_category_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_hour_sin_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_hour_cos_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_month_sin_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_month_cos_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_dayofweek_sin_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_dayofweek_cos_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_dayofmonth_sin_seq', np.float, (None,), None, True),
#     UnischemaField('sess_et_dayofmonth_cos_seq', np.float, (None,), None, True),
#     UnischemaField('user_pid_seq_bef_sess', np.int64, (None,), None, True),
#     UnischemaField('user_etime_seq_bef_sess', np.int64, (None,), None, True),
#     UnischemaField('user_etype_seq_bef_sess', np.int, (None,), None, True),
#     UnischemaField('user_csid_seq_bef_sess', np.int, (None,), None, True),
#     UnischemaField('user_ccid_seq_bef_sess', np.int, (None,), None, True),
#     UnischemaField('user_bid_seq_bef_sess', np.int, (None,), None, True),
#     UnischemaField('user_price_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_dtime_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_product_recency_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_relative_price_to_avg_category_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_hour_sin_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_hour_cos_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_month_sin_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_month_cos_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_dayofweek_sin_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_dayofweek_cos_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_dayofmonth_sin_seq_bef_sess', np.float, (None,), None, True),
#     UnischemaField('user_et_dayofmonth_cos_seq_bef_sess', np.float, (None,), None, True),
# ]

