"""
Set data-specific schema, vocab sizes, and feature extract function.
"""
 
import math
from datetime import datetime
from datetime import date, timedelta

import numpy as np
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader as PetaStormDataLoader
from petastorm.unischema import UnischemaField
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader

from recsys_utils import get_filenames


# set vocabulary sizes for discrete input seqs. 
# NOTE: First one is the output (target) size


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, cols_to_read):
        self.cols_to_read = cols_to_read
        self.data = pq.ParquetDataset(parquet_file).read(columns=self.cols_to_read).to_pandas()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        df = self.data.loc[index]
        return {col: df[col] for col in df.index}


class DataLoaderWrapper(PyTorchDataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderWrapper, self).__init__(*args, **kwargs)

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        return None


def get_avail_data_dates(data_args, date_format="%Y-%m-%d"):
    start_date, end_date = data_args.start_date, data_args.end_date
    end_date = datetime.strptime(end_date, date_format)
    start_date = datetime.strptime(start_date, date_format)

    delta = end_date - start_date

    avail_dates = []
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        avail_dates.append(day)

    return avail_dates


def get_dataset_len(data_paths):
    return sum([pq.ParquetFile(d_path).metadata.num_rows for d_path in data_paths])


def fetch_data_loaders(data_args, training_args, train_date, eval_date, test_date=None,
                       neg_sampling=False, date_format="%Y-%m-%d", load_from_path=False):
    """
    load_from_path: when a path is given, it automatically determines 
                    list of available parquet files in the path.
                    otherwise, the path should be a direct path to the parquet file 
    """
    d_path = data_args.data_path if data_args.data_path else ''
    
    # TODO: make this at outer-loop for making evaluation based on days-data-partition
    train_data_path = [
        d_path + "session_start_date={}-train.parquet".format(train_date.strftime(date_format)),
    ]

    eval_data_path = [
        d_path + "session_start_date={}-test.parquet".format(eval_date.strftime(date_format)),
    ]

    if test_date is not None:
        test_data_path = [
            d_path + "session_start_date={}.parquet".format(test_date.strftime(date_format)),
        ]

    if load_from_path:
        train_data_path = get_filenames(train_data_path)
        eval_data_path = get_filenames(eval_data_path)

    if data_args.engine == "petastorm":
        parquet_schema = parquet_schema_posneg if neg_sampling else parquet_schema_pos

        train_data_len = get_dataset_len(train_data_path)
        eval_data_len = get_dataset_len(eval_data_path)

        train_data_path = ['file://' + p for p in train_data_path]
        eval_data_path = ['file://' + p for p in eval_data_path]
        if test_date is not None:
            test_data_path = ['file://' + p for p in test_data_path]

        train_loader = DataLoaderWithLen(
            make_batch_reader(train_data_path, 
                num_epochs=None,
                schema_fields=parquet_schema,
                reader_pool_type=data_args.reader_pool_type,
                workers_count=data_args.workers_count,
            ), 
            batch_size=training_args.per_device_train_batch_size,
            len=math.ceil(train_data_len / training_args.per_device_train_batch_size),
        )

        eval_loader = DataLoaderWithLen(
            make_batch_reader(eval_data_path, 
                num_epochs=None,
                schema_fields=parquet_schema,
                reader_pool_type=data_args.reader_pool_type,
                workers_count=data_args.workers_count,
            ), 
            batch_size=training_args.per_device_eval_batch_size,
            len=math.ceil(eval_data_len / training_args.per_device_eval_batch_size),
        )

        if test_date is not None:
            test_loader = DataLoaderWithLen(
                make_batch_reader(test_data_path, 
                    num_epochs=None,
                    schema_fields=parquet_schema,
                    reader_pool_type=data_args.reader_pool_type,
                    workers_count=data_args.workers_count,
                ), 
                batch_size=training_args.per_device_eval_batch_size,
                len=math.ceil(eval_data_len / training_args.per_device_eval_batch_size),
            )

    elif data_args.engine == "pyarrow":
        cols_to_read = parquet_col_posneg if neg_sampling else parquet_col_pos

        train_dataset = ParquetDataset(train_data_path, cols_to_read)
        train_loader = DataLoaderWrapper(train_dataset, batch_size=training_args.per_device_train_batch_size)
        eval_dataset = ParquetDataset(eval_data_path, cols_to_read)
        eval_loader = DataLoaderWrapper(eval_dataset, batch_size=training_args.per_device_eval_batch_size)

        if test_date is not None:
            test_dataset = ParquetDataset(test_data_path, cols_to_read)
            test_loader = DataLoaderWrapper(test_dataset, batch_size=training_args.per_device_eval_batch_size)
    
    if test_date is None:
        test_loader = None

    return train_loader, eval_loader, test_loader


class DataLoaderWithLen(PetaStormDataLoader):
    def __init__(self, *args, **kwargs):
        if 'len' not in kwargs:
            self.len = 0
        else:
            self.len = kwargs.pop('len')

        super(DataLoaderWithLen, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.len


def f_feature_extract_posneg(inputs):
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


def f_feature_extract_pos(inputs):
    """
    This function will be used inside of trainer.py (_training_step) right before being 
    passed inputs to a model. 
    For negative sampling (NS) approach
    """
    product_seq = inputs["sess_pid_seq"].long()
    category_seq = inputs["sess_ccid_seq"].long()
    
    return product_seq, category_seq


# (PyArrow) Columns to read 
parquet_col_posneg = ['sess_pid_seq', 'sess_ccid_seq', 'sess_neg_pids', 'sess_neg_ccid']
parquet_col_pos = ['sess_pid_seq', 'sess_ccid_seq']


# (Petastorm) A schema that we use to read specific columns from parquet data file
parquet_schema_posneg = [
    UnischemaField('sess_pid_seq', np.int64, (None,), None, True),
    UnischemaField('sess_ccid_seq', np.int64, (None,), None, True),
    UnischemaField('sess_neg_pids', np.int64, (None,), None, True),
    UnischemaField('sess_neg_ccid', np.int64, (None,), None, True),
]

parquet_schema_pos = [
    UnischemaField('sess_pid_seq', np.int64, (None,), None, True),
    UnischemaField('sess_ccid_seq', np.int64, (None,), None, True),
]



# Full Schema
# parquet_schema_full = [
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

