"""
Set data-specific schema, vocab sizes, and feature extract function.
"""
 
import os
import math
from datetime import datetime
from datetime import date, timedelta
from random import randint

import logging

import numpy as np

#Petastorm data loader dependencies
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader as PetaStormDataLoader
from petastorm.unischema import UnischemaField

#Pyarrow dependencies
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, DataLoader as PyTorchDataLoader
import torch


from recsys_utils import get_filenames


logger = logging.getLogger(__name__)



# set vocabulary sizes for discrete input seqs. 
# NOTE: First one is the output (target) size


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, cols_to_read, seq_features_len_pad_trim):
        self.cols_to_read = cols_to_read
        self.data = pq.ParquetDataset(parquet_file).read(columns=self.cols_to_read).to_pandas()
        self.seq_features_len_pad_trim = seq_features_len_pad_trim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        df = self.data.loc[index]
        return {col: self.pad_seq_column_if_needed(df[col]) for col in df.index}

    def pad_seq_column_if_needed(self, values):
        if type(values) is np.ndarray:
            values = values[:self.seq_features_len_pad_trim]
            if len(values) < self.seq_features_len_pad_trim:
                placeholder = np.zeros(self.seq_features_len_pad_trim, dtype=values.dtype)
                placeholder[:len(values)] = values
                values = placeholder            
            if isinstance(values[0], np.floating) and values.dtype is not np.float32:
                values = values.astype(np.float32)
            if isinstance(values[0], np.integer) and values.dtype is not np.int64:
                values = values.astype(np.int64)
        return values


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


def fetch_data_loader(data_args, training_args, feature_map, data_paths, is_train_set,
                      shuffle_dataloader=False):

    if type(data_paths) is not list:
        data_paths = [data_paths]

    batch_size = training_args.per_device_train_batch_size if is_train_set else training_args.per_device_eval_batch_size

    if data_args.data_loader_engine == "petastorm":
        #eval_data_path = transform_petastorm_schema(feature_map)

        data_len = get_dataset_len(data_paths)

        data_paths = ['file://' + p for p in data_paths]

        
        loader = DataLoaderWithLen(
            make_batch_reader(data_paths, 
                num_epochs=None,
                schema_fields=parquet_schema,
                reader_pool_type=data_args.petastorm_reader_pool_type,
                workers_count=data_args.workers_count,
            ), 
            batch_size=batch_size,
            len=math.ceil(data_len / batch_size),
        )


    elif data_args.data_loader_engine == "pyarrow":
        cols_to_read = feature_map.keys()

        dataset = ParquetDataset(data_paths, cols_to_read, seq_features_len_pad_trim=data_args.session_seq_length_max)
        if shuffle_dataloader and training_args.shuffle_buffer_size > 0:
            dataset = ShuffleDataset(dataset, buffer_size=training_args.shuffle_buffer_size)
        loader = PyTorchDataLoader(dataset, batch_size=batch_size, 
                                        drop_last=training_args.dataloader_drop_last,                                         
                                        num_workers=data_args.workers_count,
                                        pin_memory=True)

    elif data_args.data_loader_engine == "nvtabular":

        #NVTabular dependencies
        from nvtabular.loader.torch import TorchAsyncItr as NVTDataLoader
        from nvtabular import Dataset as NVTDataset

        class NVTDatasetWrapper(NVTDataset):

            def __init__(self, *args, **kwargs):
                super(NVTDatasetWrapper, self).__init__(*args, **kwargs)

            def __len__(self):
                return self.num_rows

        class NVTDataLoaderWrapper(NVTDataLoader):
            def __init__(self, *args, **kwargs):
                if 'seq_features_len_pad_trim' in kwargs:
                    self.seq_features_len_pad_trim = kwargs.pop('seq_features_len_pad_trim')
                else:
                    raise ValueError('NVTabular data loader requires the "seq_features_len_pad_trim" argument "'+\
                                     'to create the sparse tensors for list columns')                
                self.dataset = kwargs['dataset']
                super(NVTDataLoaderWrapper, self).__init__(*args, **kwargs)

            def __len__(self):
                #TODO: The argument drop_last should be added to the NVTDataLoader (https://github.com/NVIDIA/NVTabular/issues/470), and instead of subtracting one step it should do len(dataset) // batch_size, to deal with cases when the length is multiple of batch size
                length = super(NVTDataLoaderWrapper, self).__len__()
                if training_args.dataloader_drop_last:
                     length -= 1
                return length

            def __next__(self):
                cat_features, cont_features, label_features = super(NVTDataLoaderWrapper, self).__next__()

                #TODO: This code is an uggly workaround for this bug on NVT 0.3 data loader (https://github.com/NVIDIA/NVTabular/issues/513), just to ignore the "incomplete" batch, which turns out the be the first one in the second iteration over the dataloader
                if training_args.dataloader_drop_last:
                    if cat_features is not None:
                        batch_size = cat_features[1][list(cat_features[1].keys())[0]][1].shape[0]
                        if batch_size != self.batch_size:
                            cat_features, cont_features, label_features = super(NVTDataLoaderWrapper, self).__next__()
                
                
                cat_sequence_features_transf = {}
                cont_sequence_features_transf = {}
                if cat_features is not None:
                    cat_single_features, cat_sequence_features = cat_features
                    cat_sequence_features_transf = {fname: self.get_sparse_tensor_list_column(cat_sequence_features[fname], 
                                                                                         'categorical').to_dense()[:, :self.seq_features_len_pad_trim] \
                                                for fname in cat_sequence_features}

                if cont_features is not None:
                    cont_single_features, cont_sequence_features = cont_features

                    cont_sequence_features_transf = {fname: self.get_sparse_tensor_list_column(cont_sequence_features[fname], 
                                                                                            'continuous').to_dense()[:, :self.seq_features_len_pad_trim] \
                                                    for fname in cont_sequence_features}

                '''
                #Reconstructing sequential feature tensors assuming they have the same length for all rows
                cat_sequence_features_transf = {}
                if cat_features is not None:
                    cat_sequence_features_transf = {k: v[0].reshape(-1, self.default_seq_features_len) for k, v in cat_features[1].items()}
                cont_sequence_features_transf = {}
                if cont_features is not None:
                    cont_sequence_features_transf = {k: v[0].reshape(-1, self.default_seq_features_len) for k, v in cont_features[1].items()}
                '''

                inputs = {**cat_sequence_features_transf, **cont_sequence_features_transf}
                return inputs

            def get_sparse_tensor_list_column(self, values_offset, feature_group):
                values = values_offset[0].flatten()
                offsets = values_offset[1].flatten()
                num_rows = len(offsets)

                #Appending the values length to the end of the offset vector, to be able to compute diff of the last sequence
                offsets = torch.cat([offsets, torch.LongTensor([len(values)]).to(offsets.device)])
                #Computing the difference between consecutive offsets, to get the sequence lengths
                diff_offsets = offsets[1:] - offsets[:-1]
                #Infering the number of cols based on the maximum sequence length
                max_seq_len = int(diff_offsets.max())

                if max_seq_len > self.seq_features_len_pad_trim:
                    logger.warn('The default sequence length has been configured to {}, '
                                     'but the largest sequence in this batch have {} length. Truncating to {}.' \
                                        .format(self.seq_features_len_pad_trim,
                                                max_seq_len, self.seq_features_len_pad_trim))


                #Building the indices to reconstruct the sparse tensors
                row_ids = torch.arange(len(offsets)-1).to(offsets.device)
                row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
                row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
                col_ids = torch.arange(len(row_offset_repeated)).to(offsets.device) - row_offset_repeated.to(offsets.device)
                indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)

                if feature_group == 'categorical':
                    sparse_tensor_class = torch.sparse.LongTensor
                elif feature_group == 'continuous':
                    sparse_tensor_class = torch.sparse.FloatTensor
                else:
                    raise NotImplementedError('Invalid feature group from NVTabular: {}'.format(feature_group))

                sparse_tensor = sparse_tensor_class(indices.T, values, torch.Size([num_rows, max(max_seq_len, self.seq_features_len_pad_trim)]))                
                return sparse_tensor

        
        categ_features = []
        continuous_features = []
        for fname, fprops in feature_map.items():
            if fprops['dtype'] == 'categorical':
                categ_features.append(fname)
            elif fprops['dtype'] in ['float', 'long']:
                continuous_features.append(fname)
            else:
                raise NotImplementedError("The dtype {} is not currently supported.".format(fprops['dtype']))

        data_loader_config = {
                "cats": categ_features,
                "conts": continuous_features,
                "labels": [],
                "devices": list(range(training_args.n_gpu)),
            }

        dataset = NVTDatasetWrapper(data_paths, engine="parquet", part_mem_fraction=data_args.nvt_part_mem_fraction)
        loader = NVTDataLoaderWrapper(dataset=dataset, seq_features_len_pad_trim=data_args.session_seq_length_max, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle_dataloader, **data_loader_config)
    return loader

class DataLoaderWithLen(PetaStormDataLoader):
    def __init__(self, *args, **kwargs):
        if 'len' not in kwargs:
            self.len = 0
        else:
            self.len = kwargs.pop('len')

        super(DataLoaderWithLen, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.len


def transform_petastorm_schema(feature_map):
    schema = []
    for cname, cinfo in feature_map.items():
        if cinfo['dtype'] in ['categorical', 'int']:
            dtype = np.int64
        elif cinfo['dtype'] == 'float':
            dtype = np.float
        schema.append(UnischemaField(cname, dtype, (None,), None, True))
    return schema


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):

        logger.info('[SHUFFLE] INITIALIZING BUFFER_SIZE: {}'.format(self.buffer_size))

        raise StopIteration()

        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            logger.info('[SHUFFLE] RETRIEVING FROM BUFFER AND REPLACING FROM ITERATOR: {}'.format(len(shufbuf)))
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration as e:
                    logger.info('[SHUFFLE] StopIteration EXCEPTION: {}', e)
                    break

            logger.info('[SHUFFLE] STARTING TO RETRIEVE ONLY FROM BUFFER: {}'.format(len(shufbuf)))

            while len(shufbuf) > 0:
                yield shufbuf.pop()

            logger.info('[SHUFFLE] FINISHED ITERATING')

        except GeneratorExit:
            pass

    def __len__(self):
        return len(self.dataset)