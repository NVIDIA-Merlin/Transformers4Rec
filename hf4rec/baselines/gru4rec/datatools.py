# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:20:08 2020

@author: Hidasi Balázs
"""

import time
import numpy as np
import pandas as pd

def sort_if_needed(data, columns, any_order_first_dim=False):
    is_sorted = True
    neq_masks = []
    for i, col in enumerate(columns):
        dcol = data[col]
        neq_masks.append(dcol.values[1:]!=dcol.values[:-1])
        if i == 0:
            if any_order_first_dim:
                is_sorted = is_sorted and (dcol.nunique() == neq_masks[0].sum() + 1)
            else:
                is_sorted = is_sorted and np.all(dcol.values[1:] >= dcol.values[:-1])
        else:
            is_sorted = is_sorted and np.all(neq_masks[i - 1] | (dcol.values[1:] >= dcol.values[:-1]))
        if not is_sorted:
            break
    if is_sorted:
        print('The dataframe is already sorted by {}'.format(', '.join(columns)))
    else:
        print('The dataframe is not sorted by {}, sorting now'.format(col))
        t0 = time.time()
        data.sort_values(columns, inplace=True)
        t1 = time.time()
        print('Data is sorted in {:.2f}'.format(t1 - t0))

def compute_offset(data, column):
    offset = np.zeros(data[column].nunique() + 1, dtype=np.int32)
    offset[1:] = data.groupby(column).size().cumsum()
    return offset
