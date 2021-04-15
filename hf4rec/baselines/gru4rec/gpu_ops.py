# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:17:58 2017

@author: Balázs Hidasi
"""

import numpy as np
import theano
from theano import tensor as T

from . import custom_theano_ops as cto

disable_custom_op = False


def gpu_diag_wide(X, keepdims=False):
    E = T.eye(*X.shape)
    return T.sum(X * E, axis=1, keepdims=keepdims)


def gpu_diag_tall(X, keepdims=False):
    E = T.eye(*X.shape)
    return T.sum(X * E, axis=0, keepdims=keepdims)


def gpu_diag(X, keepdims=False, disable_custom_op=disable_custom_op):
    if disable_custom_op:
        return T.switch(
            T.gt(X.shape[0], X.shape[1]),
            gpu_diag_tall(X, keepdims),
            gpu_diag_wide(X, keepdims),
        )
    else:
        return cto.GpuExtractDiag2D(keepdims=keepdims)(X)


def gpu_searchsorted_step(A, B, X, P):
    I = (A + B) // 2
    PI = P[I]
    return A * (X < PI) + (I + 1) * (X >= PI), B * (X > PI) + I * (X <= PI)


def gpu_searchsorted_scan(P, X):
    N = T.cast(T.floor(T.log2(P.shape[0])) + 1, "int64")
    (_, B), _ = theano.scan(
        gpu_searchsorted_step,
        outputs_info=[
            T.zeros_like(X, dtype="int64"),
            T.ones_like(X, dtype="int64") * (P.shape[0] - 1),
        ],
        non_sequences=[X, P],
        n_steps=N,
        allow_gc=True,
    )
    return B[-1]


def gpu_searchsorted(P, X, dtype_int64=True, disable_custom_op=disable_custom_op):
    if disable_custom_op:
        return gpu_searchsorted_scan(P, X)
    else:
        return cto.GpuBinarySearchSorted(dtype_int64=dtype_int64)(P, X)
