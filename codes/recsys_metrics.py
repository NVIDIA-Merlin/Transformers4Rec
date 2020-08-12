
from typing import Dict, NamedTuple
import numpy as np
import torch

from chameleon_metrics import (
    NDCG as C_NDCG, 
    HitRate as C_HitRate, 
    StreamingMetric
)
from ranking_metrics_torch_karlhigley.precision_recall import precision_at, recall_at
from ranking_metrics_torch_karlhigley.avg_precision import avg_precision_at
from ranking_metrics_torch_karlhigley.cumulative_gain import ndcg_at

from evaluation.metrics_commons import sort_topk_matrix_row_by_another_matrix
from evaluation.ranking_metrics import (
    precision_at_n as precision_at_n_cupy,
    recall_at_n as recall_at_n_cupy,
    mrr_at_n as mrr_at_n_cupy,
    map_at_n as map_at_n_cupy,
    ndcg_at_n as ndcg_at_n_cupy
)


from recsys_utils import Timing


from torch.utils.dlpack import to_dlpack
import cupy as cp


class EvalMetrics(object):
    def __init__(self, ks=[5, 10, 100, 1000], use_cpu=False, use_torch=True, use_cupy=True):
        self.use_cpu = use_cpu
        self.use_torch = use_torch
        self.use_cupy = use_cupy

        self.f_measures_cpu = []
        if use_cpu:
            f_ndcg_c = {f'ndcg_c@{k}': C_NDCG(k) for k in ks}
            f_recall_c = {f'recall_c@{k}': C_HitRate(k) for k in ks}

            self.f_measures_cpu.extend([
                f_ndcg_c, 
                f_recall_c
            ])

        self.f_measures_torch = []
        if use_torch:
            f_precision_kh = MetricWrapper('precision', precision_at, ks)
            f_recall_kh = MetricWrapper('recall', recall_at, ks)
            f_avgp_kh = MetricWrapper('avg_precision', avg_precision_at, ks)
            f_ndcg_kh = MetricWrapper('ndcg', ndcg_at, ks)
            self.f_measures_torch.extend([
                f_precision_kh,
                f_recall_kh,
                f_avgp_kh,
                f_ndcg_kh
            ])

        self.f_measures_cupy = []
        if use_cupy:
            f_precision_cp = MetricWrapperCuPy('precision_cupy', precision_at_n_cupy, ks)
            f_recall_cp = MetricWrapperCuPy('recall_cupy', recall_at_n_cupy, ks)
            f_mrr_cp = MetricWrapperCuPy('mrr_cupy', mrr_at_n_cupy, ks)
            f_map_cp = MetricWrapperCuPy('map_cupy', map_at_n_cupy, ks)
            f_ndcg_cp = MetricWrapperCuPy('ndcg_cupy', ndcg_at_n_cupy, ks)
            self.f_measures_cupy.extend([
                f_precision_cp,
                f_recall_cp,
                #f_mrr_cp,
                f_map_cp,
                f_ndcg_cp
            ])

    def update(self, preds, labels):
        if self.use_torch:
            #start_ts = time.time()
            with Timing("TORCH metrics"):
                # compute metrics on PyTorch
                for f_measure in self.f_measures_torch:
                    f_measure.add(*EvalMetrics.flatten(preds, labels))

            #elapsed_ts = time.time() - start_ts 
            #logger.info("TORCH metrics secs. %s", elapsed_ts)

        if self.use_cupy:
            #start_ts = time.time()

            with Timing("CUPY metrics"):    
                #Compute metrics on cuPy
                for f_measure in self.f_measures_cupy:
                    f_measure.add(preds, labels)

            #elapsed_ts = time.time() - start_ts 
            #logger.info("CUPY metrics secs. %s", elapsed_ts)

        if self.use_cpu:
            # compute metrics on CPU
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            for f_measure_ks in self.f_measures_cpu:
                for f_measure in f_measure_ks.values():
                    f_measure.add(preds_cpu, labels_cpu)

    def result(self):
        metrics = [] 

        #CPU
        metrics.extend([{name: f_measure.result() \
            for name, f_measure in f_measure_ks.items()} \
                for f_measure_ks in self.f_measures_cpu]
        )
        
        #PyTorch
        metrics.extend([f_measure.result() for f_measure in self.f_measures_torch])

        #cuPy
        metrics.extend([f_measure.result() for f_measure in self.f_measures_cupy])

        return {k: v for d in metrics for k, v in d.items()}

    @staticmethod
    def flatten(preds, labels):
        # flatten (n_batch x seq_len x n_events) to ((n_batch x seq_len) x n_events)
        preds = preds.view(-1, preds.size(-1))
        labels = labels.reshape(-1)
        
        return preds, labels


class MetricWrapper(object):
    def __init__(self, name, f_metric, topks):
        self.name = name
        self.topks = topks
        self.f_metric = f_metric
        self.reset()

    def reset(self):
        self.results = {k:[] for k in self.topks}

    def add(self, predictions, labels):

        # represent target class id as one-hot vector
        labels = torch.nn.functional.one_hot(labels, predictions.size(-1))

        metric = self.f_metric(torch.LongTensor(self.topks), predictions, labels)
        metric = metric.mean(0)
        for k, measure in zip(self.topks, metric):
            self.results[k].append(measure.cpu().item())

    def result(self):
        return {f'{self.name}@{k}': np.mean(self.results[k]) for k in self.topks}



class MetricWrapperCuPy(object):
    def __init__(self, name, f_metric, topks):
        self.name = name
        self.topks = topks
        self.f_metric = f_metric
        self.reset()

    def reset(self):
        self.results = {k:[] for k in self.topks}

    def add(self, predictions, labels):
        #Ensuring that label has two dimensions (as the metrics does accepts multiple labels (relevant items))
        labels = labels.view(-1,1)
        #Creating a matrix with the same shape of predictions, where each columns values are columns indices
        n_rows, n_candidates = predictions.shape
        pred_idxs = torch.arange(n_candidates).repeat(n_rows,1)

        #Temporary, for local test with Numpy
        #labels = labels.cpu().numpy()
        #predictions = predictions.cpu().numpy()
        #pred_idxs = pred_idxs.cpu().numpy()

        #Converting to cuPy
        labels = cp.fromDlpack(to_dlpack(labels))
        predictions = cp.fromDlpack(to_dlpack(predictions))
        pred_idxs = cp.fromDlpack(to_dlpack(pred_idxs))

        #Ranks top-k item positions high highest scores
        max_top_k = max(self.topks)
        
        #Creating a matrix with the same shape of predictions, where each columns values are columns indices
        #n_rows, n_candidates = predictions.shape
        #pred_idxs = cp.tile(cp.arange(n_candidates), (n_rows,1))
        topk_sorted_idxs = sort_topk_matrix_row_by_another_matrix(pred_idxs, sorting_array=predictions, topk=max_top_k)

        for topk in self.topks:
            result = self.f_metric(labels, topk_sorted_idxs, topn=topk, return_mean=True)
            self.results[topk].append(result)

    def result(self):
        return {f'{self.name}@{k}': np.mean(self.results[k]) for k in self.topks}