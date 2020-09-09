
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

from recsys_utils import Timing
from torch.utils.dlpack import to_dlpack

class EvalMetrics(object):
    def __init__(self, ks=[5, 10, 100, 1000], use_cpu=False, use_torch=True, use_cupy=False):
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
            from evaluation.metrics_commons import sort_topk_matrix_row_by_another_matrix
            from evaluation.ranking_metrics import (
                precision_at_n_binary as precision_at_n_cupy,
                recall_at_n_binary as recall_at_n_cupy,
                mrr_at_n_binary as mrr_at_n_cupy,
                map_at_n_binary as map_at_n_cupy,
                ndcg_at_n_binary as ndcg_at_n_cupy
            )

            f_precision_cp = MetricWrapperCuPy('precision_cupy', precision_at_n_cupy, ks)
            f_recall_cp = MetricWrapperCuPy('recall_cupy', recall_at_n_cupy, ks)
            f_mrr_cp = MetricWrapperCuPy('mrr_cupy', mrr_at_n_cupy, ks)
            f_map_cp = MetricWrapperCuPy('map_cupy', map_at_n_cupy, ks)
            f_ndcg_cp = MetricWrapperCuPy('ndcg_cupy', ndcg_at_n_cupy, ks)
            self.f_measures_cupy.extend([
                #f_precision_cp,
                #f_recall_cp,
                f_mrr_cp,
                #f_map_cp,
                #f_ndcg_cp
            ])

    def update(self, preds, labels, return_individual_metrics=False):
        metrics_results = {}
        if self.use_torch:
            #with Timing("TORCH metrics"):
                # compute metrics on PyTorch
                for f_measure in self.f_measures_torch:
                    results = f_measure.add(*EvalMetrics.flatten(preds, labels), return_individual_metrics=return_individual_metrics)  
                    # Merging metrics results
                    if return_individual_metrics:
                        metrics_results = {**metrics_results, **results}                  

        if self.use_cupy:
            #with Timing("CUPY metrics"):    
                #Compute metrics on cuPy
                for f_measure in self.f_measures_cupy:
                    metrics_results = f_measure.add(preds, labels, return_individual_metrics=return_individual_metrics)
                    # Merging metrics results
                    if return_individual_metrics:
                        metrics_results = {**metrics_results, **results}

        if self.use_cpu:
            # compute metrics on CPU
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            for f_measure_ks in self.f_measures_cpu:
                for f_measure in f_measure_ks.values():
                    f_measure.add(preds_cpu, labels_cpu)

        return metrics_results

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

    def add(self, predictions, labels, return_individual_metrics=False):

        # represent target class id as one-hot vector
        labels = torch.nn.functional.one_hot(labels, predictions.size(-1))

        #Computing the metrics at different cut-offs
        metric = self.f_metric(torch.LongTensor(self.topks), predictions, labels)

        #Retrieving individual metric results (for each next-item recommendation list), to return for debug logging purposes
        if return_individual_metrics:
            returns = {}
            for k, measures in zip(self.topks, metric.T):
                returns[f'{self.name}@{k}'] = measures.cpu().numpy()

        #Computing the mean of the batch metrics (for each cut-off at topk)
        metric_mean = metric.mean(0)

        #Storing in memory the average metric results for this batch
        for k, measure in zip(self.topks, metric_mean):
            self.results[k].append(measure.cpu().item())

        if return_individual_metrics:
            #Returning individual metric results, for debug logging purposes
            return returns
            
    def result(self):
        return {f'{self.name}@{k}': np.mean(self.results[k]) for k in self.topks}



class MetricWrapperCuPy(object):
    def __init__(self, name, f_metric, topks):
        self.name = name
        self.topks = topks
        self.f_metric = f_metric
        self.reset()

        self.import_cupy()

    def import_cupy(self):
        import cupy as cp
        self.cp = cp

    def reset(self):
        self.results = {k:[] for k in self.topks}

    def add(self, predictions, labels, return_individual_metrics=False):
        # represent target class id as one-hot vector
        labels_ohe = torch.nn.functional.one_hot(labels, predictions.size(-1))

        #For local debugging with Numpy
        #labels_ohe = labels_ohe.cpu().numpy()
        #predictions = predictions.cpu().numpy()

        #Converting to cuPy
        labels_ohe = self.cp.fromDlpack(to_dlpack(labels_ohe))
        predictions = self.cp.fromDlpack(to_dlpack(predictions))        


        #Ranks top-k item positions high highest scores
        max_top_k = max(self.topks)
        
        #Sorting labels by the predicted score
        labels_ohe_ranked = sort_topk_matrix_row_by_another_matrix(labels_ohe, sorting_array=predictions, topk=max_top_k)

        returns = {}
        for k in self.topks:
            detailed_result = self.f_metric(labels_ohe_ranked, topn=k, return_mean=False)
            if return_individual_metrics:
                returns[f'{self.name}@{k}'] = self.cp.asnumpy(detailed_result)

            result = detailed_result.mean()
            self.results[k].append(result)

        if return_individual_metrics:
            #Returning individual metric results, for debug logging purposes
            return returns

    def result(self):
        return {f'{self.name}@{k}': np.mean(self.results[k]) for k in self.topks}