import cupy as cp

from .metrics_commons import is_in_2d_rowwise


def precision_at_n(relevant_item_ids, ranked_item_ids, topn, return_mean=True):
    ranked_item_ids = ranked_item_ids[:, :topn]
    ranked_items_binary_relevance = is_in_2d_rowwise(ranked_item_ids, in_matrix=relevant_item_ids)
    precision_by_row = ranked_items_binary_relevance.mean(axis=1)
    result = float(cp.mean(precision_by_row)) if return_mean else precision_by_row
    return result


def recall_at_n(relevant_item_ids, ranked_item_ids, topn, return_mean=True):
    ranked_items_binary_relevance = is_in_2d_rowwise(ranked_item_ids, in_matrix=relevant_item_ids)
    relevant_items_by_row = ranked_items_binary_relevance.sum(axis=1)
    recall_by_row = ranked_items_binary_relevance[:, :topn].sum(axis=1) / relevant_items_by_row
    # Filling cases where the number of relevant items is zero
    recall_by_row[cp.isnan(recall_by_row)] = 0
    result = float(cp.mean(recall_by_row)) if return_mean else recall_by_row
    return result


def mrr_at_n(relevant_item_ids, ranked_item_ids, topn, return_mean=True):
    def _mrr(ranked_items_binary_relevance):
        DEFAULT_VALUE_FOR_OUT_OF_TOP_N = (
            -2
        )  # (Must be a negative value (so that we cut the MRR calculation at topn) and not -1, because when summed with 1 leads to zero division)
        first_relevant_items_idx = cp.argmax(ranked_items_binary_relevance, axis=-1)
        relevant_items_available = cp.max(ranked_items_binary_relevance, axis=-1)
        first_relevant_items_idx_fixed = cp.where(
            relevant_items_available == 1, first_relevant_items_idx, DEFAULT_VALUE_FOR_OUT_OF_TOP_N
        )
        reciprocal_rank = 1 / (first_relevant_items_idx_fixed + 1)
        reciprocal_rank_fixed = cp.where(reciprocal_rank < 0, 0, reciprocal_rank)
        mrr_results = reciprocal_rank_fixed
        return mrr_results

    ranked_item_ids = ranked_item_ids[:, :topn]
    ranked_items_binary_relevance = is_in_2d_rowwise(ranked_item_ids, in_matrix=relevant_item_ids)
    mrr_results = _mrr(ranked_items_binary_relevance)
    result = float(cp.mean(mrr_results)) if return_mean else mrr_results
    return result


def map_at_n(relevant_item_ids, ranked_item_ids, topn, return_mean=True):
    def _map_at_n(ranked_items_binary_relevance):
        relevant_items_cumsum = ranked_items_binary_relevance.cumsum(axis=1)
        ranking_pos = cp.arange(0, ranked_items_binary_relevance.shape[-1]) + 1
        prec_at_k = relevant_items_cumsum / ranking_pos
        prec_at_k_only_relevant = prec_at_k * ranked_items_binary_relevance
        relevant_items_by_row = ranked_items_binary_relevance.sum(axis=1)
        ap_by_row = prec_at_k_only_relevant.sum(axis=1) / relevant_items_by_row
        # Filling cases where idcg is 0 (inf or NaN after div by zero) to zero
        ap_by_row[cp.isinf(ap_by_row) | cp.isnan(ap_by_row)] = 0
        return ap_by_row

    ranked_item_ids = ranked_item_ids[:, :topn]
    ranked_items_binary_relevance = is_in_2d_rowwise(ranked_item_ids, in_matrix=relevant_item_ids)
    ap_by_row = _map_at_n(ranked_items_binary_relevance)
    result = float(cp.mean(ap_by_row)) if return_mean else ap_by_row
    return result


def ndcg_at_n(relevant_item_ids, ranked_item_ids, topn, return_mean=True):
    def _ndcg_at_n(r, k):
        # Based on https://gist.github.com/bwhite/3726239, but with alternative formulation of DCG
        # which places stronger emphasis on retrieving relevant documents (used in Kaggle and also described in Wikipedia): https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
        def _dcg_at_n(r, k):
            r_top = r[:, :k]
            return cp.sum(
                (cp.power(2, r_top) - 1) / cp.log2(cp.arange(2, r_top.shape[1] + 2)), axis=1
            )

        perfect_rev_sorting = cp.flip(cp.sort(r, axis=1), axis=1)
        idcg = _dcg_at_n(perfect_rev_sorting, k)
        dcg = _dcg_at_n(r, k)
        ndcg_results = dcg / idcg
        # Filling cases where idcg is 0 (inf or NaN after div by zero) to zero
        ndcg_results[cp.isinf(ndcg_results) | cp.isnan(ndcg_results)] = 0
        results = ndcg_results
        return results

    ranked_items_binary_relevance = is_in_2d_rowwise(ranked_item_ids, in_matrix=relevant_item_ids)
    ndcgs = _ndcg_at_n(ranked_items_binary_relevance, topn)
    result = float(cp.mean(ndcgs)) if return_mean else ndcgs
    return result
