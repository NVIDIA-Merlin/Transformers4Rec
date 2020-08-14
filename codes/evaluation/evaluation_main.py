import argparse
import gc
import glob
import os
import sys
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from tqdm import tqdm
import math
import json

import cupy as cp

from .metrics_commons import sort_topk_matrix_row_by_another_matrix
from .ranking_metrics import map_at_n, mrr_at_n, ndcg_at_n, precision_at_n, recall_at_n


def args_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate metrics provided by a recommender algorithm."
    )
    # Inputs
    parser.add_argument(
        "--input_recs_parquet_path_pattern",
        type=str,
        help="Path to look for parquet files with recommendations provided by a recommendation algorithm (accepts *)."
        + "Each row in this file must be a recommendation list",
    )
    parser.add_argument(
        "--input_col_name_relevant_item_ids",
        type=str,
        default="relevant_item_ids",
        help="Name of the column (in the input Parquet file) with the relevant items for a given recommendation list",
    )
    parser.add_argument(
        "--input_col_name_rec_item_ids",
        type=str,
        default="rec_item_ids",
        help="Name of the column (in the input Parquet file) with the items of a given recommendation list.\n"
        + "The items can have been ranked by the recommendation algorithm, or they can be ranked during evaluation by the corresponding "
        + "recommendation scores",
    )
    parser.add_argument(
        "--input_col_name_rec_item_scores",
        type=str,
        default="rec_item_scores",
        help="Name of the column (in the input Parquet file) with the recommendation scores for the items of a given recommendation list.\n"
        + "This column can be used to rank the corresponding items at the 'input_col_name_rec_item_ids' column",
    )
    # Output
    parser.add_argument(
        "--output_parquet_path",
        type=str,
        help="Output path of a parquet file where the computed metrics will be saved for each recommendation list",
    )
    # Evaluation config
    parser.add_argument(
        "--recommendation_task",
        type=str,
        default=0,
        help="Recommendation Task: ranking|binary_classification",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="List of metrics separated by comma. Options for Ranking task: precision,recall,mrr,map,ndcg",
    )
    parser.add_argument(
        "--rank_items_by_rec_score",
        action="store_true",
        help="For the 'ranking' recommendation task, Indicates whether the recommended items (input_col_name_rec_item_ids) should be ranked by the corresponding recommendation scores (input_col_name_rec_item_scores)",
    )
    parser.add_argument(
        "--use_ranks_as_item_ids",
        action="store_true",
        help="For the 'ranking' recommendation task, Indicates whether the recommended items (input_col_name_rec_item_ids) should be ranked by the corresponding recommendation scores (input_col_name_rec_item_scores)",
    )
    parser.add_argument(
        "--top_n",
        type=str,
        help="For the 'ranking' recommendation task, computes metric values up to this position in the recommendation list. Accepts a single N, or a list of Ns",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Number of rows (recommendation lists) to compute evaluation metrics each time. This might be important if you have a GPU with limited memory",
    )

    return parser

ITEM_ID_PADDING_VALUE = -1

# Internal columns that will be created in the data frame
RANK_RELEVANT_ITEMS_COUNT_COL = "_relevant_items_count"
RANK_RELEVANT_ITEM_IDS_PADDED_COL = "_relevant_items_padded"

METRICS_RESULTS_FILE = 'results.txt'

class MetricsResults:

    def __init__(self):
        self.metrics_chunks = []

    def add(self, metrics):
        metrics_avg = {k: v.mean() for k, v in metrics.items()}
        self.metrics_chunks.append(metrics_avg)

    def result(self):
        if len(self.metrics_chunks) > 0:
            results = pd.DataFrame(self.metrics_chunks).mean(axis=0).to_dict()
        else:
            results = {}
        return results



def load_rec_file(path):
    df = pd.read_parquet(path)
    return df


def compute_metrics(recs_df, args):
    if args.recommendation_task == "ranking":
        metrics_results = compute_ranking_metrics(recs_df, args)
    elif args.recommendation_task == "binary_classification":
        raise NotImplementedError("Binary classification metrics are not implemented yet")
    else:
        raise ValueError(
            "Invalid recommendation task. Valid choices ranking|binary_classification"
        )

    return metrics_results


def rpad(alist, size, fill_value):
    alist = list(alist)
    missing_items = size - len(alist)
    if missing_items > 0:
        result = alist + (missing_items * [fill_value])
    else:
        result = alist
    return result


def _create_relevant_items_length_col(recs_df, args):
    recs_df[RANK_RELEVANT_ITEMS_COUNT_COL] = recs_df[args.input_col_name_relevant_item_ids].apply(
        len
    )


def _pad_relevant_items(recs_df, args):
    relev_items_max_length = recs_df[RANK_RELEVANT_ITEMS_COUNT_COL].max()
    if relev_items_max_length > 1:
        recs_df[RANK_RELEVANT_ITEM_IDS_PADDED_COL] = recs_df[
            args.input_col_name_relevant_item_ids
        ].apply(lambda l: rpad(l, relev_items_max_length, ITEM_ID_PADDING_VALUE))
    else:
        recs_df[RANK_RELEVANT_ITEM_IDS_PADDED_COL] = recs_df[args.input_col_name_relevant_item_ids]


def _get_ranked_item_ids_by_rec_score(recs_df, rec_item_ids_cp, rec_item_scores_cp, max_top_n, args):
    ranked_rec_item_ids = sort_topk_matrix_row_by_another_matrix(
        rec_item_ids_cp, sorting_array=rec_item_scores_cp, topk=max_top_n
    )
    return ranked_rec_item_ids


def _compute_ranking_topn_metrics(relevant_item_ids, ranked_rec_item_ids, top_n_list):

    metrics_results = dict()
    for topn in top_n_list:

        metrics_results["precision@{}".format(topn)] = precision_at_n(
            relevant_item_ids, ranked_rec_item_ids, topn=topn, return_mean=False
        )
        metrics_results["recall@{}".format(topn)] = recall_at_n(
            relevant_item_ids, ranked_rec_item_ids, topn=topn, return_mean=False
        )
        metrics_results["mrr@{}".format(topn)] = mrr_at_n(
            relevant_item_ids, ranked_rec_item_ids, topn=topn, return_mean=False
        )
        metrics_results["map@{}".format(topn)] = map_at_n(
            relevant_item_ids, ranked_rec_item_ids, topn=topn, return_mean=False
        )
        metrics_results["ndcg@{}".format(topn)] = ndcg_at_n(
            relevant_item_ids, ranked_rec_item_ids, topn=topn, return_mean=False
        )

    return metrics_results


def _add_metrics_results_to_dataframe(recs_df, metrics_results):
    for k in metrics_results:
        recs_df[k] = metrics_results[k]


def _log_metrics_summary(metrics_results):
    for k in metrics_results:
        print("{} = {}".format(k, metrics_results[k].mean()))


def compute_ranking_metrics(recs_df, args):
    _create_relevant_items_length_col(recs_df, args)
    _pad_relevant_items(recs_df, args)

    relevant_item_ids_cp = cp.asarray(cp.vstack(recs_df[RANK_RELEVANT_ITEM_IDS_PADDED_COL]))

    rec_item_scores_cp = None
    if args.rank_items_by_rec_score:
        rec_item_scores_cp = cp.asarray(cp.vstack(recs_df[args.input_col_name_rec_item_scores]))

    if args.use_ranks_as_item_ids:
        if not args.rank_items_by_rec_score:
            raise ValueError('To use ranks as item ids, the --rank_items_by_rec_score option should be provided')
        rec_item_ids_cp = cp.tile(cp.arange(rec_item_scores_cp.shape[1]), (rec_item_scores_cp.shape[0], 1))
    else:
        rec_item_ids_cp = cp.asarray(cp.vstack(recs_df[args.input_col_name_rec_item_ids]))

    top_n_list = list([int(n) for n in args.top_n.split(",")])
    max_top_n = max(top_n_list)

    if args.rank_items_by_rec_score:
        ranked_rec_item_ids = _get_ranked_item_ids_by_rec_score(
            recs_df, rec_item_ids_cp, rec_item_scores_cp, max_top_n, args
        )
    else:
        ranked_rec_item_ids = rec_item_ids_cp

    metrics_results_dict = _compute_ranking_topn_metrics(
        relevant_item_ids_cp, ranked_rec_item_ids, top_n_list
    )
    return metrics_results_dict


def process_dataframe_into_chuncks_generator(df, chunk_size):
    number_chunks = int(math.ceil(len(df) / chunk_size))
    for i in range(number_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        yield df[start_idx:end_idx]
    del df
    gc.collect()



def create_pq_writer(new_rows_df, path):
    new_rows_pa = pyarrow.Table.from_pandas(new_rows_df)
    # Creating parent folder recursively
    parent_folder = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    # Creating parquet file
    pq_writer = pq.ParquetWriter(path, new_rows_pa.schema)
    return pq_writer


def append_new_rows_to_parquet(new_rows_df, pq_writer):
    new_rows_pa = pyarrow.Table.from_pandas(new_rows_df)
    pq_writer.write_table(new_rows_pa)
    return pq_writer


def main():
    parser = args_parser()
    args = parser.parse_args()

    input_parquet_files = sorted(
        glob.glob(args.input_recs_parquet_path_pattern.replace("'", "") + "*")
    )
    print("Found Parquet files for evaluation: {}".format(input_parquet_files))
    print("Will save metrics results to parquet: {}".format(args.output_parquet_path))

    pq_writer = None
    try:
        metrics_results = MetricsResults()

        for idx_file, input_file_path in enumerate(input_parquet_files):
            rec_df = load_rec_file(input_file_path)
            rec_df = pd.concat([pd.DataFrame(rec_df) for i in range(1000)], ignore_index=True)

            chunk_size = args.chunk_size if args.chunk_size else len(rec_df)
            recs_chunks = process_dataframe_into_chuncks_generator(rec_df, chunk_size=chunk_size) 

            for chunk_id, rec_chunk_df in tqdm(enumerate(recs_chunks)):
                print(
                    "Computing metrics of chunk {} (# rows: {})".format(
                        chunk_id, len(rec_chunk_df)
                    )
                )

                metrics_results_dict = compute_metrics(rec_chunk_df, args)

                metrics_results.add(metrics_results_dict)

                #_log_metrics_summary(metrics_results_dict)

                _add_metrics_results_to_dataframe(rec_chunk_df, metrics_results_dict)

                if not pq_writer:
                    file_name = os.path.basename(input_file_path)
                    pq_writer = create_pq_writer(rec_chunk_df, os.path.join(args.output_parquet_path, 'eval_'+file_name))

                append_new_rows_to_parquet(rec_chunk_df, pq_writer)

            if pq_writer:
                pq_writer.close()
                pq_writer = None


        print()
        print('======================= Global metrics =======================')
        global_metrics_results = metrics_results.result()
        print(global_metrics_results)

        path = os.path.join(args.output_parquet_path, METRICS_RESULTS_FILE)
        print('Saving global metrics results to {}'.format(path))
        with open(path, 'w') as fp:
            json.dump(global_metrics_results, fp, sort_keys=True, indent=2)

        print('Finished evaluation')
    finally:
        if pq_writer:
            pq_writer.close()


if __name__ == "__main__":
    main()
