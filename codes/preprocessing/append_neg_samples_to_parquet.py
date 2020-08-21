import argparse
import gc
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from tqdm import tqdm

from ..candidate_sampling.candidate_sampling import (
    PersistanceType,
    SamplingManagerFactory,
    SamplingStrategy,
)
from ..config.features_config import (
    FeatureGroups,
    FeaturesDataType,
    FeatureTypes,
    InputDataConfig,
    InstanceInfoLevel,
)

np.random.seed(42)


def args_parser():
    parser = argparse.ArgumentParser(
        description="Appends negative samples to existing parquet files."
    )
    parser.add_argument(
        "--input_parquet_path_pattern",
        type=str,
        help="Path to look for the pre-processed parquet files (accepts *)",
    )
    parser.add_argument(
        "--output_parquet_root_path",
        type=str,
        help="Output root path where the processed parquet will be saved",
    )
    parser.add_argument(
        "--subsample_first_n_sessions_by_day",
        type=int,
        default=0,
        help="Processes only the first N sessions by date, to reduce output files and speed up processing",
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        help="Valid choices: uniform|recency|recent_popularity|session_cooccurrence",
    )
    parser.add_argument(
        "--num_neg_samples",
        type=int,
        help="Number of negative samples for each positive interaction",
    )
    parser.add_argument(
        "--perc_valid_set",
        type=float,
        default=0.9,
        help="Percentage of rows for the validation set",
    )
    parser.add_argument(
        "--keep_repeated_sampled_items",
        default=False,
        action="store_true",
        help="Indicates if repeated samples items should be kept among negative samples (to keep the sampling probability distribution unchanged)",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Number of rows for buffered reading from parquet file"
    )
    parser.add_argument(
        "--update_stats_each_n_batches", type=int, help="Updates the stats each N batches"
    )
    parser.add_argument(
        "--save_each_n_batches",
        type=int,
        help="Appends new rows to the output parquet file each N batches",
    )
    parser.add_argument(
        "--sliding_windows_last_n_days",
        type=float,
        help="Considers as recommendable only items interacted in the last N days",
    )
    parser.add_argument(
        "--recent_temporal_decay_exp_factor",
        type=float,
        default=0.002,
        help='When --sampling_strategy="recency", Configures the exponential decay factor which is used to set the probability of an item to be sampled based on its first interaction (i.e. its age)',
    )

    return parser


def create_output_folder_path(output_root_path, dataset_name):
    out_folder_path = os.path.join(output_root_path, dataset_name)
    # Creates dirs recursivelly if they do not exist
    Path(out_folder_path).mkdir(parents=True, exist_ok=True)
    return out_folder_path


def get_output_path_parquet_neg_samples(input_parquet_filename, out_folder_path):
    folder_name_with_day = os.path.split(input_parquet_filename)[-1]
    output_file_path = os.path.join(out_folder_path, folder_name_with_day + ".parquet")
    return output_file_path


def padarray(A, size):
    if len(A) > size:
        A = A[:size]
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode="constant")


def get_input_data_config():
    # Only features necessary for negative sampling (item features)
    features_schema = {
        "sess_pid_seq": FeaturesDataType.LONG,
        "sess_csid_seq": FeaturesDataType.INT,
        "sess_ccid_seq": FeaturesDataType.INT,
        "sess_bid_seq": FeaturesDataType.INT,
        "sess_price_seq": FeaturesDataType.FLOAT,
        "sess_relative_price_to_avg_category_seq": FeaturesDataType.FLOAT,
        "sess_product_recency_seq": FeaturesDataType.FLOAT,
        "sess_etime_seq": FeaturesDataType.LONG,
    }

    feature_groups = FeatureGroups(
        user_id="user_idx",  # Not necessary for negative sampling
        item_id="sess_pid_seq",
        session_id="user_session",  # Not necessary for negative sampling
        implicit_feedback=None,
        event_timestamp="sess_etime_seq",
        item_metadata=[
            "sess_csid_seq",
            "sess_ccid_seq",
            "sess_bid_seq",
            "sess_price_seq",
            "sess_relative_price_to_avg_category_seq",
            "sess_product_recency_seq",
        ],
        user_metadata=[],  # Not necessary for negative sampling
        event_metadata=[],  # Not necessary for negative sampling
        sequential_features=[
            "sess_etime_seq",
            "sess_pid_seq",
            "sess_ccid_seq",
            "sess_ccid_seq",
            "sess_bid_seq",
            "sess_price_seq",
            "sess_relative_price_to_avg_category_seq",
            "sess_product_recency_seq",
        ],
    )

    feature_types = FeatureTypes(
        categorical=[
            "user_idx",
            "user_session",
            "sess_pid_seq",
            "sess_csid_seq",
            "sess_ccid_seq",
            "sess_bid_seq",
        ],  # Not necessary for negative sampling
        numerical=[
            "sess_etime_seq",
            "sess_price_seq",
            "sess_relative_price_to_avg_category_seq",
            "sess_product_recency_seq",
        ],  # Not necessary for negative sampling
    )

    input_data_config = InputDataConfig(
        schema=features_schema,
        feature_groups=feature_groups,
        feature_types=feature_types,
        positive_interactions_only=True,
        instance_info_level=InstanceInfoLevel.SESSION,
        session_padded_items_value=0,
    )

    return input_data_config


def get_sampling_managers(input_data_config, args):
    sampling_managers_chain = []
    sampling_strategy = SamplingStrategy(args.sampling_strategy)

    sampling_manager = SamplingManagerFactory.build(
        input_data_config=input_data_config,
        sampling_strategy=sampling_strategy,
        persistance_type=PersistanceType.PANDAS,
        recency_keep_interactions_last_n_days=args.sliding_windows_last_n_days,
        recent_temporal_decay_exp_factor=args.recent_temporal_decay_exp_factor,
        remove_repeated_sampled_items=not args.keep_repeated_sampled_items,
    )
    sampling_managers_chain.append(sampling_manager)

    # For Session Co-occurrence, as it sometimes is not able to provide the required number of negative samples,
    # we add to the chain the Recent Popularity Sampling Strategy to provide the remaining negative samples
    if sampling_strategy == SamplingStrategy.SESSION_COOCURRENCE:
        sampling_manager = SamplingManagerFactory.build(
            input_data_config=input_data_config,
            sampling_strategy=SamplingStrategy.RECENT_POPULARITY,
            persistance_type=PersistanceType.PANDAS,
            recency_keep_interactions_last_n_days=args.sliding_windows_last_n_days,
            recent_temporal_decay_exp_factor=args.recent_temporal_decay_exp_factor,
            remove_repeated_sampled_items=not args.keep_repeated_sampled_items,
        )
        sampling_managers_chain.append(sampling_manager)

    return sampling_managers_chain


def generate_neg_samples(
    session_pids, user_past_pids, n_samples, sampling_managers_chain, input_data_config
):
    neg_samples_dict = defaultdict(list)

    # Ignores session items and also recently interacted items
    user_interactions_item_ids = set(np.hstack([session_pids, user_past_pids]))

    categ_features = input_data_config.get_features_from_type("categorical")

    for pid in session_pids:
        if pid != 0:
            neg_item_ids, neg_item_features = None, None
            # Iterating over the chain of sampling managers, breaking when the required number of negative samples is reached
            for sampling_manager in sampling_managers_chain:
                pending_n_samples = (
                    n_samples if neg_item_ids is None else n_samples - len(neg_item_ids)
                )

                # Ignores items interacted by users and already sampled negative items
                ignore_ids = user_interactions_item_ids.union(set(neg_item_ids))
                # Sampling item ids, with metadata features
                (
                    neg_item_ids_from_strategy,
                    neg_item_features_from_strategy,
                ) = sampling_manager.get_candidate_samples(
                    pending_n_samples,
                    item_id=pid,
                    return_item_features=True,
                    ignore_items=ignore_ids,
                )
                if neg_item_ids is None:
                    neg_item_ids = neg_item_ids_from_strategy
                    neg_item_features = neg_item_features_from_strategy
                else:
                    neg_item_ids = np.hstack([neg_item_ids, neg_item_ids_from_strategy])
                    for k in neg_item_features_from_strategy:
                        neg_item_features[k] = np.hstack(
                            [neg_item_features[k], neg_item_features_from_strategy[k]]
                        )

                if len(neg_item_ids) >= n_samples:
                    break

            # neg_item_ids, neg_item_features = zip(*neg_item_features_dict.items())

            pids_padded = padarray(neg_item_ids, n_samples).astype(int)
            neg_samples_dict["sess_neg_pids"].append(pids_padded)

            for k, v in neg_item_features.items():
                values = padarray(v, n_samples)
                values = values.astype(int) if k in categ_features else values.astype(float)
                neg_samples_dict["sess_neg_{}".format(k)].append(values)

        else:
            # Creating padding neg samples for each padding interactions
            missing_padding_neg_samples = len(session_pids) - len(
                neg_samples_dict["sess_neg_pids"]
            )
            for k in neg_samples_dict:
                neg_samples = neg_samples_dict[k]
                neg_samples_zeros = np.zeros_like(neg_samples[0])
                for p in range(missing_padding_neg_samples):
                    # Copying shape and dtype from the neg samples of the first interaction
                    neg_samples.append(neg_samples_zeros)

    # Concatenating (flattening) neg. samples of all session interactions because Petastorm data loader
    # does not support lists of lists. It will require reshaping neg. samples features to shape (max_seq_len=20, n_neg_samples=50) inside the Pytorch model
    for k in neg_samples_dict:
        neg_samples_dict[k] = np.hstack(neg_samples_dict[k])

    return neg_samples_dict


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


def get_data_chunk_generator(input_file, chunck_size, subsample_first_n_sessions_by_day):
    def process_dataframe_into_chuncks_generator(df, chunk_size):
        number_chunks = len(df) // chunk_size + 1
        for i in range(number_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            yield df[start_idx:end_idx]
        del df
        gc.collect()

    # Loading parquet file and sorting sessions by timestamp
    # P.s. Everything is loaded in memory at this time
    sessions_df = pd.read_parquet(input_file)

    if subsample_first_n_sessions_by_day > 0:
        # Shuffling the order of sesions before filtering
        sessions_df = sessions_df.sample(frac=1).reset_index(drop=True)

        # Limiting the number of negative samples per day for faster processing
        sessions_df = sessions_df[:subsample_first_n_sessions_by_day]

    # Sorting the sessions by start time
    sessions_df.sort_values("session_start_ts", inplace=True)

    data_chunks_generator = process_dataframe_into_chuncks_generator(
        sessions_df, chunk_size=chunck_size
    )
    return data_chunks_generator


"""
Temporary, as the event timestamps from some rare items of the pre-processed dataset are many days after the session start
"""


def fix_event_timestamp(event_ts_sequence):
    MAXIMUM_SESSION_DURATION_SECS = 60 * 120
    DEFAULT_ELAPSED_SECS_BETWEEN_INTERACTIONS = 60
    min_event_ts = min([e for e in event_ts_sequence if e != 0])
    max_event_ts = max(event_ts_sequence)

    if max_event_ts <= min_event_ts + MAXIMUM_SESSION_DURATION_SECS:
        return event_ts_sequence

    else:
        last_ts = None

        result = []
        for etime in event_ts_sequence:
            if etime > min_event_ts + MAXIMUM_SESSION_DURATION_SECS:
                etime = last_ts + DEFAULT_ELAPSED_SECS_BETWEEN_INTERACTIONS
            last_ts = etime
            result.append(etime)

        return result


def process_date(
    idx_day, input_file, input_data_config, output_folder_path, sampling_managers_chain, args
):
    pq_writer = None
    try:
        print("=" * 40)
        print("[Day {}] Loading sessions from parquet: {}".format(idx_day, input_file))

        output_filename = get_output_path_parquet_neg_samples(input_file, output_folder_path)

        if os.path.exists(output_filename):
            raise Exception("Output parquet file already exists: {}".format(output_filename))

        data_chunks_generator = get_data_chunk_generator(
            input_file, args.batch_size, args.subsample_first_n_sessions_by_day
        )

        new_rows_buffer = []

        print("Processing batches")
        # For each processing batch
        for batch_id, batch in tqdm(enumerate(data_chunks_generator)):
            print("batch_id", batch_id)
            # For each row (session) in the batch
            for i, row in batch.iterrows():

                # Temporary, as the event timestamps from some rare items of the pre-processed dataset are many days after the session start
                row["sess_etime_seq"] = fix_event_timestamp(row["sess_etime_seq"])

                for sampling_manager in sampling_managers_chain:
                    sampling_manager.append_session_interactions(row)

                # Ignoring first batch (not computing neg. samples nor saving to parquet)
                if batch_id > 0:
                    # Generating neg. samples for each interaction in the session
                    session_neg_samples_by_pid_dict = generate_neg_samples(
                        row["sess_pid_seq"],
                        row["user_pid_seq_bef_sess"],
                        args.num_neg_samples,
                        sampling_managers_chain,
                        input_data_config,
                    )

                    # Merging user and session features with neg samples for the session
                    new_row_with_neg_samples_dict = {
                        **row.to_dict(),
                        **session_neg_samples_by_pid_dict,
                    }
                    new_rows_buffer.append(new_row_with_neg_samples_dict)

            # Each N batches updates item statistics (popularity, recency, co-occurrence)
            # Ps. Do the update for all the first five batches of the first file , for better sampling
            if ((batch_id + 1) % args.update_stats_each_n_batches == 0) or (
                idx_day == 0 and batch_id < 5
            ):
                print("[Batch {}] Updating item stats".format(batch_id))
                for sampling_manager in sampling_managers_chain:
                    # Flushes pending interactions, removes old interactions (outside recency window) and update stats for sampling
                    sampling_manager.update_stats()

            # Each N batches appends the new rows with neg. samples to parquet file
            if (
                batch_id % args.save_each_n_batches == args.save_each_n_batches - 1
                and len(new_rows_buffer) > 0
            ):
                print(
                    "[Batch {}] Appending new rows with neg samples to parquet: {}".format(
                        batch_id, output_filename
                    )
                )

                new_rows_df = pd.DataFrame(new_rows_buffer)

                if not pq_writer:
                    pq_writer = create_pq_writer(new_rows_df, output_filename)

                append_new_rows_to_parquet(new_rows_df, pq_writer)

                del new_rows_df
                del new_rows_buffer
                new_rows_buffer = []

        # Save pending rows
        if len(new_rows_buffer) > 0:
            print(
                "[Batch {}] Appending new rows with neg samples to parquet: {}".format(
                    batch_id, output_filename
                )
            )
            new_rows_df = pd.DataFrame(new_rows_buffer)
            append_new_rows_to_parquet(new_rows_df, pq_writer)
            del new_rows_df
            del new_rows_buffer
            new_rows_buffer = []

        # Flushing and releasing the current parquet file and proceeding for the new date
        if pq_writer:
            pq_writer.close()
            pq_writer = None

        return output_filename

    finally:
        if pq_writer:
            pq_writer.close()


def split_train_eval_parquet_file(parquet_path, perc_valid_set):
    sessions_df = pd.read_parquet(parquet_path)

    # Shuffling the order of sesions
    sessions_df = sessions_df.sample(frac=1).reset_index(drop=True)

    dataset_size = len(sessions_df)

    train_set_limit = int(dataset_size * perc_valid_set)

    train_df = sessions_df[:train_set_limit]
    valid_df = sessions_df[train_set_limit:dataset_size]

    try:

        # Sorting the sessions by start time
        train_df.sort_values("session_start_ts", inplace=True)
        valid_df.sort_values("session_start_ts", inplace=True)

        output_train_path = parquet_path.replace(".parquet", "-train.parquet")
        output_test_path = parquet_path.replace(".parquet", "-test.parquet")

        train_df.to_parquet(output_train_path)
        valid_df.to_parquet(output_test_path)

    finally:
        del sessions_df
        del train_df
        del valid_df
        gc.collect()


def main():
    parser = args_parser()
    args = parser.parse_args()

    input_parquet_files = sorted(glob.glob(args.input_parquet_path_pattern.replace("'", "") + "*"))

    input_data_config = get_input_data_config()
    sampling_managers_chain = get_sampling_managers(input_data_config, args)

    # Creating and output folder according to the sampling configuration
    dataset_name = "ecommerce_preproc_neg_samples_{}_strategy_{}".format(
        args.num_neg_samples, args.sampling_strategy
    )
    output_folder_path = create_output_folder_path(args.output_parquet_root_path, dataset_name)

    # For each file (day)
    for idx_day, input_file in enumerate(input_parquet_files):
        output_parquet_path = process_date(
            idx_day,
            input_file,
            input_data_config,
            output_folder_path,
            sampling_managers_chain,
            args,
        )

        # Splits the generated parquet file into two parquets (train and eval)
        split_train_eval_parquet_file(output_parquet_path, args.perc_valid_set)

    print("Preprocessing script finished")


if __name__ == "__main__":
    main()
