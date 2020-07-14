import argparse

import pandas as pd
from tqdm import tqdm

from ..candidate_sampling.candidate_sampling import CandidateSamplingManager


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
        "--output_parquet_path", type=str, help="Output path with the parquet files appended"
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
        help="Valid choices: uniform|recency|popularity|cooccurrence",
    )
    parser.add_argument(
        "--num_neg_samples",
        type=int,
        help="Number of negative samples for each positive interaction",
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


# TODO: Create a new folder tree in the output directory
def get_output_path_parquet_neg_samples(input_parquet_filename):
    return (
        input_parquet_filename.replace(
            "ecommerce_preproc.parquet",
            "ecommerce_preproc_neg_samples_{}_strategy_{}.parquet".format(
                NUM_NEG_SAMPLES, NEGATIVE_SAMPLING_STRATEGY
            ),
        )
        + ".parquet"
    )


def split_dataframe_into_chuncks_generator(df, chunk_size):
    number_chunks = len(df) // chunk_size + 1
    for i in range(number_chunks):
        yield df[i * chunk_size : (i + 1) * chunk_size]


def get_input_data_config():
    # Only features necessary for negative sampling (item features)
    features_schema = {
        "sess_pid_seq": FeaturesDataType.LONG,
        "sess_csid_seq": FeaturesDataType.INT,
        "sess_ccid_seq": FeaturesDataType.INT,
        "sess_pid_seq": FeaturesDataType.INT,
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
        instance_info_level=InstanceLevelInfo.SESSION,
        session_padded_items_value=0,
    )

    return input_data_config


def get_sampling_manager(args):
    sampling_config = CandidateSamplingConfig(
        recommendable_items_strategy=RecommendableItemSetStrategy.RECENT_INTERACTIONS,
        sampling_strategy=SamplingStrategy.RECENT_POPULARITY,
        persistance_type=PersistanceType.PANDAS,
        recency_keep_interactions_last_n_days=args.sliding_windows_last_n_days,
        recent_temporal_decay_exp_factor=args.recent_temporal_decay_exp_factor,
        remove_repeated_sampled_items=not args.keep_repeated_sampled_items,
    )

    input_data_config = get_input_data_config()

    sampling_manager = CandidateSamplingManager(input_data_config, sampling_config)

    return sampling_manager


def generate_neg_samples(session_pids, user_past_pids, n_samples, sampling_manager):
    neg_samples_dict = defaultdict(list)

    # Ignores session items and also recently interacted items
    ignore_ids = set(np.hstack([session_pids, user_past_pids]))

    for pid in session_pids:
        if pid != 0:
            # Sampling item ids, with metadata features
            neg_item_features_dict = sampling_manager.get_candidate_samples(
                n_samples, item_id=pid, return_item_features=True, ignore_list=ignore_ids
            )

            neg_item_ids, neg_item_features = zip(*neg_item_features_dict.items())

            pids_padded = padarray(neg_item_ids, n_samples).astype(int)
            neg_samples_dict["sess_neg_pids"].append(pids_padded)

            for k, v in neg_item_features.items():
                values = padarray(v, n_samples)
                values = (
                    values.astype(int) if k in ["csid", "ccid", "bid"] else values.astype(float)
                )
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
    # does not support lists of lists. It will require reshaping neg. samples features inside the Pytorch model
    for k in neg_samples_dict:
        neg_samples_dict[k] = np.hstack(neg_samples_dict[k])

    return neg_samples_dict


def main():
    parser = args_parser()
    args = parser.parse_args()

    input_parquet_files = input_parquet_files = sorted(
        glob.glob(args.input_parquet_path_pattern + "*")
    )

    sampling_manager = get_sampling_manager()

    pq_writer = None
    try:
        # For each file (day)
        for idx_day, input_file in enumerate(input_parquet_files):
            print("=" * 40)
            print("[Day {}] Loading sessions from parquet: {}".format(idx_day, input_file))
            output_filename = get_output_path_parquet_neg_samples(input_file)

            if os.path.exists(output_filename):
                raise Exception("Output parquet file already exists")

            # Loading parquet file and sorting sessions by timestamp
            # P.s. Everything is loaded in memory at this time
            sessions_df = pd.read_parquet(input_file)
            sessions_df.sort_values("session_start_ts", inplace=True)

            if args.subsample_first_n_sessions_by_day > 0:
                # Limiting the number of negative samples per day for faster processing
                sessions_df = sessions_df[:FIRST_N_SESSIONS_PER_DAY]

            new_rows_buffer = []

            print("Processing batches")
            # For each processing batch
            for batch_id, batch in tqdm(
                enumerate(
                    split_dataframe_into_chuncks_generator(sessions_df, chunk_size=args.batch_size)
                )
            ):
                print("batch_id", batch_id)
                # For each row (session) in the batch
                for i, row in batch.iterrows():

                    sampling_manager.append_session_interactions(row)

                    # Ignoring first batch (not computing neg. samples nor saving to parquet)
                    if batch_id > 0:
                        # Generating neg. samples for each interaction in the session
                        session_neg_samples_by_pid_dict = generate_neg_samples(
                            row["sess_pid_seq"],
                            row["user_pid_seq_bef_sess"],
                            args.num_neg_samples,
                            sampling_manager,
                        )
                        # Merging user and session features with neg samples for the session
                        new_row_with_neg_samples_dict = {
                            **row.to_dict(),
                            **session_neg_samples_by_pid_dict,
                        }
                        new_rows.append(new_row_with_neg_samples_dict)

                # Each N batches updates item statistics (popularity, recency, co-occurrence)
                # Ps. Do the update for all the first five batches of the first file , for better sampling
                if (
                    batch_id % args.update_stats_each_n_batches
                    == args.update_stats_each_n_batches - 1
                ) or (idx_day == 0 and batch_id < 5):
                    print("[Batch {}] Updating item stats".format(batch_id))
                    # Flushes pending interactions, removes old interactions (outside recency window) and update stats for sampling
                    sampling_manager.update_stats()

                # Each N batches appends the new rows with neg. samples to parquet file
                if (
                    batch_id % BATCHES_TO_APPEND_ROWS_WITH_NEG_SAMPLES
                    == BATCHES_TO_APPEND_ROWS_WITH_NEG_SAMPLES - 1
                ):
                    print(
                        "[Batch {}] Appending new rows with neg samples to parquet: {}".format(
                            batch_id, output_filename
                        )
                    )
                    append_new_rows_to_parquet(pd.DataFrame(new_rows_buffer), output_filename)
                    del new_rows_buffer
                    new_rows_buffer = []

            # Save pending rows
            if len(new_rows_buffer) > 0:
                print(
                    "[Batch {}] Appending new rows with neg samples to parquet: {}".format(
                        batch_id, output_filename
                    )
                )
                append_new_rows_to_parquet(pd.DataFrame(new_rows_buffer), output_filename)
                del new_rows_buffer
                new_rows_buffer = []

            # Flushing and releasing the current parquet file and proceeding for the new date
            pq_writer.close()
            pq_writer = None

            del sessions_df
            gc.collect()
    finally:
        if pq_writer:
            pq_writer.close()


if __name__ == "__main__":
    main()
