import os
import shutil
import tempfile

from tqdm import tqdm


def save_time_based_splits(
    data,
    output_dir,
    partition_col="day_idx",
    timestamp_col="ts/first",
    test_size=0.1,
    val_size=0.1,
    overwrite=True,
):
    try:
        import cudf
        import cupy
        import dask_cudf
        import nvtabular as nvt
    except ImportError as error:
        print(
            "ModuleNotFoundError: '%s' package is required to use save_time_based_splits function"
            % error.name
        )

    """
    Args:
    -----
        data
        output_dir
        partition_col
        timestamp_col
        test_size
        val_size
        overwrite
    """
    if isinstance(data, dask_cudf.DataFrame):
        data = nvt.Dataset(data)
    if not isinstance(partition_col, list):
        partition_col = [partition_col]

    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    with tempfile.TemporaryDirectory() as tmpdirname:
        data.to_parquet(tmpdirname, partition_on=partition_col)
        time_dirs = [f for f in sorted(os.listdir(tmpdirname)) if f.startswith(partition_col[0])]
        for d in tqdm(time_dirs, desc="Creating time-based splits"):
            path = os.path.join(tmpdirname, d)
            df = cudf.read_parquet(path)
            df = df.sort_values(timestamp_col)

            split_name = d.replace(f"{partition_col[0]}=", "")
            out_dir = os.path.join(output_dir, split_name)
            os.makedirs(out_dir, exist_ok=True)
            df.to_parquet(os.path.join(out_dir, "train.parquet"))

            random_values = cupy.random.rand(len(df))

            # Extracts 10% for valid and test set.
            # Those sessions are also in the train set, but as evaluation
            # happens only for the subsequent day of training,
            # that is not an issue, and we can keep the train set larger.
            valid_set = df[random_values <= val_size]
            valid_set.to_parquet(os.path.join(out_dir, "valid.parquet"))

            test_set = df[random_values >= 1 - test_size]
            test_set.to_parquet(os.path.join(out_dir, "test.parquet"))
