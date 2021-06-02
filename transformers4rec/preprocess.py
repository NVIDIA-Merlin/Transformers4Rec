import collections
from contextlib import contextmanager
import tempfile
import os
import shutil

from tqdm import tqdm
import numpy as np
import cupy
import cudf
import dask_cudf
import nvtabular as nvt
from nvtabular.ops import Operator
from cudf.core.dtypes import ListDtype

ListDtype


def get_from_col_group(self, to_get):
    if isinstance(to_get, nvt.ColumnGroup):
        to_get = set(to_get.columns)
    elif isinstance(to_get, str):
        to_get = {to_get}
    elif isinstance(to_get, collections.abc.Sequence):
        to_get = set(to_get)
    else:
        raise ValueError(f"Expected ColumnGroup, str, or list of str. Got {to_get.__class__}")
    
    new_columns = [c for c in self.columns if c in to_get]
    child = nvt.ColumnGroup(new_columns)
    child.parents = [self]
    self.children.append(child)
    child.kind = f"- {[c for c in self.columns if c not in to_get]}"
    
    return child

nvt.ColumnGroup.filter = get_from_col_group


def remove_consecutive_interactions(df, session_id_col="session_id", item_id_col="item_id", timestamp_col="timestamp"):
    print("Count with in-session repeated interactions: {}".format(len(df)))
    # Sorts the dataframe by session and timestamp, to remove consective repetitions
    df = df.sort_values([session_id_col, timestamp_col])
    df['item_id_past'] = df[item_id_col].shift(1)
    df['session_id_past'] = df[session_id_col].shift(1)
    #Keeping only no consectutive repeated in session interactions
    df = df[~((df[session_id_col] == df['session_id_past']) & \
                    (df[item_id_col] == df['item_id_past']))]
    print("Count after removed in-session repeated interactions: {}".format(len(df)))

    del(df['item_id_past'])
    del(df['session_id_past'])

    return df



def create_session_aggs(column_group, default_agg="list", extra_aggs=None, to_ignore=None):
    if not extra_aggs:
        extra_aggs = {}
    if not to_ignore:
        to_ignore = []
    
    aggs = {col: [default_agg] for col in column_group.columns if col not in to_ignore}
    for key, val in extra_aggs.items():
        if key in aggs:
            if isinstance(val, list):
                aggs[key].extend(val)
            else:
                aggs[key].append(val)
        else:
            aggs[key] = val
            
    return aggs


class Namespace(Operator):
    def __init__(self, namespace):
        self.namespace = namespace

    def transform(self, columns, gdf):
        gdf.columns = self.output_column_names(columns)
        return gdf

    def output_column_names(self, columns):
        return [
            "/".join([self.namespace, column]) if not column.startswith(self.namespace) else column
            for column in columns
        ]


class Ops(object):
    _namespace_items = []

    def __init__(self, *ops, auto_renaming=False, sequential=True):
        self._ops = list(ops) if ops else []
        self._ops_by_name = {}
        self.sequential = sequential
        self.auto_renaming = auto_renaming

    def add(self, op, name=None):
        self._ops.append(op)
        if name:
            self._ops_by_name[name] = op

        return op

    def extend(self, ops):
        self._ops.extend(ops)

        return ops

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ops[key]

        return self._ops_by_name[key]

    @property
    def namespace(self):
        return "/".join(self._namespace_items)

    def __call__(self, col_or_cols, add=False):
        x = col_or_cols
        name_parts = []

        if self.sequential:
            for op in self._ops:
                if isinstance(op, Operator):
                    name_parts.append(op.__class__.__name__)
                    x = x >> op
                else:
                    x = op(x)
        else:
            out = None
            for op in self._ops:
                if out:
                    out += col_or_cols >> op
                else:
                    out = col_or_cols >> op
            x = out

        if self.auto_renaming:
            x = x >> nvt.ops.Rename(postfix="/" + "/".join(name_parts))

        if self.namespace:
            x = x >> Namespace(self.namespace)

        if add:
            return col_or_cols + x

        return x

    def __rrshift__(self, other):
        return self.__call__(other)

    def copy(self):
        to_return = Ops(*self._ops, auto_renaming=self.auto_renaming, sequential=self.sequential)
        to_return._ops_by_name = self._ops_by_name

        return to_return


@contextmanager
def namespace(name):
    Ops._namespace_items.append(name)
    try:
        yield
    finally:
        Ops._namespace_items.pop()


class ItemRecency(Operator):
    def __init__(self, first_seen_column_name, out_col_suffix="/age_days") -> None:
        super().__init__()
        self.first_seen_column_name = first_seen_column_name
        self.out_col_suffix = out_col_suffix

    def transform(self, columns, gdf):
        for column in columns:
            col = gdf[column]
            item_first_timestamp = gdf[self.first_seen_column_name]
            delta_days = (col - item_first_timestamp).dt.days
            gdf[column + self.out_col_suffix] = delta_days * (delta_days >= 0)
        return gdf

    def output_column_names(self, columns):
        return [column + self.out_col_suffix for column in columns]

    def dependencies(self):
        return [self.first_seen_column_name]


class TimestampFeatures(Ops):
    def __init__(self, add_timestamp=True, add_cycled=True, auto_renaming=False, delimiter="/"):
        super().__init__(auto_renaming=auto_renaming, sequential=False)
        del_fn = lambda x: f"{delimiter}{x}"
        hour = self.add(
            Ops(
                nvt.ops.LambdaOp(lambda col: col.dt.hour), nvt.ops.Rename(postfix=del_fn("hour"))
            )
        )
        weekday = self.add(
            Ops(
                nvt.ops.LambdaOp(lambda col: col.dt.weekday),
                nvt.ops.Rename(postfix=del_fn("weekday")),
            )
        )
        self.add(
            Ops(nvt.ops.LambdaOp(lambda col: col.dt.day), nvt.ops.Rename(postfix=del_fn("day")))
        )
        self.add(
            Ops(
                nvt.ops.LambdaOp(lambda col: col.dt.month),
                nvt.ops.Rename(postfix=del_fn("month")),
            )
        )
        self.add(
            Ops(
                nvt.ops.LambdaOp(lambda col: col.dt.year), nvt.ops.Rename(postfix=del_fn("year"))
            )
        )

        if add_timestamp:
            self.add(
                Ops(
                    nvt.ops.LambdaOp(lambda col: (col.astype(int) / 1e6).astype(int)),
                    nvt.ops.Rename(f=lambda col: "ts"),
                )
            )

        if add_cycled:
            self.add(Ops(*hour._ops, nvt.ops.LambdaOp(lambda col: self.get_cycled_feature_value_sin(col, 24)), nvt.ops.Rename(postfix="_sin")))
            self.add(Ops(*hour._ops, nvt.ops.LambdaOp(lambda col: self.get_cycled_feature_value_cos(col, 24)), nvt.ops.Rename(postfix="_cos")))
            self.add(Ops(*weekday._ops, nvt.ops.LambdaOp(lambda col: self.get_cycled_feature_value_sin(col + 1, 7)), nvt.ops.Rename(postfix="_sin")))
            self.add(Ops(*weekday._ops, nvt.ops.LambdaOp(lambda col: self.get_cycled_feature_value_cos(col + 1, 7)), nvt.ops.Rename(postfix="_cos")))

    @staticmethod
    def get_cycled_feature_value_sin(col, max_value):
        value_scaled = (col + 0.000001) / max_value
        value_sin = np.sin(2 * np.pi * value_scaled)

        return value_sin

    @staticmethod
    def get_cycled_feature_value_cos(col, max_value):
        value_scaled = (col + 0.000001) / max_value
        value_cos = np.cos(2 * np.pi * value_scaled)

        return value_cos


class SessionDay(Ops):
    def __init__(self, name="day_idx", padding=4):
        super().__init__()
        self.extend([
            nvt.ops.LambdaOp(lambda x: (x - x.min()).dt.days + 1),
            nvt.ops.LambdaOp(lambda col: col.astype(str).str.pad(padding, fillchar='0')),
            nvt.ops.Rename(f=lambda col: name)
        ])


def create_timestap_features(column_name, add_cycled=True, add_timestamp=True):
    hour = (
        column_name
        >>
        # nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='ms').dt.hour) >>
        nvt.ops.LambdaOp(lambda col: col.dt.hour)
        >> nvt.ops.Rename(postfix="_hour")
    )
    weekday = (
        column_name
        >>
        # nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='ms').dt.weekday) >>
        nvt.ops.LambdaOp(lambda col: col.dt.weekday)
        >> nvt.ops.Rename(postfix="_wd")
    )
    day = column_name >> nvt.ops.LambdaOp(lambda col: col.dt.day) >> nvt.ops.Rename(postfix="_day")
    month = (
        column_name
        >> nvt.ops.LambdaOp(lambda col: col.dt.month)
        >> nvt.ops.Rename(postfix="_month")
    )
    year = (
        column_name >> nvt.ops.LambdaOp(lambda col: col.dt.year) >> nvt.ops.Rename(postfix="_year")
    )

    outputs = hour + weekday + day + month + year

    if add_timestamp:
        outputs += (
            column_name
            >> nvt.ops.LambdaOp(lambda col: (col.astype(int) / 1e6).astype(int))
            >> nvt.ops.Rename(f=lambda col: "ts")
        )

    if add_cycled:

        def get_cycled_feature_value_sin(col, max_value):
            value_scaled = (col + 0.000001) / max_value
            value_sin = np.sin(2 * np.pi * value_scaled)
            return value_sin

        def get_cycled_feature_value_cos(col, max_value):
            value_scaled = (col + 0.000001) / max_value
            value_cos = np.cos(2 * np.pi * value_scaled)
            return value_cos

        hour_sin = (
            hour
            >> (lambda col: get_cycled_feature_value_sin(col, 24))
            >> nvt.ops.Rename(postfix="_sin")
        )
        hour_cos = (
            hour
            >> (lambda col: get_cycled_feature_value_cos(col, 24))
            >> nvt.ops.Rename(postfix="_cos")
        )
        weekday_sin = (
            weekday
            >> (lambda col: get_cycled_feature_value_sin(col + 1, 7))
            >> nvt.ops.Rename(postfix="_sin")
        )
        weekday_cos = (
            weekday
            >> (lambda col: get_cycled_feature_value_cos(col + 1, 7))
            >> nvt.ops.Rename(postfix="_cos")
        )

        outputs += hour_sin + hour_cos + weekday_sin + weekday_cos

    return outputs


def save_time_based_splits(data, output_dir, partition_col="day_idx", timestamp_col="ts/first", test_size=0.1, val_size=0.1, overwrite=True):
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
            df.to_parquet(os.path.join(out_dir, 'train.parquet'))
            
            random_values = cupy.random.rand(len(df))
            
            #Extracts 10% for valid and test set. Those sessions are also in the train set, but as evaluation
            #happens only for the subsequent day of training, that is not an issue, and we can keep the train set larger.
            valid_set = df[random_values <= val_size]
            valid_set.to_parquet(os.path.join(out_dir, 'valid.parquet'))
            
            test_set = df[random_values >= 1 - test_size]
            test_set.to_parquet(os.path.join(out_dir, 'test.parquet'))
