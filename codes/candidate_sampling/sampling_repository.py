from abc import ABC, abstractmethod
from enum import Enum
from itertools import permutations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config.features_config import FeatureGroupType, InputDataConfig, ItemId


class PersistanceType(Enum):
    PANDAS = "pandas"


class ItemsMetadataRepository(ABC):

    # Internal columns
    ITEM_ID_COL = "item_id"
    INTERNAL_FIRST_TS_COL = "first_ts"
    INTERNAL_LAST_TS_COL = "last_ts"

    dconf: InputDataConfig
    item_features_names: List[str]

    def __init__(self, input_data_config: InputDataConfig) -> None:
        self.dconf = input_data_config
        self.item_features_names = self.dconf.get_item_feature_names()

        self.item_id_col = self.dconf.get_feature_group(FeatureGroupType.ITEM_ID)
        self.event_ts_col = self.dconf.get_feature_group(FeatureGroupType.EVENT_TS)

    def update_item_metadata(self, item_features_dict: Dict[str, Any]) -> None:
        item_features_dict = item_features_dict.copy()
        item_id = item_features_dict.pop(self.item_id_col)

        event_ts = item_features_dict.pop(self.event_ts_col)

        # Keeps a registry of the first and last interactions of an item
        if self.item_exists(item_id):
            item_row = self.get_item(item_id)
            first_ts = item_row[self.INTERNAL_FIRST_TS_COL]
            last_ts = item_row[self.INTERNAL_LAST_TS_COL]
            if event_ts > last_ts:
                last_ts = event_ts
        else:
            first_ts = event_ts
            last_ts = event_ts

        item_metadata = {
            **item_features_dict,
            self.INTERNAL_FIRST_TS_COL: first_ts,
            self.INTERNAL_LAST_TS_COL: last_ts,
        }
        # Including or updating the item metadata
        self.update_item(item_id, item_metadata)

    def update_session_items_metadata(self, session: Mapping[str, Sequence[Any]]) -> None:
        features_to_retrieve = self.item_features_names + [
            self.dconf.get_feature_group(FeatureGroupType.EVENT_TS)
        ]
        # For each interaction in the session, aligns interaction features (event timestamp and item features)
        for session_item_features in zip(*[session[fname] for fname in features_to_retrieve]):
            item_features_dict = dict(zip(features_to_retrieve, session_item_features))
            # Ignoring padded items
            if item_features_dict[self.item_id_col] != self.dconf.session_padded_items_value:
                self.update_item_metadata(item_features_dict)

    @abstractmethod
    def update_item(self, item_id: ItemId, item_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def item_exists(self, item_id: ItemId) -> bool:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_item(self, item_id) -> Mapping[str, Any]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_item_ids(self, only_interacted_since_ts: Optional[int] = None) -> np.array:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_items_first_interaction_ts(
        self, only_interacted_since_ts: Optional[int] = None
    ) -> Tuple[np.array, np.array]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_last_interaction_ts(self) -> int:
        raise NotImplementedError("Not implemented")


class PandasItemsMetadataRepository(ItemsMetadataRepository):

    items_df: pd.DataFrame
    # Dummy string col to avoid Pandas converting all features to float when using .loc[] for querying, inserting or updating
    DUMMY_STR_COL = "dummy"

    def __init__(self, input_data_config: InputDataConfig) -> None:
        super().__init__(input_data_config)

        columns = {
            fname: self.dconf.get_feature_numpy_dtype(fname)
            for fname in self.dconf.get_feature_group(FeatureGroupType.ITEM_METADATA)
        }
        columns[self.ITEM_ID_COL] = self.dconf.get_feature_numpy_dtype(self.item_id_col)
        columns[self.INTERNAL_FIRST_TS_COL] = self.dconf.get_feature_numpy_dtype(self.event_ts_col)
        columns[self.INTERNAL_LAST_TS_COL] = columns[self.INTERNAL_FIRST_TS_COL]
        columns[self.DUMMY_STR_COL] = np.str

        self.items_df = df_empty(columns, self.ITEM_ID_COL)

    def update_item(self, item_id: ItemId, item_dict: Mapping[str, Any]) -> None:
        item_dict = {**item_dict, self.DUMMY_STR_COL: ""}
        # Including or updating the item metadata
        self.items_df.loc[item_id] = pd.Series(item_dict)

    def item_exists(self, item_id: ItemId) -> bool:
        return item_id in self.items_df.index

    def get_item(self, item_id: ItemId) -> Mapping[str, Any]:
        item = self.items_df.loc[item_id].to_dict()
        del item[self.DUMMY_STR_COL]
        return item

    def get_item_ids(self, only_interacted_since_ts: Optional[int] = None) -> np.array:
        df = self._get_recently_interacted_items_df(only_interacted_since_ts)
        return df.index.values

    def get_items_first_interaction_ts(
        self, only_interacted_since_ts: Optional[int] = None
    ) -> Tuple[np.array, np.array]:
        df = self._get_recently_interacted_items_df(only_interacted_since_ts)
        return (df.index.values, df[self.INTERNAL_FIRST_TS_COL].values)

    def get_last_interaction_ts(self) -> int:
        return self.items_df[self.INTERNAL_LAST_TS_COL].max()

    def _get_recently_interacted_items_df(self, only_interacted_since_ts: Optional[int] = None):
        if only_interacted_since_ts:
            df = self.items_df[
                self.items_df[self.INTERNAL_LAST_TS_COL] >= only_interacted_since_ts
            ]
        else:
            df = self.items_df
        return df


#################################################################################


class ItemsRecentPopularityRepository(ABC):
    dconf: InputDataConfig
    keep_last_days: float

    item_id_feature: str
    event_ts_feature: str

    def __init__(self, input_data_config: InputDataConfig, keep_last_days: float) -> None:
        self.dconf = input_data_config
        self.keep_last_days = keep_last_days

        self.item_id_feature = self.dconf.get_feature_group(FeatureGroupType.ITEM_ID)
        self.event_ts_feature = self.dconf.get_feature_group(FeatureGroupType.EVENT_TS)

    def append_interaction(self, item_features_dict):
        item_id = item_features_dict[self.item_id_feature]
        event_ts = item_features_dict[self.event_ts_feature]
        self._append_interaction(item_id, event_ts)

    def append_session(self, session: Mapping[str, List[Any]]) -> None:
        for item_id, ts in zip(session[self.item_id_feature], session[self.event_ts_feature]):
            # Ignoring padded items
            if item_id != self.dconf.session_padded_items_value:
                self._append_interaction(item_id, ts)

    @abstractmethod
    def _append_interaction(self, item_id: ItemId, timestamp: int) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update_stats(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def purge_old_interactions(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def log_count(self) -> int:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_candidate_items_probs(self) -> Tuple[np.array, np.array]:
        raise NotImplementedError("Not implemented")


class PandasItemsRecentPopularityRepository(ItemsRecentPopularityRepository):

    item_interactions_buffer: List[pd.DataFrame]
    item_interactions_df: pd.DataFrame
    item_pop_df: pd.DataFrame

    ITEM_ID_COL = "item_id"
    TS_COL = "ts"
    # Dummy string col to avoid Pandas converting all features to float when using .loc[] for querying, inserting or updating
    DUMMY_STR_COL = "dummy"
    COUNT_COL = "count"
    PROB_COL = "prob"

    def __init__(self, input_data_config: InputDataConfig, keep_last_days: float) -> None:
        super().__init__(input_data_config, keep_last_days)
        columns = {
            self.ITEM_ID_COL: self.dconf.get_feature_numpy_dtype(self.item_id_feature),
            self.TS_COL: np.int32,
            self.DUMMY_STR_COL: np.str,
        }
        self.item_interactions_df = df_empty(columns=columns, index=self.ITEM_ID_COL)
        self.item_pop_df = None
        self._reset_log_buffer()

    def _reset_log_buffer(self):
        self.item_interactions_buffer = []

    def _append_interaction(self, item_id: ItemId, timestamp: int) -> None:
        row_dict = {self.ITEM_ID_COL: item_id, self.TS_COL: timestamp, self.DUMMY_STR_COL: ""}

        self.item_interactions_buffer.append(pd.Series(row_dict).to_frame().T)

    def _flush_log_buffer_to_dataframe(self):
        if len(self.item_interactions_buffer) > 0:
            self.item_interactions_df = pd.concat(
                [self.item_interactions_df] + self.item_interactions_buffer
            )
            self._reset_log_buffer()

    def update_stats(self) -> None:
        self._flush_log_buffer_to_dataframe()
        self.purge_old_interactions()
        self.item_pop_df = (
            self.item_interactions_df.groupby(self.ITEM_ID_COL)
            .size()
            .to_frame(self.COUNT_COL)
            .reset_index()
        )
        self.item_pop_df[self.PROB_COL] = (
            self.item_pop_df[self.COUNT_COL] / self.item_pop_df[self.COUNT_COL].sum()
        )
        pass

    def purge_old_interactions(self) -> None:
        self._flush_log_buffer_to_dataframe()
        last_ts = self.item_interactions_df[self.TS_COL].max()
        keep_last_n_secs = self.keep_last_days * 24 * 60 * 60
        self.item_interactions_df = self.item_interactions_df[
            self.item_interactions_df[self.TS_COL] >= (last_ts - keep_last_n_secs)
        ]

    def log_count(self) -> int:
        return len(self.item_interactions_df)

    def get_candidate_items_probs(self) -> Tuple[np.array, np.array]:
        return (self.item_pop_df[self.ITEM_ID_COL].values, self.item_pop_df[self.PROB_COL].values)


#################################################################################


class ItemsSessionCoOccurrencesRepository(ABC):
    dconf: InputDataConfig
    keep_last_days: float

    def __init__(self, input_data_config: InputDataConfig, keep_last_days: float) -> None:
        self.dconf = input_data_config
        self.keep_last_days = keep_last_days

    @abstractmethod
    def append_session(self, session: Mapping[str, List[Any]]) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update_stats(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def purge_old_interactions(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def log_count(self) -> int:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_candidate_items_probs(self, item_id: ItemId) -> Tuple[np.array, np.array]:
        raise NotImplementedError("Not implemented")


class PandasItemsSessionCoOccurrencesRepository(ItemsSessionCoOccurrencesRepository):

    item_cooccurrences_log_buffer: List[pd.DataFrame]
    items_coocurrence_df: pd.DataFrame
    items_coocurence_counts_df: pd.DataFrame

    def __init__(self, input_data_config: InputDataConfig, keep_last_days: float) -> None:
        super().__init__(input_data_config, keep_last_days)
        self.items_coocurrence_df = None
        self.items_coocurence_counts_df = None
        self._reset_log_buffer()

    def _reset_log_buffer(self):
        self.item_cooccurrences_log_buffer = []

    def append_session(self, session: Mapping[str, List[Any]]) -> None:
        item_id_feat_name = self.dconf.get_feature_group(FeatureGroupType.ITEM_ID)
        ts_feat_name = self.dconf.get_feature_group(FeatureGroupType.EVENT_TS)

        min_ts = min([t for t in session[ts_feat_name] if t > 0])
        valid_pids = list(
            set(
                list(
                    [
                        p
                        for p in session[item_id_feat_name]
                        if p != self.dconf.session_padded_items_value
                    ]
                )
            )
        )

        if len(valid_pids) > 1:
            # Compute paired permutations so that we can have statistics for all items in the sequence
            items_permutations = permutations(valid_pids, 2)
            new_coo_df = pd.DataFrame(items_permutations, columns=["item_id_a", "item_id_b"])
            new_coo_df["ts"] = min_ts

            self.item_cooccurrences_log_buffer.append(new_coo_df)

    def update_stats(self) -> None:
        self._flush_log_buffer_to_dataframe()
        self.purge_old_interactions()
        self.items_coocurence_counts_df = (
            self.items_coocurrence_df.groupby(["item_id_a", "item_id_b"])
            .size()
            .to_frame("count")
            .reset_index(level=[1])
        )

    def log_count(self) -> int:
        return len(self.items_coocurrence_df)

    def _flush_log_buffer_to_dataframe(self) -> None:
        if len(self.item_cooccurrences_log_buffer) > 0:
            self.items_coocurrence_df = pd.concat(
                [self.items_coocurrence_df] + self.item_cooccurrences_log_buffer
            )
            self._reset_log_buffer()

    def purge_old_interactions(self) -> None:
        self._flush_log_buffer_to_dataframe()
        last_ts = self.items_coocurrence_df["ts"].max()
        keep_last_n_secs = self.keep_last_days * 24 * 60 * 60
        self.items_coocurrence_df = self.items_coocurrence_df[
            self.items_coocurrence_df["ts"] >= (last_ts - keep_last_n_secs)
        ]

    def get_candidate_items_probs(
        self, item_id: ItemId
    ) -> Tuple[Sequence[np.dtype], Sequence[np.dtype]]:
        candidate_items_probs: Tuple[Sequence[np.dtype], Sequence[np.dtype]] = (
            np.array([]),
            np.array([]),
        )
        if item_id in self.items_coocurence_counts_df.index:
            coocurrent_df = self.items_coocurence_counts_df.loc[item_id]
            # Dealing with cases when there is only one co-occurrent item (loc() returns a Series in that case)
            if type(coocurrent_df) is pd.Series:
                coocurrent_df = coocurrent_df.to_frame().T
            coocurrent_df["prob"] = coocurrent_df["count"] / coocurrent_df["count"].sum()

            candidate_items_probs = (
                coocurrent_df["item_id_b"].values,
                coocurrent_df["prob"].values,
            )
        return candidate_items_probs


def df_empty(columns: Mapping[str, np.dtype], index=None) -> pd.DataFrame:
    df = pd.DataFrame()
    for c in columns:
        df[c] = pd.Series(dtype=columns[c])
    return df.set_index(index)


class ItemsMetadataRepositoryFactory:
    @staticmethod
    def build(persistance_type: PersistanceType, **kwargs) -> ItemsMetadataRepository:
        if persistance_type == PersistanceType.PANDAS:
            return PandasItemsMetadataRepository(**kwargs)
        raise ValueError("Invalid persistance type!")


class ItemsRecentPopularityRepositoryFactory:
    @staticmethod
    def build(persistance_type: PersistanceType, **kwargs) -> ItemsRecentPopularityRepository:
        if persistance_type == PersistanceType.PANDAS:
            return PandasItemsRecentPopularityRepository(**kwargs)
        raise ValueError("Invalid persistance type!")


class ItemsSessionCoOccurrencesRepositoryFactory:
    @staticmethod
    def build(persistance_type: PersistanceType, **kwargs):
        if persistance_type == PersistanceType.PANDAS:
            return PandasItemsSessionCoOccurrencesRepository(**kwargs)
        raise ValueError("Invalid persistance type!")
