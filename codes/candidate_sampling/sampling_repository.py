import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from ..config.features_config import FeaturesDataType

class ItemsSamplingRepository(ABC):

    ITEM_ID_COL = "item_id"
    FIRST_TS_COL = "first_ts"
    LAST_TS_COL = "last_ts"

    def __init__(self, input_data_config):
        self.dconf = input_data_config
        self.item_features_names = self.dconf.get_item_feature_names()

    def update_item_metadata(self, item_features_dict):
        item_features_dict = item_features_dict.copy()
        item_id = item_features_dict.pop(self.dconf.get_feature_group('item_id'))

        if item_id == self.dconf.session_padded_items_value:
            return

        event_ts = item_features_dict.pop(self.dconf.get_feature_group('event_timestamp'))

        #Keeps a registry of the first and last interactions of an item
        if self.item_exists(item_id):
            item_row = self.get_item(item_id)
            first_ts = item_row[self.FIRST_TS_COL]
            last_ts = item_row[self.LAST_TS_COL]
            if event_ts > last_ts:
                last_ts = event_ts
        else:
            first_ts = event_ts
            last_ts = event_ts

        item_metadata = {**item_features_dict,
                        self.FIRST_TS_COL: first_ts,
                        self.LAST_TS_COL: last_ts,}
        #Including or updating the item metadata
        self.update_item(item_id, item_metadata)


    def update_session_items_metadata(self, session):
        #For each interaction in the session, aligns interaction features (event timestamp and item features)
        for session_item_features in zip(*[session[fname] for fname in self.item_features_names or \
                                                              fname == input_data_config.feature_groups.event_timestamp]):
            item_features_dict = dict(zip(self.item_features_names, session_item_features))
            
            self.update_item_metadata(item_features_dict)

    @abstractmethod
    def update_item(self, item_id, item_dict):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def item_exists(self, item_id):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_item(self, item_id):
        raise NotImplementedError("Not implemented")



class PandasItemsSamplingRepository(ItemsSamplingRepository):

    def __init__(self, input_data_config):        
        super().__init__(input_data_config)

        columns = { fname: self.dconf.get_feature_numpy_dtype(fname) for fname in self.dconf.get_feature_group('item_metadata') }
        columns[self.ITEM_ID_COL] = self.dconf.get_feature_numpy_dtype(self.dconf.get_feature_group('item_id'))
        columns[self.FIRST_TS_COL] = self.dconf.get_feature_numpy_dtype(self.dconf.get_feature_group('event_timestamp'))
        columns[self.LAST_TS_COL] = columns[self.FIRST_TS_COL]

        self.items_df = self._df_empty(columns, self.ITEM_ID_COL)

    def update_item(self, item_id, item_dict):
        #Including or updating the item metadata
        self.items_df.loc[item_id] = pd.Series(item_dict)

    def item_exists(self, item_id):
        return item_id in self.items_df.index

    def get_item(self, item_id):
        return self.items_df.loc[item_id]

    def _df_empty(self, column_dtype_mapping, index=None):    
        df = pd.DataFrame()
        for c in column_dtype_mapping:
            df[c] = pd.Series(dtype=column_dtype_mapping[c])
        return df.set_index(index)
