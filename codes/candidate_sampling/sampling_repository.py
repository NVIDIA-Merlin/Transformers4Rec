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
        self.fconf = input_data_config.features_config

    def update_session_items_metadata(self, session):
        metadata_feature_names = self.fconf.item_metadata
        #For each interaction in the session
        for session_item_features in zip(*[session[f] for f in metadata_feature_names]):
            item_features_dict = dict(zip(metadata_feature_names, session_item_features))
            
            #TODO: Temporary implementation. Reprocess the eCommerce dataset to include correct event timestamp (instead of the session start timestamp) for each interaction within the session, so that the timestamp can be zipped with the other item metadata features
            item_features_dict[self.fconf.event_timestamp] = session[self.fconf.event_timestamp]
            
            self.update_item_metadata(item_features_dict)
                        

    def update_item_metadata(self, item_features_dict):
        item_features_dict = item_features_dict.copy()
        item_id = item_features_dict.pop(self.fconf.item_id.name)

        if item_id == self.dconf.session_padded_items_value:
            return

        event_ts = item_features_dict.pop(self.fconf.event_timestamp.name)

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

        columns = {f.name: FeaturesDataType.to_numpy(f.dtype) for f in self.fconf.item_metadata}
        columns[self.ITEM_ID_COL] = FeaturesDataType.to_numpy(self.fconf.item_id.dtype)
        columns[self.FIRST_TS_COL] = FeaturesDataType.to_numpy(self.fconf.event_timestamp.dtype)
        columns[self.LAST_TS_COL] = FeaturesDataType.to_numpy(self.fconf.event_timestamp.dtype)

        self.items_df = pd.DataFrame(columns=columns).set_index(self.ITEM_ID_COL)

    def update_item(self, item_id, item_dict):
        #Including or updating the item metadata
        self.items_df.loc[item_id] = pd.Series(item_dict)

    def item_exists(self, item_id):
        return item_id in self.items_df.index

    def get_item(self, item_id):
        return self.items_df.loc[item_id]
