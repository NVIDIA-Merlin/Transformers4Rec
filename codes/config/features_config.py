from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
from enum import Enum
import numpy as np

class InstanceInfoLevel(Enum):
    INTERACTION = "interaction"
    SESSION = "session"

class FeaturesDataType(Enum):    
    INT = "int32"
    LONG = "int64"
    FLOAT = "float16"
    DOUBLE = "float32"    
    STR = "str"
    BOOL = "bool"

    @classmethod
    def to_numpy_dtype(cls, data_type):
        switcher ={
            cls.INT: np.int32,
            cls.LONG: np.int64,
            cls.FLOAT: np.float16,
            cls.DOUBLE: np.float32,
            cls.STR: np.str,
            cls.BOOL: np.bool,
        }

        if data_type not in switcher:
            raise Exception('Invalid data type: {}'.format(data_type))

        return switcher[data_type]

@dataclass
class FeatureGroups:    
    item_id: str
    event_timestamp: str
    user_id: Optional[str]
    session_id: Optional[str]
    implicit_feedback: Optional[str]
    item_metadata: List[str] = field(default_factory=list) 
    user_metadata: List[str] = field(default_factory=list) 
    event_metadata: List[str] = field(default_factory=list)    
    sequential_features: List[str] = field(default_factory=list)

    def get_feature_group(self, group):
        if not hasattr(self, group):
            raise Exception('Invalid feature group: {}'.format(group))
        return getattr(self, group)

@dataclass 
class FeatureTypes:
    categorical: List[str] = field(default_factory=list) 
    numerical: List[str] = field(default_factory=list) 

@dataclass
class InputDataConfig:
    schema: Dict[str,FeaturesDataType]
    feature_groups: FeatureGroups    
    feature_types: FeatureTypes
    instance_info_level: InstanceInfoLevel  
    session_padded_items_value: int = field(default=0) 
    positive_interactions_only: bool = field(default=False) 

    def get_item_feature_names(self):
        return self.feature_groups.item_metadata+[self.feature_groups.item_id]

    def get_feature_dtype(self, fname):
        return self.schema[fname]

    def get_feature_numpy_dtype(self, fname):
        return FeaturesDataType.to_numpy_dtype(self.schema[fname])

    def get_feature_group(self, group):
        return self.feature_groups.get_feature_group(group)

    def get_feature_group_dtype(self, group):
        return self.feature_groups.get_feature_group(group)