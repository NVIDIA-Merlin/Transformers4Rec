from dataclasses import dataclass, field
from typing import List, Union
from enum import Enum
import numpy as np

class InstanceInfoLevel(Enum):
    INTERACTION = "interaction"
    SESSION = "session"

class FeaturesType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"

class FeaturesDataType(Enum):    
    INT = "int32"
    LONG = "int64"
    FLOAT = "float16"
    DOUBLE = "float32"    
    STR = "str"
    BOOL = "bool"

    @classmethod
    def to_numpy(cls, data_type):
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
class FeatureInfo:
    name: str
    ftype: FeaturesType
    dtype: FeaturesDataType

@dataclass
class FeaturesConfig:    
    user_id: FeatureInfo
    item_id: FeatureInfo
    session_id: FeatureInfo
    implicit_feedback: FeatureInfo
    event_timestamp: FeatureInfo
    item_metadata: List[FeatureInfo]
    user_metadata: List[FeatureInfo]
    event_metadata: List[FeatureInfo] 
    sequential_features: List[str]


@dataclass
class InputDataConfig:
    features_config: FeaturesConfig    
    instance_info_level: InstanceInfoLevel  
    session_padded_items_value: int = field(default=0) 
    positive_interactions_only: bool = field(default=False) 