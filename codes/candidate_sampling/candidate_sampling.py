import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class RecommendableItemSetStrategy(Enum):
    GLOBAL = "global"
    PAST = "past"
    RECENT = "recent"    
    BATCH = "batch"

class SamplingStrategy(Enum):
    UNIFORM = "uniform"
    RECENCY = "recency"
    RECENT_POPULARITY = "popularity"
    COOCURRENCE = "cooccurrence"
    SIMILARITY = "similarity"

@dataclass
class CandidateSamplingConfig:
    recommendable_items_strategy: RecommendableItemSetStrategy
    sampling_strategy: SamplingStrategy
    ignore_session_items_on_sampling: bool


'''
class CandidateSamplingManager():

    def __init__(self, input_data_config, repository, candidate_sampling_config):
        self.input_data_config = input_data_config
        self.repository = repository
        self.candidate_sampling_config = candidate_sampling_config


    def get_candidate_samples(self, item_id: ItemIdType, n_samples: int) -> Sequence[ItemIdType]:
        item_ids, item_probs = self.get_candidate_items_probs(item_id)

        samples = np.random.choice(item_ids, min(n_samples, len(item_ids)), replace=False, 
                                   p=item_probs).tolist()
        return samples
'''