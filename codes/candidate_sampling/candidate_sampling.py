import numpy as np
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Sequence, Mapping, Union, Optional, NewType, Any, TypeVar

from ..config.features_config import (ItemId, FeatureGroupType, 
                                      InstanceInfoLevel, InputDataConfig)

from .sampling_repository import (ItemsMetadataRepository, PandasItemsMetadataRepository,  
                                                      PandasItemsSessionCoOccurrencesRepository, ItemsRecentPopularityRepository, 
                                                      PandasItemsRecentPopularityRepository)


class RecommendableItemSetStrategy(Enum):
    RECENT_INTERACTIONS = "recent"

class SamplingStrategy(Enum):
    UNIFORM = "uniform"
    RECENCY = "recency"
    RECENT_POPULARITY = "popularity"
    ITEM_COOCURRENCE = "cooccurrence"
    #ITEM_SIMILARITY = "similarity"

@dataclass
class CandidateSamplingConfig:
    recommendable_items_strategy: RecommendableItemSetStrategy
    sampling_strategy: SamplingStrategy
    recency_keep_interactions_last_n_days: float
    recent_temporal_decay_exp_factor: float
    


class CandidateSamplingManager():

    def __init__(self, input_data_config: InputDataConfig, 
                        candidate_sampling_config: CandidateSamplingConfig) -> None:
        self.input_data_config = input_data_config
        self.sampling_config = candidate_sampling_config
        self.sampling_strategy = self.sampling_config.sampling_strategy      
        self._check_config()  
    
        self._create_repositories()
        self.item_id_col = self.input_data_config.get_feature_group(FeatureGroupType.ITEM_ID)
        self.event_ts_col = self.input_data_config.get_feature_group(FeatureGroupType.EVENT_TS)

    def _check_config(self) -> None:
        if self.sampling_strategy == SamplingStrategy.ITEM_COOCURRENCE and \
           self.input_data_config.instance_info_level != InstanceInfoLevel.SESSION:
           raise Exception('The "{}" strategy is only available the the instance info level is {}' \
                            .format(SamplingStrategy.ITEM_COOCURRENCE, self.input_data_config.InstanceInfoLevel.SESSION))  


    def _create_item_metadata_repo(self) -> None:
        self.items_metadata_repo = PandasItemsMetadataRepository(self.input_data_config)

    def _create_items_recent_popularity_repo(self) -> None:
        self.items_recent_popularity_repo = PandasItemsRecentPopularityRepository(
                        self.input_data_config, self.sampling_config.recency_keep_interactions_last_n_days
                )

    def _create_items_session_cooccurrences_repo(self) -> None:
        self.items_session_cooccurrences_repo = PandasItemsSessionCoOccurrencesRepository(
                        self.input_data_config, self.sampling_config.recency_keep_interactions_last_n_days
                )

    def _create_repositories(self) -> None:
        self._create_item_metadata_repo()

        if self.sampling_config.sampling_strategy in [SamplingStrategy.RECENT_POPULARITY,
                                                                SamplingStrategy.ITEM_COOCURRENCE]:
            self._create_items_recent_popularity_repo()

        if self.sampling_config.sampling_strategy == SamplingStrategy.ITEM_COOCURRENCE:
            self._create_items_session_cooccurrences_repo()
        

    def append_item_interaction(self, item_features_dict: Dict[str,Any]) -> None:
        self.items_metadata_repo.update_item_metadata(item_features_dict)

        if self.sampling_config.sampling_strategy in [SamplingStrategy.RECENT_POPULARITY,
                                                                SamplingStrategy.ITEM_COOCURRENCE]:
            self.items_recent_popularity_repo.append_interaction(item_features_dict)

    def append_session_interactions(self, session: Mapping[str,List[Any]]) -> None:
        if self.input_data_config.instance_info_level != InstanceInfoLevel.SESSION:
            raise Exception('Appending session interactions is only allowed when InstanceInfoLevel == SESSION')

        self.items_metadata_repo.update_session_items_metadata(session)

        if self.sampling_config.sampling_strategy in [SamplingStrategy.RECENT_POPULARITY,
                                                                SamplingStrategy.ITEM_COOCURRENCE]:
            self.items_metadata_repo.append_session(session)

        if self.sampling_config.sampling_strategy == SamplingStrategy.ITEM_COOCURRENCE:
            self.items_session_cooccurrences_repo.append_session(session)


    def get_candidate_samples(self, item_id: ItemId, n_samples: int, ignore_items: Optional[List[ItemId]]) -> Sequence[ItemId]:
        #To ensure that after removing sessions from the current session we have the required number of samples
        SAMPLES_MULITPLIER = 2

        if self.sampling_strategy == SamplingStrategy.UNIFORM:
            sampled_item_ids = self._get_neg_samples_uniform(n_samples*SAMPLES_MULITPLIER)

        elif self.sampling_strategy == SamplingStrategy.RECENCY:
            sampled_item_ids = self._get_neg_samples_recency(n_samples*SAMPLES_MULITPLIER)

        elif self.sampling_strategy == SamplingStrategy.RECENT_POPULARITY:
            sampled_item_ids = self._get_neg_samples_recent_popularity(n_samples*SAMPLES_MULITPLIER)

        elif self.sampling_strategy == SamplingStrategy.ITEM_COOCURRENCE:
            sampled_item_ids = self._get_neg_samples_cooccurrence(item_id*SAMPLES_MULITPLIER)

            #If there is not enough co-occurring items, complete the number of 
            # neg. samples with global popularity-biased sampling
            if len(sampled_item_ids) < n_samples:
                sampled_item_ids += self._get_neg_samples_recent_popularity(n_samples)

        #Removing repeated entries
        sampled_item_ids = list(set(sampled_item_ids))

        #Removing samples from the ignore list
        if ignore_items is not None:
            sampled_item_ids = list([i for i in sampled_item_ids if i not in ignore_items])

        #Shuffles the list inplace
        random.shuffle(sampled_item_ids)

        return sampled_item_ids[:n_samples]

    def _get_neg_samples_uniform(self, n_samples: int):
        item_ids = self.items_metadata_repo.get_item_ids()
        sampled_item_ids = self._get_samples_from_prob_distribution(item_ids,item_probs= None, 
                                                                    n_samples=n_samples)
        return sampled_item_ids

    def _prod_relevance_decay(self, days_age : int):
        return np.exp(-days_age * self.recent_temporal_decay_exp_factor)

    def _get_neg_samples_recency(self, n_samples: int):

        item_ids, first_interaction_ts = self.items_metadata_repo.get_items_first_interaction_ts()
        
        last_global_ts = first_interaction_ts.max()
        item_days_age = (last_global_ts - first_interaction_ts) / (60 * 60 * 24)
        items_relevance_time_decayed = self._prod_relevance_decay(item_days_age)

        items_relevance_time_decayed_norm = items_relevance_time_decayed / items_relevance_time_decayed.sum()
        
        sampled_item_ids = self._get_samples_from_prob_distribution(item_ids,
                                                                    item_probs= items_relevance_time_decayed_norm, 
                                                                    n_samples=n_samples)
        return sampled_item_ids


    def _get_neg_samples_recent_popularity(self, n_samples: int):
        item_ids, item_probs = self.items_recent_popularity_repo \
                                .get_candidate_items_probs()

        sampled_item_ids = self._get_samples_from_prob_distribution(item_ids,
                                                                    item_probs=item_probs, 
                                                                    n_samples=n_samples)
        return sampled_item_ids


    def _get_neg_samples_cooccurrence(self, item_id: ItemId, n_samples: int):
        item_ids, item_probs = self.items_session_cooccurrences_repo \
                                .get_candidate_items_probs(item_id)

        sampled_item_ids = self._get_samples_from_prob_distribution(item_ids,
                                                                    item_probs=item_probs, 
                                                                    n_samples=n_samples)
        return sampled_item_ids


    def _get_samples_from_prob_distribution(item_ids: np.array, item_probs: Optional[np.array], 
                                            n_samples: int) -> List[ItemId]:
        sampled_item_ids = np.random.choice(item_ids, min(n_samples, len(item_ids)), 
                                            replace=False, p=item_probs).tolist()
        return sampled_item_ids