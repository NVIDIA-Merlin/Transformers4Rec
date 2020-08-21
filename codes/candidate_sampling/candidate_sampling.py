import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..config.features_config import FeatureGroupType, InputDataConfig, InstanceInfoLevel, ItemId
from .sampling_repository import (
    ItemsMetadataRepositoryFactory,
    ItemsRecentPopularityRepositoryFactory,
    ItemsSessionCoOccurrencesRepositoryFactory,
    PersistanceType,
)


class SamplingStrategy(Enum):
    UNIFORM = "uniform"
    RECENCY = "recency"
    RECENT_POPULARITY = "recent_popularity"
    SESSION_COOCURRENCE = "session_cooccurrence"
    # ITEM_SIMILARITY = "similarity"


@dataclass
class CandidateSamplingConfig:
    recency_keep_interactions_last_n_days: float
    persistance_type: PersistanceType
    remove_repeated_sampled_items: bool


@dataclass
class UniformCandidateSamplingConfig(CandidateSamplingConfig):
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM


@dataclass
class RecencyCandidateSamplingConfig(CandidateSamplingConfig):
    recent_temporal_decay_exp_factor: float
    sampling_strategy: SamplingStrategy = SamplingStrategy.RECENCY


@dataclass
class RecentPopularitySamplingConfig(CandidateSamplingConfig):
    sampling_strategy: SamplingStrategy = SamplingStrategy.RECENT_POPULARITY


@dataclass
class ItemCooccurrenceSamplingConfig(CandidateSamplingConfig):
    sampling_strategy: SamplingStrategy = SamplingStrategy.SESSION_COOCURRENCE


class SamplingManagerFactory:
    @classmethod
    def build(
        cls,
        input_data_config: InputDataConfig,
        sampling_strategy: SamplingStrategy,
        recency_keep_interactions_last_n_days: float = 1.0,
        recent_temporal_decay_exp_factor: float = 0.002,
        remove_repeated_sampled_items: bool = True,
        persistance_type: PersistanceType = PersistanceType.PANDAS,
    ):
        sampling_manager_class = None
        sampling_config = None

        if sampling_strategy == SamplingStrategy.UNIFORM:
            sampling_manager_class = UniformCandidateSamplingManager
            sampling_config = UniformCandidateSamplingConfig(
                sampling_strategy=sampling_strategy,
                persistance_type=persistance_type,
                recency_keep_interactions_last_n_days=recency_keep_interactions_last_n_days,
                remove_repeated_sampled_items=remove_repeated_sampled_items,
            )

        elif sampling_strategy == SamplingStrategy.RECENCY:
            sampling_manager_class = RecencyCandidateSamplingManager
            sampling_config = RecencyCandidateSamplingConfig(
                sampling_strategy=sampling_strategy,
                persistance_type=persistance_type,
                recency_keep_interactions_last_n_days=recency_keep_interactions_last_n_days,
                recent_temporal_decay_exp_factor=recent_temporal_decay_exp_factor,
                remove_repeated_sampled_items=remove_repeated_sampled_items,
            )

        elif sampling_strategy == SamplingStrategy.RECENT_POPULARITY:
            sampling_manager_class = RecentPopularityCandidateSamplingManager
            sampling_config = RecentPopularitySamplingConfig(
                sampling_strategy=sampling_strategy,
                persistance_type=persistance_type,
                recency_keep_interactions_last_n_days=recency_keep_interactions_last_n_days,
                remove_repeated_sampled_items=remove_repeated_sampled_items,
            )

        elif sampling_strategy == SamplingStrategy.SESSION_COOCURRENCE:
            sampling_manager_class = ItemCooccurrenceCandidateSamplingManager
            sampling_config = ItemCooccurrenceSamplingConfig(
                sampling_strategy=sampling_strategy,
                persistance_type=persistance_type,
                recency_keep_interactions_last_n_days=recency_keep_interactions_last_n_days,
                remove_repeated_sampled_items=remove_repeated_sampled_items,
            )

        sampling_manager = sampling_manager_class(input_data_config, sampling_config)

        return sampling_manager


class CandidateSamplingManager(ABC):
    def __init__(
        self,
        input_data_config: InputDataConfig,
        candidate_sampling_config: CandidateSamplingConfig,
    ) -> None:
        self.input_data_config = input_data_config
        self.sampling_config = candidate_sampling_config
        self.sampling_strategy = self.sampling_config.sampling_strategy
        self.persistance_type = self.sampling_config.persistance_type

        self._create_item_metadata_repository()
        self.item_id_col = self.input_data_config.get_feature_group(FeatureGroupType.ITEM_ID)
        self.event_ts_col = self.input_data_config.get_feature_group(FeatureGroupType.EVENT_TS)

    def _create_item_metadata_repository(self) -> None:
        self.items_metadata_repo = ItemsMetadataRepositoryFactory.build(
            self.persistance_type, input_data_config=self.input_data_config
        )

    def append_item_interaction(self, item_features_dict: Dict[str, Any]) -> None:
        self.items_metadata_repo.update_item_metadata(item_features_dict)

    def append_session_interactions(self, session: Mapping[str, List[Any]]) -> None:
        if self.input_data_config.instance_info_level != InstanceInfoLevel.SESSION:
            raise Exception(
                "Appending session interactions is only allowed when InstanceInfoLevel == SESSION"
            )
        self.items_metadata_repo.update_session_items_metadata(session)

    def update_stats(self) -> None:
        pass

    def get_candidate_samples(
        self,
        n_samples: int,
        item_id: Optional[ItemId] = None,
        return_item_features: bool = False,
        ignore_items: Optional[List[ItemId]] = [],
    ) -> Union[Sequence[ItemId], Mapping[ItemId, Mapping[str, Any]]]:
        # To ensure that after removing sessions from the current session we have the required number of samples
        SAMPLES_MULITPLIER = 2

        sampled_item_ids = self.get_candidate_samples_item_ids(
            n_samples * SAMPLES_MULITPLIER, item_id=item_id
        )

        # Removing repeated entries
        if self.sampling_config.remove_repeated_sampled_items:
            sampled_item_ids = list(set(sampled_item_ids))

        # Removing samples from the ignore list
        if ignore_items is not None:
            sampled_item_ids = list([i for i in sampled_item_ids if i not in ignore_items])

        # Shuffles the list inplace
        random.shuffle(sampled_item_ids)

        # Limiting the number of negative samples
        sampled_item_ids = sampled_item_ids[:n_samples]

        if return_item_features:
            sampled_items = (
                sampled_item_ids,
                self.items_metadata_repo.get_items(sampled_item_ids),
            )
            return sampled_items
        else:
            return sampled_item_ids

    def get_start_sliding_window(self):
        last_interaction_ts = self.items_metadata_repo.get_last_interaction_ts()
        ts_start_sliding_window = last_interaction_ts - int(
            self.sampling_config.recency_keep_interactions_last_n_days * 24 * 60 * 60
        )
        return ts_start_sliding_window

    def get_samples_from_prob_distribution(
        self, item_ids: np.array, n_samples: int, item_probs: Optional[np.array]
    ) -> List[ItemId]:
        sampled_item_ids = []
        if len(item_ids):
            sampled_item_ids = np.random.choice(
                item_ids, min(n_samples, len(item_ids)), replace=False, p=item_probs
            ).tolist()
        return sampled_item_ids

    @abstractmethod
    def get_candidate_samples_item_ids(
        self, n_samples: int, item_id: Optional[ItemId] = None
    ) -> Sequence[ItemId]:
        pass


class UniformCandidateSamplingManager(CandidateSamplingManager):
    def __init__(
        self,
        input_data_config: InputDataConfig,
        candidate_sampling_config: UniformCandidateSamplingConfig,
    ) -> None:
        super().__init__(input_data_config, candidate_sampling_config)

    def get_candidate_samples_item_ids(
        self, n_samples: int, item_id: Optional[ItemId] = None
    ) -> Sequence[ItemId]:
        ts_start_sliding_window = self.get_start_sliding_window()
        item_ids = self.items_metadata_repo.get_item_ids(
            only_interacted_since_ts=ts_start_sliding_window
        )

        sampled_item_ids = self.get_samples_from_prob_distribution(
            item_ids=item_ids, item_probs=None, n_samples=n_samples
        )
        return sampled_item_ids


class RecencyCandidateSamplingManager(CandidateSamplingManager):
    def __init__(
        self,
        input_data_config: InputDataConfig,
        candidate_sampling_config: RecencyCandidateSamplingConfig,
    ) -> None:
        super().__init__(input_data_config, candidate_sampling_config)

    def get_candidate_samples_item_ids(
        self, n_samples: int, item_id: Optional[ItemId] = None
    ) -> Sequence[ItemId]:
        ts_start_sliding_window = self.get_start_sliding_window()

        item_ids, first_interaction_ts = self.items_metadata_repo.get_items_first_interaction_ts(
            only_interacted_since_ts=ts_start_sliding_window
        )

        last_global_ts = first_interaction_ts.max()
        item_days_age = (last_global_ts - first_interaction_ts) / (60 * 60 * 24)
        items_relevance_time_decayed = self._prod_relevance_decay(item_days_age)

        items_relevance_time_decayed_norm = (
            items_relevance_time_decayed / items_relevance_time_decayed.sum()
        )

        sampled_item_ids = self.get_samples_from_prob_distribution(
            item_ids=item_ids, item_probs=items_relevance_time_decayed_norm, n_samples=n_samples
        )
        return sampled_item_ids

    def _prod_relevance_decay(self, days_age: int):
        return np.exp(-days_age * self.sampling_config.recent_temporal_decay_exp_factor)


class RecentPopularityCandidateSamplingManager(CandidateSamplingManager):
    def __init__(
        self,
        input_data_config: InputDataConfig,
        candidate_sampling_config: RecentPopularitySamplingConfig,
    ) -> None:
        super().__init__(input_data_config, candidate_sampling_config)

        self._create_interactions_repository()

    def _create_interactions_repository(self):
        self.items_recent_popularity_repo = ItemsRecentPopularityRepositoryFactory.build(
            self.persistance_type,
            input_data_config=self.input_data_config,
            keep_last_days=self.sampling_config.recency_keep_interactions_last_n_days,
        )

    def append_item_interaction(self, item_features_dict: Dict[str, Any]) -> None:
        super().append_item_interaction(item_features_dict)
        self.items_recent_popularity_repo.append_interaction(item_features_dict)

    def append_session_interactions(self, session: Mapping[str, List[Any]]) -> None:
        super().append_session_interactions(session)
        self.items_recent_popularity_repo.append_session(session)

    def get_candidate_samples_item_ids(
        self, n_samples: int, item_id: Optional[ItemId] = None
    ) -> Sequence[ItemId]:
        item_ids, item_probs = self.items_recent_popularity_repo.get_candidate_items_probs()

        sampled_item_ids = self.get_samples_from_prob_distribution(
            item_ids=item_ids, item_probs=item_probs, n_samples=n_samples
        )
        return sampled_item_ids

    def update_stats(self) -> None:
        self.items_recent_popularity_repo.update_stats()


class ItemCooccurrenceCandidateSamplingManager(CandidateSamplingManager):
    def __init__(
        self,
        input_data_config: InputDataConfig,
        candidate_sampling_config: ItemCooccurrenceSamplingConfig,
    ) -> None:
        super().__init__(input_data_config, candidate_sampling_config)
        self._check_config()

        self._create_interactions_repository()

    def _check_config(self) -> None:
        if self.input_data_config.instance_info_level != InstanceInfoLevel.SESSION:
            raise ValueError(
                'The "{}" strategy is only available the the instance info level is {}'.format(
                    SamplingStrategy.SESSION_COOCURRENCE, InstanceInfoLevel.SESSION
                )
            )

    def _create_interactions_repository(self):
        self.items_session_cooccurrences_repo = ItemsSessionCoOccurrencesRepositoryFactory.build(
            self.persistance_type,
            input_data_config=self.input_data_config,
            keep_last_days=self.sampling_config.recency_keep_interactions_last_n_days,
        )

    def append_item_interaction(self, item_features_dict: Dict[str, Any]) -> None:
        super().append_item_interaction(item_features_dict)
        self.items_recent_popularity_repo.append_interaction(item_features_dict)

    def append_session_interactions(self, session: Mapping[str, List[Any]]) -> None:
        super().append_session_interactions(session)
        self.items_session_cooccurrences_repo.append_session(session)

    def get_candidate_samples_item_ids(
        self, n_samples: int, item_id: Optional[ItemId] = None
    ) -> Sequence[ItemId]:

        item_ids, item_probs = self.items_session_cooccurrences_repo.get_candidate_items_probs(
            item_id
        )

        sampled_item_ids = self.get_samples_from_prob_distribution(
            item_ids=item_ids, item_probs=item_probs, n_samples=n_samples
        )
        return sampled_item_ids

    def update_stats(self) -> None:
        self.items_session_cooccurrences_repo.update_stats()
