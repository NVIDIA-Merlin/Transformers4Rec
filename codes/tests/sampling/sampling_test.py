import pytest
import mock
from pytest_mock import mocker
from unittest.mock import call

from ..tests_utils import get_input_data_config
from ...candidate_sampling.candidate_sampling import CandidateSamplingManager, CandidateSamplingConfig, SamplingStrategy, RecommendableItemSetStrategy

def build_sampling_manager(
        sampling_strategy: SamplingStrategy,
        recommendable_items_strategy: RecommendableItemSetStrategy = RecommendableItemSetStrategy.RECENT_INTERACTIONS,
        recency_keep_interactions_last_n_days: float = 1.0,
        recent_temporal_decay_exp_factor: float = 0.002,
        session_level: bool = True
        ):

    sampling_config = CandidateSamplingConfig(
        recommendable_items_strategy = RecommendableItemSetStrategy.RECENT_INTERACTIONS,
        sampling_strategy = sampling_strategy,
        recency_keep_interactions_last_n_days =recency_keep_interactions_last_n_days,
        recent_temporal_decay_exp_factor = recent_temporal_decay_exp_factor
    )

    input_data_config = get_input_data_config(session_level)

    sampling_manager = CandidateSamplingManager(input_data_config,
                                                sampling_config)

    return sampling_manager


class TestCandidateSamplingManager():
    
    def test_sampling_manager_uniform_constructor(self):     
        manager = build_sampling_manager(SamplingStrategy.UNIFORM)

    def test_sampling_manager_recency_constructor(self):     
        manager = build_sampling_manager(SamplingStrategy.RECENCY)

    def test_sampling_manager_popularity_constructor(self):     
        manager = build_sampling_manager(SamplingStrategy.RECENT_POPULARITY)

    def test_sampling_manager_cooccurrence_constructor(self):     
        manager = build_sampling_manager(SamplingStrategy.ITEM_COOCURRENCE)
      