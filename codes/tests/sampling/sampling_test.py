import pytest
import mock
import random
import numpy as np
from collections import Counter
from pytest_mock import mocker
from unittest.mock import call

from ..tests_utils import get_input_data_config, gini_index, power_law_distribution
from ...candidate_sampling.candidate_sampling import CandidateSamplingManager, CandidateSamplingConfig, SamplingStrategy, RecommendableItemSetStrategy
from ...candidate_sampling.sampling_repository import PersistanceType
from ...config.features_config import InstanceInfoLevel

def build_sampling_manager(
        sampling_strategy: SamplingStrategy,
        recommendable_items_strategy: RecommendableItemSetStrategy = RecommendableItemSetStrategy.RECENT_INTERACTIONS,
        recency_keep_interactions_last_n_days: float = 1.0,
        recent_temporal_decay_exp_factor: float = 0.002,
        instance_info_level: InstanceInfoLevel = InstanceInfoLevel.SESSION,
        remove_repeated_sampled_items: bool = True
        ):

    sampling_config = CandidateSamplingConfig(
        recommendable_items_strategy = RecommendableItemSetStrategy.RECENT_INTERACTIONS,
        sampling_strategy = sampling_strategy,
        persistance_type = PersistanceType.PANDAS,
        recency_keep_interactions_last_n_days = recency_keep_interactions_last_n_days,
        recent_temporal_decay_exp_factor = recent_temporal_decay_exp_factor,
        remove_repeated_sampled_items=remove_repeated_sampled_items
    )

    input_data_config = get_input_data_config(instance_info_level)

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

    def test_sampling_manager_cooccurrence_without_session_constructor(self):    
        with pytest.raises(ValueError):
            manager = build_sampling_manager(SamplingStrategy.ITEM_COOCURRENCE,
                                        instance_info_level = InstanceInfoLevel.INTERACTION)


    def test_append_item_interaction(self):
        manager = build_sampling_manager(SamplingStrategy.UNIFORM)
        
        item_interaction = {'sess_etime_seq': 1594130629, 
                         'sess_pid_seq': 1,
                         'sess_csid_seq': 10,
                         'sess_price_seq': 55.97
                         }
        manager.append_item_interaction(item_interaction)



    def test_session_interactions(self):
        manager = build_sampling_manager(SamplingStrategy.UNIFORM)
        
        session = {
            'sess_etime_seq': [1594133000, 1594134000, 0],
            'sess_pid_seq': [1, 2, 0],
            'sess_csid_seq': [10, 48, 0],
            'sess_price_seq': [58.20, 47.82, 0]
        }

        manager.append_session_interactions(session)


    def test_get_candidate_samples_uniform(self):
        manager = build_sampling_manager(SamplingStrategy.UNIFORM)

        self._append_interactions_popularity_biased(manager, 
                    num_sessions=100, session_len=10, max_item_id=100)

        sampled_ids = []
        for i in range(100):
            sampled_ids.extend(manager.get_candidate_samples(10))
        
        assert len(sampled_ids) == 1000

        counts = np.array(list(Counter(sampled_ids).values()), dtype=np.float32)
        #Check if the distribution of items count is long-tailed (power law)
        #according the the popularity-biased sampling used to generate interactions
        gini = gini_index(counts)
        #A resonable gini index for a close to uniform distribution
        assert gini < 0.20


    def test_get_candidate_samples_popularity(self):
        manager = build_sampling_manager(SamplingStrategy.RECENT_POPULARITY,
                        remove_repeated_sampled_items=False)

        self._append_interactions_popularity_biased(manager, 
                    num_sessions=100, session_len=5, max_item_id=100)

        manager.update_stats()

        sampled_ids = []
        for i in range(100):
            sampled_ids.extend(manager.get_candidate_samples(10))
        
        assert len(sampled_ids) == 1000

        counts = np.array(list(Counter(sampled_ids).values()), dtype=np.float32)
        #Check if the distribution of items count is long-tailed (power law)
        #according the the popularity-biased sampling used to generate interactions
        gini = gini_index(counts)
        #A gini index for a slightly long-tailed distribution
        assert gini > 0.30


    def test_get_candidate_samples_recency(self):
        manager = build_sampling_manager(SamplingStrategy.RECENCY,
                        remove_repeated_sampled_items=False,                        
                        recency_keep_interactions_last_n_days=2.0,
                        # Use an accelerated temporal exponential decay for testing, 
                        # keeping only the last two days of interaction
                        # (i.e. items first interacted 12h before have ~36% and of relevance and 24h before have ~13% compared to items interacted now)
                        recent_temporal_decay_exp_factor=2.0)

        self._append_interactions_popularity_biased(manager, 
                    num_sessions=200, session_len=5, max_item_id=100)

        sampled_ids = []
        for i in range(100):
            sampled_ids.extend(manager.get_candidate_samples(10))
        
        assert len(sampled_ids) == 1000

        #There should be two times more odds than even ids sampled, as they are more recent according to our data generation process
        even_ids_count = len([i for i in sampled_ids if i % 2 == 0])
        odd_ids_count = len([i for i in sampled_ids if i % (2+1) == 0])
        assert odd_ids_count > (even_ids_count*2)



    def test_get_candidate_samples_cooccurrence(self):
        manager = build_sampling_manager(SamplingStrategy.ITEM_COOCURRENCE,
                        remove_repeated_sampled_items=False,                        
                        recency_keep_interactions_last_n_days=1.0)

        self._append_interactions_popularity_biased(manager, 
                    num_sessions=100, session_len=5, max_item_id=100)

        manager.update_stats()
        
        ANCHOR_ITEM_ID = 1 #Co-occurs with ID 2 a lot
        COOCURRENT_ITEM_ID = 2
        
        sampled_ids = []
        for i in range(100):
            sampled_ids.extend(manager.get_candidate_samples(10, 
                                                    item_id=ANCHOR_ITEM_ID))
        
        assert len(sampled_ids) == 1000

        cooccurrent_item_ids_sorted = sorted(list(Counter(sampled_ids).items()), key=lambda x: -x[1])
        #The most co-occurrent item should be the 2
        assert cooccurrent_item_ids_sorted[0][0] == COOCURRENT_ITEM_ID


    def _append_interactions_popularity_biased(self, manager, num_sessions, session_len, max_item_id=100):
        INITIAL_TS = 1594133000
        SECS_ELAPSED_BETWEEN_SESSIONS = 1200 #20 minutes (to make 100 sessions last longer than one day)
        SECS_BETWEEN_INTERACTIONS = 60

        #We are using the factor of 0.05 on tests to smooth the distribution
        #and increase the changes to have more items sampled at least once with a few interactions and samplings
        #The value 0.2 would be a factor closer to the distribution of items popularity on real-world datasets i
        POWER_LAW_FACTOR = 0.05

        item_ids = np.arange(1,max_item_id+1)
        item_probs = power_law_distribution(size=max_item_id, factor=POWER_LAW_FACTOR)
        odd_ids_mask = item_ids % 2

        for i in range(1,num_sessions):            
            #For the first half of the sessions we sample only even ids, and even ids in the second half
            #To change items popularity over time and have different recencies for sampling
            mask_even_ids = i < (num_sessions // 2)
            items_mask = (item_ids + int(mask_even_ids)) % 2 
            item_probs_fixed = item_probs * items_mask
            item_probs_fixed = item_probs_fixed / item_probs_fixed.sum()

            sampled_item_ids = np.random.choice(item_ids, session_len, replace=False, 
                                                p=item_probs_fixed).tolist()

            #Enforces a co-ocurrence pattern
            sampled_item_ids += [1,2]

            session_start = INITIAL_TS + (i*SECS_ELAPSED_BETWEEN_SESSIONS)
            interactions_ts = list([session_start + (t*SECS_BETWEEN_INTERACTIONS) \
                                   for t in range(len(sampled_item_ids))])
            session = {
                'sess_etime_seq': interactions_ts,
                'sess_pid_seq': sampled_item_ids,
                'sess_csid_seq': list([i*10 for i in sampled_item_ids]),
                'sess_price_seq': list([i*100. for i in sampled_item_ids]),
            }
            manager.append_session_interactions(session)

    





      