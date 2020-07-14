from itertools import permutations
from unittest.mock import call

import numpy as np
from pytest_mock import mocker

from ...candidate_sampling.sampling_repository import (
    ItemsMetadataRepository,
    ItemsRecentPopularityRepository,
    PandasItemsMetadataRepository,
    PandasItemsRecentPopularityRepository,
    PandasItemsSessionCoOccurrencesRepository,
)
from ..tests_utils import get_input_data_config


# Only to implement abstract methods, that will be mocked
class ItemsMetadataRepositoryInstantiable(ItemsMetadataRepository):
    def update_item(self, item_id, item_dict):
        raise NotImplementedError("Not implemented")

    def item_exists(self, item_id):
        raise NotImplementedError("Not implemented")

    def get_item(self, item_id):
        raise NotImplementedError("Not implemented")

    def get_item_ids(self, only_interacted_since_ts):
        raise NotImplementedError("Not implemented")

    def get_items_first_interaction_ts(self, only_interacted_since_ts):
        raise NotImplementedError("Not implemented")

    def get_last_interaction_ts(self) -> int:
        raise NotImplementedError("Not implemented")


class TestItemsMetadataRepository:
    def setup_method(self):
        self.input_data_config = get_input_data_config()

    def test_update_item_metadata_insert(self, mocker):
        repository = ItemsMetadataRepositoryInstantiable(self.input_data_config)

        mocker.patch.object(repository, "item_exists")
        repository.item_exists.return_value = False

        mocker.patch.object(repository, "update_item")

        item_interaction = {
            "sess_etime_seq": 1594130629,
            "sess_pid_seq": 1,
            "sess_csid_seq": 10,
            "sess_price_seq": 55.97,
        }
        repository.update_item_metadata(item_interaction)
        repository.update_item.assert_called_with(
            1,
            {
                "first_ts": 1594130629,
                "last_ts": 1594130629,
                "sess_csid_seq": 10,
                "sess_price_seq": 55.97,
            },
        )

    def test_update_item_metadata_update(self, mocker):
        repository = ItemsMetadataRepositoryInstantiable(self.input_data_config)

        mocker.patch.object(repository, "update_item")
        repository.update_item.return_value = None

        mocker.patch.object(repository, "item_exists")
        repository.item_exists.return_value = True

        mocker.patch.object(repository, "get_item")
        repository.get_item.return_value = {
            "first_ts": 1594100000,
            "last_ts": 1594100000,
            "sess_csid_seq": 10,
            "sess_price_seq": 55.97,
        }

        item_interaction = {
            "sess_etime_seq": 1594130629,
            "sess_pid_seq": 1,
            "sess_csid_seq": 10,
            "sess_price_seq": 58.20,
        }

        repository.update_item_metadata(item_interaction)

        repository.item_exists.assert_called_with(1)
        repository.update_item.assert_called_with(
            1,
            {
                "first_ts": 1594100000,
                "last_ts": 1594130629,
                "sess_csid_seq": 10,
                "sess_price_seq": 58.20,
            },
        )

    def test_update_session_items_metadata(self, mocker):
        repository = ItemsMetadataRepositoryInstantiable(self.input_data_config)

        mocker.patch.object(repository, "update_item_metadata")

        session = {
            "sess_etime_seq": [1594133000, 1594134000, 0],
            "sess_pid_seq": [1, 2, 0],
            "sess_csid_seq": [10, 48, 0],
            "sess_price_seq": [58.20, 47.82, 0],
        }

        repository.update_session_items_metadata(session)

        calls = [
            call(
                {
                    "sess_etime_seq": 1594133000,
                    "sess_pid_seq": 1,
                    "sess_csid_seq": 10,
                    "sess_price_seq": 58.20,
                }
            ),
            call(
                {
                    "sess_etime_seq": 1594134000,
                    "sess_pid_seq": 2,
                    "sess_csid_seq": 48,
                    "sess_price_seq": 47.82,
                }
            ),
        ]

        repository.update_item_metadata.assert_has_calls(calls, any_order=False)
        # Ensures that padded item ids (0) are not inserted/updated
        assert repository.update_item_metadata.call_count == 2


class TestPandasItemsMetadataRepository:
    def setup_method(self):
        self.input_data_config = get_input_data_config()

    def test_insert_item_metadata(self):
        repository = PandasItemsMetadataRepository(self.input_data_config)

        item_id = 10
        item_features_dict = {"sess_csid_seq": 100, "sess_price_seq": 270.90}

        assert not repository.item_exists(item_id)

        repository.update_item(item_id, item_features_dict)

        assert repository.item_exists(item_id)

        item = repository.get_item(item_id)
        assert item["sess_csid_seq"] == item_features_dict["sess_csid_seq"]
        assert item["sess_price_seq"] == item_features_dict["sess_price_seq"]

    def test_get_items_last_interacted_recently(self):

        repository = PandasItemsMetadataRepository(self.input_data_config)

        item_features_dict = {
            "sess_pid_seq": 10,
            "sess_csid_seq": 100,
            "sess_price_seq": 270.90,
            "sess_etime_seq": 1594100000,
        }
        repository.update_item_metadata(item_features_dict)

        item_features_dict = {
            "sess_pid_seq": 20,
            "sess_csid_seq": 120,
            "sess_price_seq": 605.75,
            "sess_etime_seq": 1594200000,
        }
        repository.update_item_metadata(item_features_dict)

        item_ids = repository.get_item_ids(only_interacted_since_ts=1594200000)

        # Ensures that the id from the first interaction is not returned, as its last interaction was before the reference date
        assert len(item_ids) == 1
        assert np.all(item_ids == np.array([20]))

    def test_get_items_first_interaction_ts(self):

        repository = PandasItemsMetadataRepository(self.input_data_config)

        item_features_dict = {
            "sess_pid_seq": 10,
            "sess_csid_seq": 100,
            "sess_price_seq": 270.90,
            "sess_etime_seq": 1594100000,
        }
        repository.update_item_metadata(item_features_dict)

        item_features_dict = {
            "sess_pid_seq": 10,
            "sess_csid_seq": 100,
            "sess_price_seq": 270.90,
            "sess_etime_seq": 1594300000,
        }
        repository.update_item_metadata(item_features_dict)

        item_features_dict = {
            "sess_pid_seq": 20,
            "sess_csid_seq": 120,
            "sess_price_seq": 605.75,
            "sess_etime_seq": 1594200000,
        }
        repository.update_item_metadata(item_features_dict)

        item_ids, first_ts = repository.get_items_first_interaction_ts()
        assert np.all(item_ids == np.array([10, 20]))
        assert np.all(first_ts == np.array([1594100000, 1594200000]))


##########################################################

# Only to implement abstract methods, that will be mocked
class ItemsRecentPopularityRepositoryInstantiable(ItemsRecentPopularityRepository):
    def _append_interaction(self, item_id, timestamp):
        raise NotImplementedError("Not implemented")

    def update_stats(self):
        raise NotImplementedError("Not implemented")

    def purge_old_interactions(self):
        raise NotImplementedError("Not implemented")

    def log_count(self):
        raise NotImplementedError("Not implemented")

    def get_candidate_items_probs(self):
        raise NotImplementedError("Not implemented")


class TestItemsRecentPopularityRepository:
    def test_append_session(self, mocker):
        input_data_config = get_input_data_config()
        repository = ItemsRecentPopularityRepositoryInstantiable(
            input_data_config, keep_last_days=1.0
        )

        mocker.patch.object(repository, "_append_interaction")

        session = {
            "sess_etime_seq": [1594133000, 1594134000, 0],
            "sess_pid_seq": [1, 2, 0],
            "sess_csid_seq": [10, 48, 0],
            "sess_price_seq": [58.20, 47.82, 0],
        }

        repository.append_session(session)

        calls = [call(1, 1594133000), call(2, 1594134000)]

        repository._append_interaction.assert_has_calls(calls, any_order=False)
        # Ensures that padded item ids (0) are not inserted/updated
        assert repository._append_interaction.call_count == 2


class TestPandasItemsRecentPopularityRepository:
    def setup_method(self):
        self.input_data_config = get_input_data_config()
        self.repository = PandasItemsRecentPopularityRepository(
            self.input_data_config, keep_last_days=1.0
        )

    def test_append_interaction(self):
        self.repository.append_interaction({"sess_pid_seq": 1, "sess_etime_seq": 1594133000})
        self.repository.append_interaction({"sess_pid_seq": 21, "sess_etime_seq": 1594134000})
        self.repository.update_stats()
        assert self.repository.log_count() == 2

    def test_purge_old_interaction(self):
        # 03-07-2020 00:00
        self.repository.append_interaction({"sess_pid_seq": 10, "sess_etime_seq": 1593734400})
        # 03-07-2020 13:00
        self.repository.append_interaction({"sess_pid_seq": 11, "sess_etime_seq": 1593781200})
        self.repository.purge_old_interactions()
        assert self.repository.log_count() == 2
        # 04-07-2020 12:00
        self.repository.append_interaction({"sess_pid_seq": 15, "sess_etime_seq": 1593864000})
        # Should remove the first interaction (from previous day)
        self.repository.purge_old_interactions()
        assert self.repository.log_count() == 2

    def test_get_candidate_items_probs(self):
        self.repository.append_interaction({"sess_pid_seq": 10, "sess_etime_seq": 1593734400})
        self.repository.append_interaction({"sess_pid_seq": 10, "sess_etime_seq": 1593734400})
        self.repository.append_interaction({"sess_pid_seq": 10, "sess_etime_seq": 1593734400})
        self.repository.append_interaction({"sess_pid_seq": 11, "sess_etime_seq": 1593734400})
        self.repository.append_interaction({"sess_pid_seq": 11, "sess_etime_seq": 1593734400})
        self.repository.append_interaction({"sess_pid_seq": 12, "sess_etime_seq": 1593734400})
        self.repository.update_stats()

        interactions_count = self.repository.log_count()
        assert interactions_count == 6

        item_ids, probs = self.repository.get_candidate_items_probs()
        assert len(item_ids) == len(probs) == 3
        items_prob = dict(zip(item_ids, probs))
        assert items_prob[10] == 3 / interactions_count
        assert items_prob[11] == 2 / interactions_count
        assert items_prob[12] == 1 / interactions_count


############################################################


class TestPandasItemsSessionCoOccurrencesRepository:
    def setup_method(self):
        self.input_data_config = get_input_data_config()
        self.keep_last_days = 1.0

        self.repository = PandasItemsSessionCoOccurrencesRepository(
            self.input_data_config, keep_last_days=self.keep_last_days
        )

    def test_append_session(self):
        session = {
            "sess_pid_seq": [1, 2, 0],
            "sess_etime_seq": [1594133000, 1594134000, 0],
            "sess_csid_seq": [10, 48, 0],
            "sess_price_seq": [58.20, 47.82, 0],
        }

        self.repository.append_session(session)
        self.repository.update_stats()

        assert self.repository.log_count() == len(list(permutations(range(2), 2)))

    def test_append_two_session(self):
        session_1 = {
            "sess_pid_seq": [1, 2, 0],
            "sess_etime_seq": [1594133000, 1594134000, 0],
            "sess_csid_seq": [10, 48, 0],
            "sess_price_seq": [58.20, 47.82, 0],
        }

        session_2 = {
            "sess_pid_seq": [1, 4, 5],
            "sess_etime_seq": [1594135000, 1594136000, 1594137000],
            "sess_csid_seq": [110, 148, 10],
            "sess_price_seq": [26.58, 72.25, 45.84],
        }

        self.repository.append_session(session_1)
        self.repository.append_session(session_2)
        self.repository.update_stats()

        assert self.repository.log_count() == len(list(permutations(range(2), 2))) + len(
            list(permutations(range(3), 2))
        )

    def test_get_candidate_item_probs(self):
        session_1 = {
            "sess_pid_seq": [1, 2, 3],
            "sess_etime_seq": [1594133000, 1594134000, 0],
            "sess_csid_seq": [10, 48, 0],
            "sess_price_seq": [58.20, 47.82, 0],
        }

        session_2 = {
            "sess_pid_seq": [1, 3, 4],
            "sess_etime_seq": [1594135000, 1594136000, 1594137000],
            "sess_csid_seq": [110, 148, 10],
            "sess_price_seq": [26.58, 72.25, 45.84],
        }

        self.repository.append_session(session_1)
        self.repository.append_session(session_2)
        self.repository.update_stats()

        candidate_items_probs = self.repository.get_candidate_items_probs(1)
        items, probs = candidate_items_probs
        assert len(items) == len(probs) == 3
        assert list(sorted(items)) == [2, 3, 4]

        item_probs = dict(zip(items, probs))
        assert item_probs[2] == 0.25
        assert item_probs[3] == 0.5
        assert item_probs[4] == 0.25

        items, probs = self.repository.get_candidate_items_probs(-1)
        assert len(items) == len(probs) == 0

    def test_purge_old_interactions(self):
        session_1 = {
            "sess_pid_seq": [1, 2],
            # 02-07-2020 0:00, 02-07-2020 0:05
            "sess_etime_seq": [1593648000, 1593648300],
            "sess_csid_seq": [10, 48],
            "sess_price_seq": [58.20, 47.82],
        }

        self.repository.append_session(session_1)
        self.repository.update_stats()
        assert self.repository.log_count() == 2

        session_2 = {
            "sess_pid_seq": [1, 3],
            # 02-07-2020 13:00, 02-07-2020 13:05
            "sess_etime_seq": [1593694800, 1593695100],
            "sess_csid_seq": [110, 148],
            "sess_price_seq": [26.58, 72.25],
        }

        self.repository.append_session(session_2)
        self.repository.update_stats()
        assert self.repository.log_count() == 4

        session_3 = {
            "sess_pid_seq": [1, 4],
            # 03-07-2020 0:10, 03-07-2020 0:15
            "sess_etime_seq": [1593781200, 1593781500],
            "sess_csid_seq": [110, 148],
            "sess_price_seq": [26.58, 72.25],
        }

        self.repository.append_session(session_3)
        # This update should remove logs before 1 day
        self.repository.update_stats()
        assert self.repository.log_count() == 4
