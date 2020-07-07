import pytest
import mock
from pytest_mock import mocker 

from ..candidate_sampling.sampling_repository import ItemsSamplingRepository, PandasItemsSamplingRepository

from ..config.features_config import InputDataConfig, FeaturesDataType, FeatureTypes, FeatureGroups

def get_input_data_config():
    features_schema = {'user_idx': FeaturesDataType.LONG,
                    'sess_pid_seq': FeaturesDataType.LONG,
                    'user_session': FeaturesDataType.STR,
                    'sess_csid_seq': FeaturesDataType.INT,
                    'sess_price_seq': FeaturesDataType.FLOAT,
                    'user_seq_length_bef_sess': FeaturesDataType.INT,
                    'sess_etime_seq': FeaturesDataType.LONG,
                    }

    feature_groups = FeatureGroups(
        user_id = 'user_idx',
        item_id = 'sess_pid_seq',
        session_id = 'user_session',
        implicit_feedback = None,
        event_timestamp = 'sess_etime_seq',
        item_metadata = ['sess_csid_seq', 'sess_price_seq'],
        user_metadata = ['user_seq_length_bef_sess'],
        event_metadata = [],
        sequential_features=['sess_pid_seq', 'sess_csid_seq',  'sess_price_seq', 'sess_etime_seq'],
    )

    feature_types = FeatureTypes(
        categorical = ['user_idx', 'user_session', 'sess_pid_seq', 'sess_csid_seq'],
        numerical = ['sess_etime_seq', 'sess_price_seq', 'user_seq_length_bef_sess']
    )

    input_data_config = InputDataConfig(schema=features_schema,
                                        feature_groups=feature_groups,
                                        feature_types=feature_types,
                                        positive_interactions_only=True,
                                        instance_info_level='session',
                                        session_padded_items_value=0,
                                        )

    return input_data_config

# Only to implement abstract methods, that will be mocked
class ItemsSamplingRepositoryInstantiable(ItemsSamplingRepository):

    def update_item(self, item_id, item_dict):
        raise NotImplementedError("Not implemented")

    def item_exists(self, item_id):
        raise NotImplementedError("Not implemented")

    def get_item(self, item_id):
        raise NotImplementedError("Not implemented")


class TestItemsSamplingRepository:

    '''
    @pytest.fixture(autouse=True)  # pytest will auto-run this fixture for every test in this class.
    def mocking_function(self, mocker):  # Name is arbitrary, it's the decorator that's important.
        #mocker.patch('some.func')  # pytest-mock's auto-teardown will still happen, as well.
        self.repository = ItemsSamplingRepositoryChild(ECOM_DATASET_FEATURE_CONFIG)
        mocker.patch.object(self.repository, 'update_item') 
        self.repository.update_item.return_value = None

    def setup_method(self):
        pass

    def teardown_method(self):
        pass
    '''

    def setup_method(self):
        self.input_data_config = get_input_data_config()

    def test_update_item_metadata_insert(self, mocker):        
        repository = ItemsSamplingRepositoryInstantiable(self.input_data_config)

        mocker.patch.object(repository, 'item_exists') 
        repository.item_exists.return_value = False

        mocker.patch.object(repository, 'update_item') 

        item_interaction = {'sess_etime_seq': 1594130629, 
                         'sess_pid_seq': 1,
                         'sess_csid_seq': 10,
                         'sess_price_seq': 55.97
                         }
        repository.update_item_metadata(item_interaction)
        repository.update_item.assert_called_with(1, {'first_ts': 1594130629, 
                                                           'last_ts': 1594130629,
                                                           'sess_csid_seq': 10,
                                                           'sess_price_seq': 55.97,
                                                            }
                                                           )


    def test_update_item_metadata_not_insert_padded_item(self, mocker):        
        repository = ItemsSamplingRepositoryInstantiable(self.input_data_config)

        mocker.patch.object(repository, 'item_exists') 
        repository.item_exists.return_value = False

        mocker.patch.object(repository, 'update_item') 

        item_interaction = {'sess_etime_seq': 1594130629, 
                            'sess_pid_seq': 0,
                            'sess_csid_seq': 0,
                            'sess_price_seq': 55.97,
                            }
        repository.update_item_metadata(item_interaction)
        repository.update_item.assert_not_called()

    def test_update_item_metadata_update(self, mocker):
        repository = ItemsSamplingRepositoryInstantiable(self.input_data_config)

        mocker.patch.object(repository, 'update_item') 
        repository.update_item.return_value = None

        mocker.patch.object(repository, 'item_exists') 
        repository.item_exists.return_value = True

        mocker.patch.object(repository, 'get_item') 
        repository.get_item.return_value = {'first_ts': 1594100000, 
                                            'last_ts':  1594100000,
                                            'sess_csid_seq': 10,
                                            'sess_price_seq': 55.97,
                                            }

        item_interaction = {'sess_etime_seq': 1594130629, 
                            'sess_pid_seq': 1,
                            'sess_csid_seq': 10,
                            'sess_price_seq': 58.20,
                            }
        
        repository.update_item_metadata(item_interaction)

        repository.item_exists.assert_called_with(1)
        repository.update_item.assert_called_with(1, {'first_ts': 1594100000, 
                                                      'last_ts':  1594130629,
                                                      'sess_csid_seq': 10,
                                                      'sess_price_seq': 58.20,
                                                      })



class TestPandasItemsSamplingRepository:

    def setup_method(self):
        self.input_data_config = get_input_data_config()

    def test_insert_item_metadata(self):        
        repository = PandasItemsSamplingRepository(self.input_data_config)

        item_id = 10
        item_features_dict = {
                                'sess_csid_seq': 100,
                                'sess_price_seq': 270.90,
                                'first_ts': 1594100000, 
                                'last_ts':  1594100000,
                             }
        
        assert not repository.item_exists(item_id)

        repository.update_item(item_id, item_features_dict)

        assert repository.item_exists(item_id)
        
        item = repository.get_item(item_id)
        #TODO: Fix the problem that after adding a new row using .loc[] in the DataFrame all columns are changed to float
        assert item == item_features_dict