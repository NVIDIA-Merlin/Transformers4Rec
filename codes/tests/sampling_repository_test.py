import pytest
import mock
from pytest_mock import mocker 

from ..candidate_sampling.sampling_repository import ItemsSamplingRepository, PandasItemsSamplingRepository

from ..config.features_config import InputDataConfig, FeaturesConfig, FeaturesType, FeaturesDataType, FeatureInfo




features_config = FeaturesConfig(user_id=FeatureInfo('user_idx', ftype=FeaturesType.CATEGORICAL, dtype=FeaturesDataType.LONG),
                                 item_id=FeatureInfo('sess_pid_seq', ftype=FeaturesType.CATEGORICAL, dtype=FeaturesDataType.LONG),
                                 session_id=FeatureInfo('user_session', ftype=FeaturesType.CATEGORICAL, dtype=FeaturesDataType.STR),
                                 event_timestamp=FeatureInfo('session_start_ts', ftype=FeaturesType.NUMERICAL, dtype=FeaturesDataType.LONG),
                                 implicit_feedback=None,
                                 item_metadata=[    
                                                FeatureInfo('sess_csid_seq', ftype=FeaturesType.CATEGORICAL, dtype=FeaturesDataType.INT),                                            
                                                FeatureInfo('sess_price_seq', ftype=FeaturesType.NUMERICAL, dtype=FeaturesDataType.FLOAT),
                                 ],
                                 user_metadata=[
                                                FeatureInfo('user_seq_length_bef_sess', ftype=FeaturesType.NUMERICAL, dtype=FeaturesDataType.INT)
                                 ],
                                 event_metadata=[                                                
                                                FeatureInfo('sess_etime_seq', ftype=FeaturesType.NUMERICAL, dtype=FeaturesDataType.INT),
                                 ],
                                 sequential_features=['sess_pid_seq', 'sess_csid_seq',  'sess_price_seq', 'sess_etime_seq'],
                                 )

input_data_config = InputDataConfig(features_config=features_config,
                                    positive_interactions_only=True,
                                    instance_info_level='session',
                                    session_padded_items_value=0,
                                    )

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

    def setup_method(self, mocker):
        pass

    def teardown_method(self):
        pass
    '''

    def test_update_item_metadata_insert(self, mocker):        
        repository = ItemsSamplingRepositoryInstantiable(input_data_config)

        mocker.patch.object(repository, 'item_exists') 
        repository.item_exists.return_value = False

        mocker.patch.object(repository, 'update_item') 

        item_interaction = {'session_start_ts': 1594130629, 
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
        repository = ItemsSamplingRepositoryInstantiable(input_data_config)

        mocker.patch.object(repository, 'item_exists') 
        repository.item_exists.return_value = False

        mocker.patch.object(repository, 'update_item') 

        item_interaction = {'session_start_ts': 1594130629, 
                            'sess_pid_seq': 0,
                            'sess_csid_seq': 0,
                            'sess_price_seq': 55.97,
                            }
        repository.update_item_metadata(item_interaction)
        repository.update_item.assert_not_called()

    def test_update_item_metadata_update(self, mocker):
        repository = ItemsSamplingRepositoryInstantiable(input_data_config)

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

        item_interaction = {'session_start_ts': 1594130629, 
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

    def test_insert_item_metadata(self):        
        repository = PandasItemsSamplingRepository(input_data_config)

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
        #assert item == item_features_dict