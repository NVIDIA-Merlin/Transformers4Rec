from ..config.features_config import InputDataConfig, FeaturesDataType, FeatureTypes, FeatureGroups, InstanceInfoLevel

def get_input_data_config(session_level: bool = True):
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
                                        instance_info_level=InstanceInfoLevel.SESSION,
                                        session_padded_items_value=0,
                                        )

    return input_data_config