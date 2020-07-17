import numpy as np

from ..config.features_config import (
    FeatureGroups,
    FeaturesDataType,
    FeatureTypes,
    InputDataConfig,
    InstanceInfoLevel,
)


def get_input_data_config(instance_info_level: InstanceInfoLevel = InstanceInfoLevel.SESSION):
    features_schema = {
        "user_idx": FeaturesDataType.LONG,
        "sess_pid_seq": FeaturesDataType.LONG,
        "user_session": FeaturesDataType.STR,
        "sess_csid_seq": FeaturesDataType.INT,
        "sess_price_seq": FeaturesDataType.FLOAT,
        "user_seq_length_bef_sess": FeaturesDataType.INT,
        "sess_etime_seq": FeaturesDataType.LONG,
    }

    feature_groups = FeatureGroups(
        user_id="user_idx",
        item_id="sess_pid_seq",
        session_id="user_session" if instance_info_level.SESSION else None,
        implicit_feedback=None,
        event_timestamp="sess_etime_seq",
        item_metadata=["sess_csid_seq", "sess_price_seq"],
        user_metadata=["user_seq_length_bef_sess"],
        event_metadata=[],
        sequential_features=["sess_pid_seq", "sess_csid_seq", "sess_price_seq", "sess_etime_seq"]
        if instance_info_level.SESSION
        else [],
    )

    feature_types = FeatureTypes(
        categorical=["user_idx", "user_session", "sess_pid_seq", "sess_csid_seq"],
        numerical=["sess_etime_seq", "sess_price_seq", "user_seq_length_bef_sess"],
    )

    input_data_config = InputDataConfig(
        schema=features_schema,
        feature_groups=feature_groups,
        feature_types=feature_types,
        positive_interactions_only=True,
        instance_info_level=instance_info_level,
        session_padded_items_value=0,
    )

    return input_data_config


def gini_index(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = np.array(array)
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def power_law_distribution(size=100, factor=0.2):
    values = np.exp(-np.arange(0, size) * factor)
    prob_distribution = values / values.sum()
    return prob_distribution
