from meta_model import MetaModel


def get_inference(triton_client, input_):
    """
    A mock code for the demo, to be removed
    """
    feature_group_configs = {
        "name": "session_based_features_itemid",
        "feature_map": "./datasets/session_based_features_itemid.yaml",
    }

    print("Computing output for input data:\n\t", input_)
    meta_model = MetaModel(
        feature_group_config=[feature_group_configs],
        model_type="xlnet",
        masking_task="mlm",
        max_seq_length=20,
        n_head=4,
        n_layer=2,
    )
    output = meta_model(training=True, **input_)

    print("result:\n", output["predictions"].cpu().detach().numpy())
