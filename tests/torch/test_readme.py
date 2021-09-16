import pytest

tr = pytest.importorskip("transformers4rec.torch")


def test_readme_quick_start_example_pytorch():
    schema: tr.Schema = tr.data.tabular_sequence_testing_data.schema
    # Or read schema from disk: tr.Schema().from_json(SCHEMA_PATH)
    max_sequence_length, d_model = 20, 64

    # Define input module to process tabular input-features
    input_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=max_sequence_length,
        continuous_projection=d_model,
        aggregation="concat",
        masking="causal",
    )
    # Define one or multiple prediction-tasks
    prediction_tasks = tr.NextItemPredictionTask()

    # Define a transformer-config, like the XLNet architecture
    transformer_config = tr.XLNetConfig.build(
        d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
    )
    model: tr.Model = transformer_config.to_torch_model(input_module, prediction_tasks)

    assert isinstance(model, tr.Model)
