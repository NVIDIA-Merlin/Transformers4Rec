from transformers import TFTrainingArguments, TrainingArguments

from transformers4rec.config import trainer


def test_torch_trainer_config():
    config = trainer.T4RecTrainingArguments(
        output_dir=".", avg_session_length=20, predict_top_k=5, learning_rate=0.008
    )

    assert isinstance(config, TrainingArguments)
    assert config.learning_rate == 0.008
    assert config.predict_top_k == 5


def test_tf_trainer_config():
    config = trainer.T4RecTrainingArgumentsTF(
        output_dir=".", avg_session_length=20, predict_top_k=5, learning_rate=0.008
    )

    assert isinstance(config, TFTrainingArguments)
    assert config.learning_rate == 0.008
    assert config.predict_top_k == 5
