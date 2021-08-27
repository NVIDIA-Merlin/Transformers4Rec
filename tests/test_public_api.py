import pytest

import transformers4rec as tr

transformer_config_names = [
    "ReformerConfig",
    "XLNetConfig",
    "ElectraConfig",
    "LongformerConfig",
    "GPT2Config",
]


@pytest.mark.parametrize("config", transformer_config_names)
def test_transformer_config_imports(config):
    config_cls = getattr(tr, config)

    assert issubclass(config_cls, tr.T4RecConfig)


def test_column_schema_import():
    assert tr.ColumnSchema is not None


def test_dataset_schema_import():
    assert tr.DatasetSchema is not None


def test_tf_import():
    pytest.importorskip("tensorflow")

    assert tr.tf is not None
    assert tr.tf.Head is not None
    assert tr.tf.Model is not None


def test_torch_import():
    pytest.importorskip("torch")

    assert tr.torch is not None
    assert tr.torch.Head is not None
    assert tr.torch.Model is not None
