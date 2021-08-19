import pytest
from transformers import PreTrainedModel, TFPreTrainedModel

from transformers4rec.config import transformer as tconf

config_classes = [
    tconf.ReformerConfig,
    tconf.XLNetConfig,
    tconf.ElectraConfig,
    tconf.LongformerConfig,
    tconf.GPT2Config,
]


@pytest.mark.parametrize("config_cls", config_classes)
def test_to_torch_model(config_cls):
    config = config_cls.build(100, 4, 2, 20)

    model = config.to_torch_model()

    assert isinstance(model, PreTrainedModel)


@pytest.mark.parametrize("config_cls", list(set(config_classes) - {tconf.ReformerConfig}))
def test_to_tf_model(config_cls):
    config = config_cls.build(100, 4, 2, 20)

    model = config.to_tf_model()

    assert isinstance(model, TFPreTrainedModel)
