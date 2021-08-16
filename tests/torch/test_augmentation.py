import pytest

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


def test_stochastic_swap_noise(yoochoose_schema, torch_yoochoose_like):
    tab_module = torch4rec.EmbeddingFeatures.from_schema(yoochoose_schema)

    block = tab_module >> torch4rec.StochasticSwapNoise(pad_token=0, replacement_prob=0.5)

    out = tab_module(torch_yoochoose_like)
    out_aug = block(torch_yoochoose_like)

    assert not pytorch.equal(out["item_id/list"], out_aug["item_id/list"])
    assert not pytorch.equal(out["category/list"], out_aug["category/list"])
