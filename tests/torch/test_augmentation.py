import pytest

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


@pytest.mark.parametrize("replacement_prob", [0.1, 0.3, 0.5, 0.7])
def test_stochastic_swap_noise(yoochoose_schema, torch_yoochoose_like, replacement_prob):
    tab_module = torch4rec.EmbeddingFeatures.from_schema(yoochoose_schema)

    block = tab_module >> torch4rec.StochasticSwapNoise(
        pad_token=0, replacement_prob=replacement_prob
    )

    out = tab_module(torch_yoochoose_like)
    out_aug = block(torch_yoochoose_like)

    replacement_rate = (out["item_id/list"] == out_aug["item_id/list"]).double().mean()
    assert replacement_rate == pytest.approx(1 - replacement_prob, 0.1)

    assert not pytorch.equal(out["item_id/list"], out_aug["item_id/list"])
    assert not pytorch.equal(out["category/list"], out_aug["category/list"])
