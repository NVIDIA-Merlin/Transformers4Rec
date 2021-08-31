import pytest

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


@pytest.mark.parametrize("replacement_prob", [0.1, 0.3, 0.5, 0.7])
def test_stochastic_swap_noise(replacement_prob):
    NUM_SEQS = 100
    SEQ_LENGTH = 80
    PAD_TOKEN = 0

    # Creating some input sequences with padding in the end
    # (to emulate sessions with different lengths)
    seq_inputs = {
        "categ_feat": pytorch.tril(
            pytorch.randint(low=1, high=100, size=(NUM_SEQS, SEQ_LENGTH)), 1
        ),
        "cont_feat": pytorch.tril(pytorch.rand((NUM_SEQS, SEQ_LENGTH)), 1),
    }

    ssn = torch4rec.StochasticSwapNoise(pad_token=PAD_TOKEN, replacement_prob=replacement_prob)
    out_features_ssn = ssn(seq_inputs)

    for fname in seq_inputs:
        replaced_mask = out_features_ssn[fname] != seq_inputs[fname]
        replaced_mask_non_padded = pytorch.masked_select(
            replaced_mask, seq_inputs[fname] != PAD_TOKEN
        )
        replacement_rate = replaced_mask_non_padded.float().mean()
        assert replacement_rate == pytest.approx(replacement_prob, abs=0.1)


@pytest.mark.parametrize("replacement_prob", [0.1, 0.3, 0.5, 0.7])
def test_stochastic_swap_noise_with_tabular_features(
    yoochoose_schema, torch_yoochoose_like, replacement_prob
):
    inputs = torch_yoochoose_like
    tab_module = torch4rec.TabularSequenceFeatures.from_schema(yoochoose_schema)
    out_features = tab_module(inputs)

    PAD_TOKEN = 0
    ssn = torch4rec.StochasticSwapNoise(pad_token=PAD_TOKEN, replacement_prob=replacement_prob)
    out_features_ssn = ssn(out_features)

    for fname in out_features:
        replaced_mask = out_features_ssn[fname] != out_features[fname]

        # Ignoring padding items to compute the mean replacement rate
        feat_non_padding_mask = inputs[fname] != PAD_TOKEN
        # For embedding features it is necessary to expand the mask
        if len(replaced_mask.shape) > len(feat_non_padding_mask.shape):
            feat_non_padding_mask = feat_non_padding_mask.unsqueeze(-1)
        replaced_mask_non_padded = pytorch.masked_select(replaced_mask, feat_non_padding_mask)
        replacement_rate = replaced_mask_non_padded.float().mean()
        assert replacement_rate == pytest.approx(replacement_prob, abs=0.1)
