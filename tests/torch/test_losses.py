import pytest

pytorch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers4rec.torch")
examples_utils = pytest.importorskip("transformers4rec.torch.losses")


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.6])
def test_item_prediction_with_label_smoothing_ce_loss(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, label_smoothing
):
    np = pytest.importorskip("numpy")
    custom_loss = examples_utils.LabelSmoothCrossEntropyLoss(
        reduction="mean", smoothing=label_smoothing
    )
    input_module = torch_yoochoose_tabular_transformer_features
    body = tr.SequentialBlock(input_module, tr.MLPBlock([64]))
    head = tr.Head(
        body, tr.NextItemPredictionTask(weight_tying=True, loss=custom_loss), inputs=input_module
    )

    body_outputs = body(torch_yoochoose_like)

    trg_flat = input_module.masking.masked_targets.flatten()
    non_pad_mask = trg_flat != input_module.masking.padding_idx
    labels_all = pytorch.masked_select(trg_flat, non_pad_mask)
    predictions = head(body_outputs)

    loss = head.prediction_task_dict["next-item"].compute_loss(
        inputs=body_outputs,
        targets=labels_all,
    )

    n_classes = 51997
    manuall_loss = pytorch.nn.NLLLoss(reduction="mean")
    target_with_smoothing = labels_all * (1 - label_smoothing) + label_smoothing / n_classes
    manual_output_loss = manuall_loss(predictions, target_with_smoothing.to(pytorch.long))

    assert np.allclose(manual_output_loss.detach().numpy(), loss.detach().numpy(), rtol=1e-3)
