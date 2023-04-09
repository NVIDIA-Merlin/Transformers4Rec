import numpy as np
import pytest
import torch

import transformers4rec.torch as tr


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.6])
def test_item_prediction_with_label_smoothing_ce_loss(
    torch_yoochoose_tabular_transformer_features, torch_yoochoose_like, label_smoothing
):
    custom_loss = tr.LabelSmoothCrossEntropyLoss(reduction="mean", smoothing=label_smoothing)
    input_module = torch_yoochoose_tabular_transformer_features
    body = tr.SequentialBlock(input_module, tr.MLPBlock([64]))
    head = tr.Head(
        body, tr.NextItemPredictionTask(weight_tying=True, loss=custom_loss), inputs=input_module
    )

    body_outputs = body(torch_yoochoose_like, training=True)

    trg_flat = input_module.masking.masked_targets.flatten()
    non_pad_mask = trg_flat != input_module.masking.padding_idx
    labels_all = torch.masked_select(trg_flat, non_pad_mask)
    head_output = head(body_outputs, training=True)

    loss = head_output["loss"]

    head_predictions = head_output["predictions"]["next-item"]
    manuall_loss = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)
    manual_output_loss = manuall_loss(head_predictions, labels_all)

    assert np.allclose(manual_output_loss.detach().numpy(), loss.detach().numpy(), rtol=1e-3)
