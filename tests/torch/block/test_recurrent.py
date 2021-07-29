# import pytest
#
# from transformers4rec.config.transformer import XLNetConfig
#
# torch4rec = pytest.importorskip("transformers4rec.torch")
#
#
# def test_recurrent_block(yoochoose_column_group, torch_yoochoose_like):
#     col_group = yoochoose_column_group
#     tab_module = torch4rec.SequentialTabularFeatures.from_column_group(
#         col_group, max_sequence_length=20, aggregation="sequential_concat"
#     )
#
#     block = tab_module >> torch4rec.MLPBlock([64])
#
#     transformer_config = XLNetConfig.for_rec(64, 4, 2)
#     transformer = transformer_config.to_torch_model()
#
#     outputs = tab_module(torch_yoochoose_like)
#
#     assert len(outputs.keys()) == 3
#     assert all(tensor.shape[-1] == 64 for tensor in outputs.values())
#     assert all(tensor.shape[1] == 20 for tensor in outputs.values())
