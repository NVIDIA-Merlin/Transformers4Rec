import pytest
import torch

torch4rec = pytest.importorskip("transformers4rec.torch")

if torch.cuda.is_available():
    devices = ["cpu", "cuda"]
else:
    devices = ["cpu"]


def test_filter_features(torch_con_features):
    features = ["con_a", "con_b"]
    con = torch4rec.FilterFeatures(features)(torch_con_features)

    assert list(con.keys()) == features


def test_as_tabular(torch_con_features):
    name = "tabular"
    con = torch4rec.AsTabular(name)(torch_con_features)

    assert list(con.keys()) == [name]


def test_tabular_module(torch_con_features):
    class _DummyTabular(torch4rec.TabularModule):
        def forward(self, inputs):
            return inputs

    tabular = _DummyTabular()

    assert tabular(torch_con_features) == torch_con_features
    assert tabular(torch_con_features, aggregation="concat").size()[1] == 6
    assert tabular(torch_con_features, aggregation=torch4rec.ConcatFeatures()).size()[1] == 6
    assert tabular(torch_con_features, concat_outputs=True, filter_columns=["con_b"]).size()[1] == 1

    tabular_concat = _DummyTabular(aggregation="concat")
    assert tabular_concat(torch_con_features).size()[1] == 6

    tab_a = ["con_a"] >> _DummyTabular()
    tab_b = ["con_b"] >> _DummyTabular()

    assert tab_a(torch_con_features, merge_with=tab_b, stack_outputs=True).size()[1] == 1
    assert (tab_a + tab_b)(torch_con_features, concat_outputs=True).size()[1] == 2


@pytest.mark.parametrize("device", devices)
def test_tabular_module_to_device(yoochoose_schema, device):
    schema = yoochoose_schema
    tab_module = torch4rec.SequentialTabularFeatures.from_schema(
        schema, max_sequence_length=20, aggregation="sequential_concat"
    )
    tab_module.to(device)

    # Flatten nested torch modules
    def flatten(el):
        flattened = [flatten(children) for children in el.children()]
        res = [el]
        for c in flattened:
            res += c
        return res

    flatten_layers = flatten(tab_module)

    # Check params of pytorch modules are moved to appropriate device
    assert all(
        [
            list(el.parameters())[-1].device.type == device
            for el in flatten_layers
            if list(el.parameters())
        ]
    )
