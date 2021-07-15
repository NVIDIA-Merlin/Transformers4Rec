import pytest

torch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


def test_filter_features(torch_con_features):
    features = ["a", "b"]
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
    assert tabular(torch_con_features, concat_outputs=True, filter_columns=["b"]).size()[1] == 1

    tabular_concat = _DummyTabular(aggregation="concat")
    assert tabular_concat(torch_con_features).size()[1] == 6

    tab_a = ["a"] >> _DummyTabular()
    tab_b = ["b"] >> _DummyTabular()

    assert tab_a(torch_con_features, merge_with=tab_b, stack_outputs=True).size()[1] == 1
    assert (tab_a + tab_b)(torch_con_features, concat_outputs=True).size()[1] == 2
