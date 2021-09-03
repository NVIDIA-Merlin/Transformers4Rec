import pytest

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_filter_features(tf_con_features):
    features = ["a", "b"]
    con = tf4rec.FilterFeatures(features)(tf_con_features)

    assert list(con.keys()) == features


def test_as_tabular(tf_con_features):
    name = "tabular"
    con = tf4rec.AsTabular(name)(tf_con_features)

    assert list(con.keys()) == [name]


def test_tabular_module(tf_con_features):
    _DummyTabular = tf4rec.TabularBlock

    tabular = _DummyTabular()

    assert tabular(tf_con_features) == tf_con_features
    assert tabular(tf_con_features, aggregation="concat").shape[1] == 6
    assert tabular(tf_con_features, aggregation=tf4rec.ConcatFeatures()).shape[1] == 6

    tabular_concat = _DummyTabular(aggregation="concat")
    assert tabular_concat(tf_con_features).shape[1] == 6

    tab_a = ["a"] >> _DummyTabular()
    tab_b = ["b"] >> _DummyTabular()

    assert tab_a(tf_con_features, merge_with=tab_b, aggregation="stack").shape[1] == 1
