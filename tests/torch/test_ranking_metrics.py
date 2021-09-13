#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest

tr = pytest.importorskip("transformers4rec.torch")

# fixed parameters for tests
list_metrics = list(tr.ranking_metric.ranking_metrics_registry.keys())


# Test length of output equal to number of cutoffs
@pytest.mark.parametrize("metric", list_metrics)
def test_avg_precision_at(torch_ranking_metrics_inputs, metric):
    metric = tr.ranking_metric.ranking_metrics_registry[metric]
    metric.top_ks = torch_ranking_metrics_inputs["ks"]
    result = metric(
        torch_ranking_metrics_inputs["scores"], torch_ranking_metrics_inputs["labels_one_hot"]
    )
    assert len(result) == len(torch_ranking_metrics_inputs["ks"])


# Test label one hot encoding
@pytest.mark.parametrize("metric", list_metrics)
def test_score_with_transform_onehot(torch_ranking_metrics_inputs, metric):
    metric = tr.ranking_metric.ranking_metrics_registry[metric]
    metric.top_ks = torch_ranking_metrics_inputs["ks"]
    metric.labels_onehot = True
    result = metric(torch_ranking_metrics_inputs["scores"], torch_ranking_metrics_inputs["labels"])
    assert len(result) == len(torch_ranking_metrics_inputs["ks"])


# TODO: Compare the metrics @K between pytorch and numpy
@pytest.mark.parametrize("metric", list_metrics)
def test_numpy_comparison(metric):
    pass
