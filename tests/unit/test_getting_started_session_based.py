import os
from importlib.util import find_spec

import numpy as np
import pytest
from merlin.core.dispatch import HAS_GPU
from testbook import testbook

from tests.conftest import REPO_ROOT

pytest.importorskip("torch")
pytest.importorskip("transformers")

# flake8: noqa


@pytest.mark.notebook
@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
def test_func():
    with testbook(
        REPO_ROOT / "examples" / "getting-started-session-based" / "01-ETL-with-NVTabular.ipynb",
        execute=False,
    ) as tb1:
        tb1.inject(
            """
            import os
            os.environ["INPUT_DATA_DIR"] = "/tmp/data/"
            os.environ["NUM_ROWS"] = "10000"
            """
        )
        tb1.execute()
        assert os.path.isdir("/tmp/data/processed_nvt")
        assert os.path.isdir("/tmp/data/sessions_by_day")
        assert os.path.isdir("/tmp/data/workflow_etl")

    with testbook(
        REPO_ROOT
        / "examples"
        / "getting-started-session-based"
        / "02-session-based-XLNet-with-PyT.ipynb",
        execute=False,
    ) as tb2:
        tb2.inject(
            """
            import os
            os.environ["INPUT_DATA_DIR"] = "/tmp/data/"
            os.environ["INPUT_SCHEMA_PATH"] = "/tmp/data/processed_nvt/schema.pbtxt"
            os.environ["per_device_train_batch_size"] = "32"
            os.environ["per_device_eval_batch_size"] = "8"
            os.environ["final_window_index"] = "2"
            """
        )
        tb2.execute()
        eval_metrics = tb2.ref("eval_metrics")
        assert set(eval_metrics.keys()) == set(
            [
                "eval_/loss",
                "eval_/next-item/ndcg_at_20",
                "eval_/next-item/ndcg_at_40",
                "eval_/next-item/recall_at_20",
                "eval_/next-item/recall_at_40",
                "eval_runtime",
                "eval_samples_per_second",
                "eval_steps_per_second",
            ]
        )
        assert os.path.isdir("/tmp/data/saved_model")
