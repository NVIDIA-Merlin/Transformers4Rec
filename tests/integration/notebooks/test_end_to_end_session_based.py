import os

import pytest
from merlin.core.dispatch import HAS_GPU
from testbook import testbook

from tests.conftest import REPO_ROOT

pytest.importorskip("transformers")

# flake8: noqa


@pytest.mark.notebook
@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
def test_func():
    with testbook(
        REPO_ROOT / "examples" / "end-to-end-session-based" / "01-ETL-with-NVTabular.ipynb",
        execute=False,
    ) as tb1:
        tb1.inject(
            """
            import os
            os.environ["DATA_FOLDER"] = "/tmp/data/"
            os.environ["USE_SYNTHETIC"] = "True"
            os.environ["START_DATE"] = "2014/4/1"
            os.environ["END_DATE"] = "2014/4/5"
            os.environ["THRESHOLD_DAY_INDEX"] = "1"
            """
        )
        tb1.execute()
        assert os.path.isdir("/tmp/data/processed_nvt")
        assert os.path.isdir("/tmp/data/preproc_sessions_by_day")
        assert os.path.isdir("/tmp/data/workflow_etl")

    with testbook(
        REPO_ROOT
        / "examples"
        / "end-to-end-session-based"
        / "02-End-to-end-session-based-with-Yoochoose-PyT.ipynb",
        timeout=720,
        execute=False,
    ) as tb2:
        tb2.inject(
            """
            import os
            os.environ["INPUT_DATA_DIR"] = "/tmp/data/"
            os.environ["OUTPUT_DIR"] = "/tmp/data/preproc_sessions_by_day"
            os.environ["START_TIME_INDEX"] = "1"
            os.environ["END_TIME_INDEX"] = "3"
            os.environ["BATCH_SIZE_TRAIN"] = "64"
            os.environ["BATCH_SIZE_VALID"] = "32"
            """
        )
        NUM_OF_CELLS = len(tb2.cells)
        tb2.execute_cell(list(range(0, NUM_OF_CELLS - 20)))
        topk = tb2.ref("topk")
        tb2.inject(
            """
            import pandas as pd
            interactions_merged_df = pd.read_parquet(os.path.join(INPUT_DATA_DIR, "interactions_merged_df.parquet"))
            interactions_merged_df = interactions_merged_df.sort_values('timestamp')
            batch = interactions_merged_df[-500:]
            sessions_to_use = batch.session_id.value_counts()
            filtered_batch = batch[batch.session_id.isin(sessions_to_use[sessions_to_use.values>1].index.values)]
            
            from merlin.systems.triton.utils import run_ensemble_on_tritonserver
            response = run_ensemble_on_tritonserver(
                "/tmp/data/models", workflow.input_schema, filtered_batch, model.output_schema.column_names,  'executor_model'
            )
            response_array = list(response['item_ids'][1])
            """
        )
        tb2.execute_cell(NUM_OF_CELLS - 8)
        response_array = tb2.ref("response_array")
        assert len(response_array) == topk

    with testbook(
        REPO_ROOT
        / "examples"
        / "end-to-end-session-based"
        / "03-Session-based-Yoochoose-multigpu-training-PyT.ipynb",
        execute=False,
    ) as tb3:
        tb3.inject(
            """
            import os
            os.environ["INPUT_DATA_DIR"] = "/tmp/data/"
            os.environ["OUTPUT_DIR"] = "/tmp/data/preproc_sessions_by_day"
            os.environ["START_TIME_INDEX"] = "1"
            os.environ["END_TIME_INDEX"] = "4"
            os.environ["LEARNING_RATE"] = "0.0005"
            os.environ["BATCH_SIZE_TRAIN"] = "64"
            os.environ["BATCH_SIZE_VALID"] = "32"
            """
        )
        tb3.execute()
