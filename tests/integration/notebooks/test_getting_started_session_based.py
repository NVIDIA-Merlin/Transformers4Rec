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

    with testbook(
        REPO_ROOT
        / "examples"
        / "getting-started-session-based"
        / "03-serving-session-based-model-torch-backend.ipynb",
        execute=False,
    ) as tb3:
        tb3.inject(
            """
            import os
            os.environ["INPUT_DATA_DIR"] = "/tmp/data/"
            os.environ["OUTPUT_DIR"] = "/tmp/data/sessions_by_day"
            os.environ["model_path"] = "/tmp/data/saved_model"
            """
        )
        NUM_OF_CELLS = len(tb3.cells)
        tb3.execute_cell(list(range(0, NUM_OF_CELLS - 12)))
        tb3.inject(
            """
            NUM_ROWS =1000
            long_tailed_item_distribution = np.clip(np.random.lognormal(3., 1., int(NUM_ROWS)).astype(np.int32), 1, 50000)
            df = pd.DataFrame(np.random.randint(70000, 90000, int(NUM_ROWS)), columns=['session_id'])
            df['item_id'] = long_tailed_item_distribution
            df['category'] = pd.cut(df['item_id'], bins=334, labels=np.arange(1, 335)).astype(np.int32)
            df['age_days'] = np.random.uniform(0, 1, int(NUM_ROWS)).astype(np.float32)
            df['weekday_sin']= np.random.uniform(0, 1, int(NUM_ROWS)).astype(np.float32)
            map_day = dict(zip(df.session_id.unique(), np.random.randint(1, 10, size=(df.session_id.nunique()))))
            df['day'] =  df.session_id.map(map_day)

            from merlin.systems.triton.utils import run_ensemble_on_tritonserver
            response = run_ensemble_on_tritonserver(
                "/tmp/data/models", workflow.input_schema, df, output_schema.column_names,  'executor_model'
            )
            response_array = list(response['next-item'][1])
            cardinality = workflow.output_schema['item_id-list'].properties['embedding_sizes']['cardinality']
            """
        )
        tb3.execute_cell(NUM_OF_CELLS - 3)
        item_cardinality = tb3.ref("cardinality")
        response_array = tb3.ref("response_array")
        assert len(response_array) == item_cardinality
