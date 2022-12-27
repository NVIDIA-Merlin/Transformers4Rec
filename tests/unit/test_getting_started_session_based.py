import os
from importlib.util import find_spec

import numpy as np
import pytest
from merlin.systems.triton.utils import run_ensemble_on_tritonserver
from testbook import testbook

from tests.conftest import REPO_ROOT

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.mark.skipif(find_spec("cudf") is None, reason="needs cudf")

# flake8: noqa


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
            os.environ["INPUT_SCHEMA_PATH"] = "/tmp/data/processed_nvt/schema.pbtxt"
            os.environ["model_path"] = "/tmp/data/saved_model"
            """
        )
        NUM_OF_CELLS = len(tb3.cells)
        tb3.execute_cell(list(range(0, NUM_OF_CELLS - 10)))
        tb3.inject(
            """
            eval_batch_size = 4
            eval_paths = os.path.join('/tmp/data/sessions_by_day', f"{1}/valid.parquet")
            eval_dataset = Dataset(eval_paths, shuffle=False)
            eval_loader = generate_dataloader(schema, eval_dataset, batch_size=eval_batch_size)
            test_dict = next(iter(eval_loader))

            df_cols = {}
            for name, tensor in test_dict[0].items():
                if name in input_schema.column_names:
                    dtype = input_schema[name].dtype
                    df_cols[name] = tensor.cpu().numpy().astype(dtype)
                    if len(tensor.shape) > 1:
                        df_cols[name] = list(df_cols[name])

            df = make_df(df_cols)
            
            
            from merlin.systems.triton.utils import run_ensemble_on_tritonserver
            response = run_ensemble_on_tritonserver(
                "/tmp/data/models", ensemble.graph.input_schema, df[input_schema.column_names], output_schema.column_names,  'executor_model'
            )
            response_array = [x.tolist()[0] for x in response["next-item"]]
            """
        )
        tb3.execute_cell(NUM_OF_CELLS - 3)
        batch_size = tb3.ref("eval_batch_size")
        response_array = tb3.ref("response_array")
        assert len(response_array) == batch_size
