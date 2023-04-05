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
            """
        )
        tb1.execute()
        assert os.path.isdir("/tmp/data/processed_nvt")
        assert os.path.isdir("/tmp/data/preproc_sessions_by_day")
        assert os.path.isdir("/tmp/data/workflow_etl")

    # with testbook(
    #     REPO_ROOT
    #     / "examples"
    #     / "end-to-end-session-based"
    #     / "03-Session-based-Yoochoose-multigpu-training-PyT.ipynb",
    #     execute=False,
    # ) as tb3:
    #     tb3.inject(
    #         """
    #         import os
    #         os.environ["INPUT_DATA_DIR"] = "/tmp/data/"
    #         os.environ["OUTPUT_DIR"] = "/tmp/data/sessions_by_day"
    #         os.environ["model_path"] = "/tmp/data/saved_model"
    #         """
    #     )
    #     NUM_OF_CELLS = len(tb3.cells)
    #     tb3.execute_cell(list(range(0, NUM_OF_CELLS - 12)))
    #     tb3.inject(
    #         """
    #         eval_batch_size = 4
    #         eval_paths = os.path.join('/tmp/data/sessions_by_day', f"{1}/valid.parquet")
    #         eval_dataset = Dataset(eval_paths, shuffle=False)
    #         eval_loader = generate_dataloader(schema, eval_dataset, batch_size=eval_batch_size)
    #         test_dict = next(iter(eval_loader))
    #         df_cols = {}
    #         for name, tensor in test_dict[0].items():
    #             if name in input_schema.column_names:
    #                 df_cols[name] = tensor.cpu().numpy()
    #                 if len(tensor.shape) > 1:
    #                     df_cols[name] = list(df_cols[name])
    #         df = make_df(df_cols)
    #         from merlin.systems.triton.utils import run_ensemble_on_tritonserver
    #         response = run_ensemble_on_tritonserver(
    #             "/tmp/data/models", ensemble.graph.input_schema, df[input_schema.column_names], output_schema.column_names,  'executor_model'
    #         )
    #         response_array = [x.tolist()[0] for x in response["next-item"]]
    #         """
    #     )
    #     tb3.execute_cell(NUM_OF_CELLS - 3)
    #     batch_size = tb3.ref("eval_batch_size")
    #     response_array = tb3.ref("response_array")
    #     assert len(response_array) == batch_size
