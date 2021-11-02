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

import importlib
import itertools
import json
import os
import subprocess
import sys
from os.path import dirname, realpath

import pytest

TEST_PATH = dirname(dirname(realpath(__file__)))
SESSION_PATH = "examples/getting-started-session-based/"


@pytest.mark.skipif(importlib.util.find_spec("cudf") is None, reason="needs cudf")
def test_session(tmpdir):
    BASE_PATH = os.path.join(dirname(TEST_PATH), SESSION_PATH)
    # Run ETL
    nb_path = os.path.join(BASE_PATH, "01-ETL-with-NVTabular.ipynb")
    _run_notebook(tmpdir, nb_path)

    # Run session based
    torch = importlib.util.find_spec("torch")
    if torch is not None:
        os.environ["INPUT_SCHEMA_PATH"] = BASE_PATH + "schema.pb"
        nb_path = os.path.join(BASE_PATH, "02-session-based-XLNet-with-PyT.ipynb")
        _run_notebook(tmpdir, nb_path)
    else:
        print("Torch needed for this notebook")


def _run_notebook(tmpdir, notebook_path, transform=None):
    # read in the notebook as JSON, and extract a python script from it
    notebook = json.load(open(notebook_path))
    source_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    lines = [
        transform(line.rstrip()) if transform else line
        for line in itertools.chain(*source_cells)
        if not (line.startswith("%") or line.startswith("!"))
    ]

    # save the script to a file, and run with the current python executable
    # we're doing this in a subprocess to avoid some issues using 'exec'
    # that were causing a segfault with globals of the exec'ed function going
    # out of scope
    script_path = os.path.join(tmpdir, "notebook.py")
    with open(script_path, "w") as script:
        script.write("\n".join(lines))
    subprocess.check_output([sys.executable, script_path])
