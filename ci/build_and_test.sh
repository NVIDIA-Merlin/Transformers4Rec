#!/bin/bash
set -e

# get the nvtabular directory
ci_directory="$(dirname -- "$(readlink -f -- "$0")")"
nvt_directory="$(dirname -- $ci_directory)"
cd $nvt_directory

echo "Installing models"
pip install --user -e .[tensorflow,pytorch,nvtabular]

# following checks requirement requirements-dev.txt to be installed
echo "Running black --check"
black --check .
echo "Running flake8"
flake8 .
echo "Running isort"
isort -c .
echo "Running bandit"
bandit -q -ll --recursive merlin_models
# echo "Running pylint"
# pylint tests
echo "Running flake8-nb"
flake8-nb examples

# test out our codebase
py.test --cov-config tests/unit/.coveragerc --cov-report term-missing --cov-report xml --cov-fail-under 70 --cov=. tests/tensorflow
