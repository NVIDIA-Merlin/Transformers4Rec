#!/bin/bash
set -e

# Get latest Transformers4Rec version
cd /transformers4rec/
git pull origin main

container=$1

## Tensorflow container
if [ "$container" == "merlin-tensorflow-training" ]; then
    pytest tests/tf
# Pytorch container
elif [ "$container" == "merlin-pytorch-training" ]; then
    pytest tests/torch
# Inference container
elif [ "$container" == "merlin-inference" ]; then
    pytest tests
fi
