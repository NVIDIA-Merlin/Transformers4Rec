pipeline {
    agent {
        docker {
            image 'nvcr.io/nvstaging/merlin/merlin-ci-runner-wrapper'
            label 'merlin_gpu'
            registryCredentialsId 'jawe-nvcr-io'
            registryUrl 'https://nvcr.io'
            args "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"
        }
    }

    options {
      buildDiscarder(logRotator(numToKeepStr: '10'))
      ansiColor('xterm')
      disableConcurrentBuilds(abortPrevious: true)
    }

    stages {
        stage("test-gpu") {
            options {
                timeout(time: 60, unit: 'MINUTES', activity: true)
            }
            steps {
                sh """#!/bin/bash
set -e
printenv
rm -rf $HOME/.cudf/

export TF_MEMORY_ALLOCATION="0.1"
export CUDA_VISIBLE_DEVICES=2,3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
export MKL_SERVICE_FORCE_INTEL=1
export NR_USER=true

python -m pip install --upgrade git+https://github.com/NVIDIA-Merlin/NVTabular.git
python -m pytest -rsx tests/unit
                """
            }
        }
    }
}