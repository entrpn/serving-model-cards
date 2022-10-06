#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate ldm
export PORT=$AIP_HTTP_PORT
echo $PORT
# For debugging
#export CUDA_LAUNCH_BLOCKING=1
uvicorn main:app --proxy-headers --host 0.0.0.0 --port $PORT