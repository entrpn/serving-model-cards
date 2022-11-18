#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate ldm
pip install huggingface_hub
python download_weights.py --hf_token=$1