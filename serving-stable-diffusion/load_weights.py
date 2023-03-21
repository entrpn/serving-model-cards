import os
from diffusion_utils import get_pipeline

MODEL_NAME = os.getenv("MODEL_NAME",None)
print("MODEL_NAME:",MODEL_NAME)
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
print("MODEL_REVISION:", MODEL_REVISION)
USE_XFORMERS = os.getenv("USE_XFORMERS",False)
print("USE_XFORMERS",USE_XFORMERS)
pipe = get_pipeline(model_name=MODEL_NAME,use_cuda=False, revision=MODEL_REVISION, use_xformers=USE_XFORMERS)