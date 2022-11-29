import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
from jax import pmap

from flax import serialization

from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from diffusers import FlaxStableDiffusionPipeline

dtype = jnp.bfloat16

# Your model should be located in this folder
model_name='sd-sks-model'
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    model_name,
    revision="bf16",
    dtype=dtype)

prompt = 'a photo of sks man wearing an iron man suit'
prompt = [prompt] * jax.device_count()

prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids.shape

p_params = replicate(params)

prompt_ids = shard(prompt_ids)
prompt_ids.shape

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())

images = pipeline(prompt_ids, p_params, rng, jit=True)[0]

images = images.reshape((images.shape[0],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)

def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    grid.save('output.png')

image_grid(images, 2, 4)
