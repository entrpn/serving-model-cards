# Convert original stable diffusion checkpoint to diffusers

# Intro

This guide uses the `diffusers` library to convert an original stable diffusion pytorch checkpoint into a diffusers checkpoint.

# Setup

1. Install dependencies

    ```bash
    pip install omegaconf safetensors diffusers
    ```

1. Download the model checkpoint you would like to convert. In this example, we'll use a checkpoint from [civitai](https://civitai.com/). Choose any model that you're interested in and download it.

    ```bash
    wget https://civitai.com/api/download/models/6987 -O 6987.safetensors
    ```

1. We'll use [this](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py) script from the `diffusers` repo. You can copy it into this directory and then run it. Make sure you read the parameters involved based on your model (1.x or 2.x). The model in this example is a 1.5 model.

    ```bash
    python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path 6987.safetensors --dump_path realistic_vision_v1.3_checkpoint/ --scheduler_type pndm --from_safetensors
    ```

1. We can now load the weights using diffusers as follows:

    ```python
    from diffusers import DiffusionPipeline
    import PIL
    import requests
    from io import BytesIO
    import torch

    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    pipe = DiffusionPipeline.from_pretrained("realistic_vision_v1.3_checkpoint", custom_pipeline="stable_diffusion_mega", torch_dtype=torch.float32)
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    ## Text-to-Image

    images = pipe.text2img("RAW photo, a close up shot of a woman wearing a dark blue denim skirt with a green patch").images
    images[0].save("txt2img.png")
```