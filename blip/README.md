# BLIP
<center>
    <image src="../images/blip.jpeg" width="256px">
</center>

## Intro

This repo containerizes [BLIP](https://github.com/CompVis/stable-diffusion) into a serving container using [fastapi](https://fastapi.tiangolo.com/). 

**The model license can be found [here.](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt)**

Features:
- Image captioning
- Open-ended visual question answering
- Multimodal / unimodal feature extraction
- Image-text matching

## Setup

1. Clone repo if you haven't. Navigate to the `blip` folder.
1. Build container. Don't forget to change the `project_id` to yours.

    ```bash
    docker build . -t gcr.io/{project_id}/blip:latest
    ```

1. Run container. You need [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) and a GPU.

    ```bash
    docker run -p 80:8080 --gpus all -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/{project_id}/blip:latest -d
    ```