# Stable Diffusion
<center>
    <image src="../images/ddlm_2.png" width="256px">
    <p>prompt: A woman dressed like the Mexican Holiday Dia de los Muertos</p>
</center>

## Intro

This repo containerizes [stable diffusion](https://github.com/CompVis/stable-diffusion) into a serving container using [fastapi](https://fastapi.tiangolo.com/). 

**The model license can be found [here.](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)**

Features:
- Text to image.
- Image to image.
- Negative prompting.
- Word emphasis.

## Setup

1. Clone repo if you haven't. Navigate to the `stable-diffusion` folder.
1. Build container. Don't forget to change the `project_id` to yours.

    ```bash
    docker build . -t gcr.io/{project_id}/stable-diffusion:latest
    ```

1. Run container. You need [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) and a GPU.

    ```bash
    docker run -p 80:8080 --gpus all -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/{project_id}/stable-diffusion:latest -d
    ```

1. Make a prediction

    ```bash
    python generate_request.py
    curl -X POST -d @request.json -H "Content-Type: application/json; charset=utf-8" localhost/predict > response.json
    ```

## Deploy in Vertex AI.

You'll need to enable Vertex AI and have authenticated with a service account that has the Vertex AI admin or editor role.

1. Push the image

    ```bash
    gcloud auth configure-docker
    docker build . -t gcr.io/{project_id}/stable-diffusion:latest
    docker push gcr.io/{project_id}/stable-diffusion:latest
    ```
  
 1. Deploy in Vertex AI Endpoints.

    ```bash
    python ../gcp_deploy.py --image-uri gcr.io/<project_id>/stable-diffusion:latest --model-name stable-diffusion --endpoint-name stable-diffusion-endpoint --endpoint-deployed-name stable-diffusion-deployed-name
    ```

1. Test the endpoint. 

    ```bash
    python generate_request_vertex.py
    ```

