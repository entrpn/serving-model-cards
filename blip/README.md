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

1. Run container. No GPU is needed for this model.

    ```bash
    docker run --rm -p 80:8080 -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/{project_id}/blip:latest
    ```

1. Make predictions

    ```bash
    python test_container.py
    ```

## Deploy in Vertex AI.

You'll need to enable Vertex AI and have authenticated with a service account that has the Vertex AI admin or editor role.

1. Push the image

    ```bash
    gcloud auth configure-docker
    docker build . -t gcr.io/{project_id}/blip:latest
    docker push gcr.io/{project_id}/blip:latest
    ```
  
 1. Deploy in Vertex AI Endpoints.

    ```bash
    python ../gcp_deploy.py --image-uri gcr.io/<project_id>/blip:latest --accelerator-count 0 --model-name blip --endpoint-name blip-endpoint --endpoint-deployed-name blip-deployed-name
    ```

1. Test the endpoint. 

    ```bash
    python generate_request_vertex.py
    ```
