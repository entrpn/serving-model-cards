# Flan-T5

## Intro

This repo containerizes [flan-t5](https://huggingface.co/google/flan-t5-large) into a serving container using [fastapi](https://fastapi.tiangolo.com/). 

Since this uses Huggingface's [transformers](https://github.com/huggingface/transformers) library, specifically the `AutoModeForSeq2SeqLM` class, it should be able to load other models supported by the library.

**The model license can be found [here.](https://github.com/google-research/t5x)**

Features:
- Text generation.
- Language translation.
- Sentiment analysis.
- text classification.

## Setup

1. Clone repo if you haven't. Navigate to the `serving-flant5` folder.
1. Build container. Don't forget to change the `project_id` to yours.

    ```bash
    docker build --build-arg model_name=google/flan-t5-large . -t gcr.io/{project_id}/serving-t5:latest
    ```

1. Run container. You need [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) and a GPU.

    ```bash
    docker run -p 80:8080 --gpus all -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/{project_id}/serving-t5:latest -d
    ```

1. Make predictions

    ```bash
    python test_container.py
    ```

## Deploy in Vertex AI

You'll need to enable Vertex AI and have authenticated with a service account that has the Vertex AI admin or editor role.

1. Push the image

    ```bash
    gcloud auth configure-docker
    docker push gcr.io/{project_id}/serving-t5:latest
    ```

1. Deploy in Vertex AI Endpoints

    ```bash
    python ../gcp_deploy.py --image-uri gcr.io/<project_id>/serving-t5:latest --machine-type n1-standard-8 --model-name flant5 --endpoint-name flant5-endpoint --endpoint-deployed-name flant5-deployed-name

1. Test the endpoint

    ```bash
    python generate_request_vertex.py
    ```