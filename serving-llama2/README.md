# Llama2

## Intro

This repo containerizes llama2 using Huggingface's transformers library into a serving container using [fastapi](https://fastapi.tiangolo.com/) which can be served with Vertex AI prediction.

Features:
- Load any llama model via the model_name arg.
- Four bit quantization using bitsandbytes.

## Setup

1. Clone repo if you haven't. Navigate to the `serving-llama2` folder.

1. Build container. Change the `project-id` to yours. This requires a machine with at least 48GB of RAM.

    ```bash
    PROJECT_ID=<project-id>
    HF_AUTH=<your hf token>
    sudo docker build -t gcr.io/$PROJECT_ID/serving-llama2:latest --build-arg model_name=meta-llama/Llama-2-13b-chat-hf --build-arg hf_auth=$HF_AUTH --build-arg max_new_tokens=128 .
    ```

1. Run container. You need [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) and a GPU.

    ```bash
    sudo nvidia-ctk runtime configure
    docker run -p 80:8080 --gpus all -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/$PROJECT_ID/serving-llama2:latest
    ```

1. Test the prediction locally. You will need a local GPU for this.

    ```bash
    python test_container.py
    ```

## Deploy in Vertex AI.

You'll need to enable Vertex AI and have authenticated with a service account that has the Vertex AI admin or editor role.

1. Push the image

    ```bash
    gcloud auth configure-docker
    docker push gcr.io/$PROJECT_ID/serving-llama2:latest
    ```

1. Deploy in Vertex AI prediction.

    ```bash
    python ../gcp_deploy --image-uri gcr.io/PROJECT_ID/serving-llama2:latest --model-name llama2 --endpoint-name llama2-endpoint --endpoint-deployed-name llama2-deployed-name --machine-type n1-standard-8 --gpu-type NVIDIA_TESLA_V100
    ```

1. The last command will display the endpoint name, it should look like `projects/123456454345/localtions/us-central1/endpoints/34559494848493`

    Test the endpoint using the endpoint name.

    ```bash
    python generate_request_vertex.py --endpoint-name projects/123456454345/localtions/us-central1/endpoints/34559494848493
    ```