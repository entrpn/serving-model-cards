## bart

## Intro

This repo containerizes [BART](https://huggingface.co/facebook/bart-large-cnn) into a serving container using [fastapi](https://fastapi.tiangolo.com/).

CPU and GPU inference supported.

**The model license can be found [here](https://huggingface.co/models?license=license:apache-2.0)**

## Setup

1. Clone repo if you haven't, Navigate to the `bart` folder.
1. Build container. Don't forget to change the `project_id` to yours.

    ```bash
    docker build . -t gcr.io/{project_id}/bart:latest
    ```

1. Run container. No GPU is needed for this model.

    ```bash
    docker run --rm -p 80:8080 -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/{project_id}/bart:latest
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
    docker build . -t gcr.io/{project_id}/bart:latest
    docker push gcr.io/{project_id}/bart:latest
    ```
  
 1. Deploy in Vertex AI Endpoints.

    ```bash
    python ../gcp_deploy.py --image-uri gcr.io/<project_id>/bart:latest --accelerator-count 0
    ```

1. Test the endpoint. 

    ```bash
    python generate_request_vertex.py
    ```
