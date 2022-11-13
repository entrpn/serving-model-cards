# Finetuning Stable Diffusion

## Intro

This repo prepares a Vertex AI training job for [finetuning stable diffusion]() on TPUs.

**The model license can be found [here.](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)**

Features:
- Finetune stable diffusion to have a style based on the training dataset.

## Setup

1. Clone repo if you haven't. Navigate to the `finetuning-stable-diffusion` folder.
1. Create a folder `dataset` with a metadata.jsonl and images as described [here](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata)
1. Build and push the image. Don't forget to change `project_id` to yours.

    ```bash
    gcloud auth configure-docker
    docker build . -t gcr.io/{project_id}/finetuning-sd:latest
    docker push gcr.io/{project_id}/finetuning-sd:latest
    ```

1. Deploy the training job.