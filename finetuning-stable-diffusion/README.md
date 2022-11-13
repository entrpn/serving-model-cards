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

    ```bash
    python gcp_run_train.py --project-id={project-id} --region=us-central1 \
    --image-uri=gcr.io/{project-id}/finetuning-sd:latest \
    --gcs-output-dir=gs://{project-id}-bucket/sd-finetuned-model \
    --hf-token="some hf token" --max-train-steps=15000 \
    --batch-size=4 --learning-rate=1e-5
    ```

1. Once your job is finished, the model will be uploaded to `gcs-output-dir`. You can use it in a GCP TPU-VM or Colab. Take a look at the [`infer_jax.py`](../training-dreambooth/infer_jax.py) script for an example of how to create images.