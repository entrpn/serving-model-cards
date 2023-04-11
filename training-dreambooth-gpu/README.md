# Training Dreambooth With GPUs

## Intro

This repo prepares a Vertex AI training job for [dreambooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) to run on GPUs.

**The model license can be found [here.](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)**

Features:
- Dreambooth is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
- LoRA (optional)
- Tunes the tokenizer.

## Setup

1. Clone repo if you haven't. Navigate to the `training-dreambooth-gpu` folder.
1. Create a folder `images` with your subject images.
1. Build and push the image. Don't forget to change the `project_id` to yours.

    ```bash
    PROJECT_ID=<project_id>
    gcloud auth configure-docker
    docker docker build -t gcr.io/$PROJECT_ID/training-dreambooth-gpus:latest --build-arg use_lora=1 --build-arg use_xformers=1 .
    docker push gcr.io/$PROJECT_ID/training-dreambooth-gpus:latest
    ```

1. Deploy the training job. Don't forget to change the `bucket-name` to yours.

    ```bash
    BUCKET_NAME=<bucket-name>
    python gcp_run_train.py --project-id=$PROJECT_ID --region=us-central1 \
        --image-uri=gcr.io/$PROJECT_ID/training-dreambooth-gpus:latest \
        --model-name=runwayml/stable-diffusion-v1-5 \
        --model-revision=fp16 \
        --gcs-output-dir=gs://$BUCKET_NAME \
        --instance-prompt="A photo of sks man" \
        --class-prompt="A photo of a man" \
        --max-train-steps=800
    ```