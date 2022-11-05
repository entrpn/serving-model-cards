# Training Dreambooth

## Intro

This repo prepares a Vertex AI training job for [dreambooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) to run on TPUs.

**The model license can be found [here.](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)**

Features:
- Dreambooth is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
- Tunes the tokenizer.

## Setup

1. Clone repo if you haven't. Navigate to the `training-dreambooth` folder.
1. Create a folder `images` with your subject images.
1. Build and push the image. Don't forget to change the `project_id` to yours.

    ```bash
    gcloud auth configure-docker
    docker build . -t gcr.io/{project_id}/training-dreambooth:latest
    docker push gcr.io/{project_id}/training-dreambooth:latest
    ```

1. Deploy the training job.

    ```bash
    python gcp_run_train.py --project-id={project-id} --region=us-central1 \
        --image-uri=gcr.io/{project_id}/training-dreambooth:latest \
        --gcs-output-dir=gs://my-bucket-name \
        --instance-prompt="A photo of sks man" \
        --hf-token="some hf token" \
        --class-prompt="A photo of sks dog" \
        --max-train-steps=800
    ```