# Stable Diffusion Batch Job

## Intro

Create a long running batch inference job to generate multiple images with multiple finetunned stable diffusion models from Huggingface.

## Setup

1. Clone repo if you haven't. Navigate to the `stable-diffusion-batch-job` folder.
1. Create a `jsonl` formatted file and name it `metadata.jsonl` with the parameters you would like to pass. There is a sample `metadata.jsonl` in this folder you can override. For example:

    ```json
    {"prompt" : "girl with cool pink hair, glossy eyes, smiling, detailed", "negative_prompt" : "realistic, bad fingers, bad hands, bad anatomy, missing fingers, mutant hands", "num_images" : 4, "scale" : 12.0, "model_id" : "Linaqruf/anything-v3.0"}
    {"prompt" : "a magical princess with golden hair, modern disney style", "negative_prompt" : "realistic, bad fingers, bad hands, bad anatomy, missing fingers, mutant hands", "num_images" : 4, "scale" : 7.5, "model_id" : "nitrosocke/mo-di-diffusion"}
    {"prompt" : "a beautiful perfect face girl in dgs illustration style, Anime fine details portrait of school girl in front of modern tokyo city landscape on the background deep bokeh, anime masterpiece, 8k, sharp high quality anime", "negative_prompt" : "realistic, bad fingers, bad hands, bad anatomy, missing fingers, mutant hands", "num_images" : 4, "scale" : 7.5, "model_id" : "DGSpitzer/Cyberpunk-Anime-Diffusion"}
    ```

1. Build container. Don't forget to change the `project_id` to yours.

    ```bash
    gcloud auth configure-docker
    docker build . -t gcr.io/{project_id}/stable-diffusion-batch-job:latest
    docker push gcr.io/{project_id}/stable-diffusion-batch-job:latest
    ```

1. Deploy.

    ```bash
    python gcp_deploy.py --project-id={project-id} --image-uri gcr.io/{project-id}/stable-diffusion-batch-job:latest --gcs-output-dir=gs://{project-id}-bucket/stable-diffusion-batch-job-results --hf-token={huggingface_token}
    ```

    The results will be in the `gcs-output-dir` folder with a `results.jsonl` specifying the parameters and the image uri. For example:

    ```json
    {"prompt" : "girl with cool pink hair, glossy eyes, smiling, detailed", "negative_prompt" : "realistic, bad fingers, bad hands, bad anatomy, missing fingers, mutant hands", "num_images" : 4, "scale" : 12.0, "model_id" : "Linaqruf/anything-v3.0", "image_uris" : ["gs://{project-id}-bucket/stable-diffusion-batch-job-results/4hgu43jd.png", "gs://{project-id}-bucket/stable-diffusion-batch-job-results/4hgu48jd.png"]}
    ```
