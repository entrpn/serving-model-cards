# UI
<center>
    <image src="../images/ui.png" width="512px">
</center>

UI is a gradio interface that you can use to run stable diffusion batch prediction jobs with different model types without writing any code.

# Intro

Using Google cloud to create jobs has the following benefits:

- Managed infrastructure makes deployment fast an intuitive.
- Jobs in Vertex AI are ephemeral, meaning, resources are not long lasting. You only pay for the time you use and infrastructure is destroyed after job completion, making it more secure than downloading models in your own computer and only paying for the time you use resources.

# Setup

1. You'll first need to install the `gcloud` cli. You can find the instructions [here](https://cloud.google.com/sdk/docs/install).

1. Open a new shell window and install the requirements. **It is recommeneded to use a virtual python environment such as `pipenv`.**

    ```bash
    pip install -r requirements.txt
    ```

1. There are different ways to authenticate to the GCP account, such as creating a service account and setting `GOOGLE_APPLICATION_CREDENTIALS` or using `gcloud auth`. Here we'll use the latter.

    ```bash
    gcloud auth application-default login
    ```

1. Set your project id. Run the command below and change `<project_id>` to your project name.

    ```bash
    gcloud config set project <project_id>
    ```

1. Create a bucket to store all outputs. The name can be anything but needs to be unique across all of GCP.

    ```bash
    gsutil mb gs://<project_id>-serving-models-folder
    ```

1. If this is the first time you are using this project, then you'll need to enable the services to run this job.

    ```bash
    gcloud services enable compute.googleapis.com  containerregistry.googleapis.com aiplatform.googleapis.com cloudbuild.googleapis.com cloudfunctions.googleapis.com dataflow.googleapis.com
    ```

1. You might need to create a quota increase. View [here](https://cloud.google.com/docs/quota#about_increase_requests) to read about quota increases. In the quotas page, search for custom model training for the GPU type you like to use. For example, for a `T4` GPU, you can search for `Custom model training Nvidia T4 GPUs per region (default)` and ask for a quota increase for this GPU.

1. Start the gradio app.

    ```bash
    gradio ui.py
    ```
