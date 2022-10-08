# Deploying a Model to Vertex AI Endpoints

## Let's get started!

This guide will show you how to build a serving container for any of the models in this repo and deploy it on Vertex AI endpoints, a managed serving infrastructure for serving machine learning models.

Click the **Start** button to move to the next step.

## Set up project dependencies

### Set your Project id

Run the following command in Cloud Shell to confirm that the gcloud command knows about your project:

```shell
gcloud config list project
```

If it is not, you can set it with this command:

```shell
gcloud config set project <PROJECT_ID>
```

Create a variable with your project_id. We'll be using this in later steps.

```shell
project_id=`gcloud config list project | awk '{ print $3 }' | tr -d '\n'`
```

### Enable APIs

In later steps, you'll see where these services are needed (and why), but for now, run this command to give your project access to Container Registry and Vertex AI services:

```shell
gcloud services enable artifactregistry.googleapis.com  \
                        cloudbuild.googleapis.com \
                        aiplatform.googleapis.com 
```

### Set Permissions

We'll be creating a service account that has all the permissions required for the next steps.

```shell
gcloud iam service-accounts create model-deploy \
    --description="My model building account" \
    --display-name="model-deploy"
SVC_ACCOUNT=model-deploy@${project_id}.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT --member serviceAccount:$SVC_ACCOUNT --role roles/storage.objectAdmin
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT --member serviceAccount:$SVC_ACCOUNT --role roles/aiplatform.user
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT --member serviceAccount:$SVC_ACCOUNT --role roles/iam.serviceAccountUser
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT --member serviceAccount:$SVC_ACCOUNT --role roles/cloudbuild.builds.editor
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT --member serviceAccount:$SVC_ACCOUNT --role roles/artifactregistry.admin
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT --member serviceAccount:$SVC_ACCOUNT --role roles/storage.admin
```

Now let's create a service account key and authenticate as this service account. This will grant us permissions for the next steps.

```shell
gcloud iam service-accounts keys create sa_key.json \
    --iam-account=model-deploy@${project_id}.iam.gserviceaccount.com
gcloud auth activate-service-account model-deploy@${project_id}.iam.gserviceaccount.com --key-file=./sa_key.json --project=${project_id}
```

## Build the serving container

### Choose a model

Each folder in this repository contains a Dockerfile that packages the model into a serving container using fastapi. We'll be using the `stable-diffusion` model. If you want to use another model, change the `model` variable below to the folder which you want to use.

We will also set a variable `model` which will be used to create the container.

```shell
model=stable-diffusion
cd ${model}
```

### Create an Artifact Registry Repository

Artifact Registry manages container images and language packages. Let's create a repository for our serving model container. This example uses the `us-central1` location, but feel free to change it.

```shell
location=us-central1
gcloud artifacts repositories create ${model}-repo --repository-format=docker \
    --location=${location} --description="${model} repository"
```

### Build the container with Cloud Run

Now let's build the container using cloud build.

```shell
gcloud builds submit --tag ${location}-docker.pkg.dev/${project_id}/${model}-repo/${model}:latest --timeout 3600 --machine-type=N1_HIGHCPU_32 --disk-size 500
```

This operations takes about 15 minutes to complete.

## Deploy to Vertex AI

Now that the container has been uploaded to the container registry, we can deploy it to Vertex AI. First let's see what options are available in the deployment script.

```shell
python gcp_deploy.py --help
```

```shell
usage: gcp_deploy.py [-h] [--min-replica-count MIN_REPLICA_COUNT] [--machine-type MACHINE_TYPE] [--max-replica-count MAX_REPLICA_COUNT] [--gpu-type GPU_TYPE]
                     [--accelerator-count ACCELERATOR_COUNT] [--region REGION] [--model-name MODEL_NAME] [--endpoint-name ENDPOINT_NAME] [--endpoint-deployed-name ENDPOINT_DEPLOYED_NAME]
                     --image-uri IMAGE_URI [--accelerator-duty-cycle ACCELERATOR_DUTY_CYCLE] [--cpu-duty-cycle CPU_DUTY_CYCLE]

optional arguments:
  -h, --help            show this help message and exit
  --min-replica-count MIN_REPLICA_COUNT
                        Minimum number of replicas
  --machine-type MACHINE_TYPE
                        Machine type
  --max-replica-count MAX_REPLICA_COUNT
                        Maximum number of replicas
  --gpu-type GPU_TYPE   GPU type
  --accelerator-count ACCELERATOR_COUNT
                        GPU count
  --region REGION       gcp region
  --model-name MODEL_NAME
                        name of model
  --endpoint-name ENDPOINT_NAME
                        Name of endpoint
  --endpoint-deployed-name ENDPOINT_DEPLOYED_NAME
                        Endpoint deployed name
  --image-uri IMAGE_URI
                        name of image in gcr. Ex: gcr.io/project-name/stable-diffusion:latest
  --accelerator-duty-cycle ACCELERATOR_DUTY_CYCLE
                        Autoscaling for GPUs. 20 forces the endpoint to autoscale immediately if --min-replica-count > 1
  --cpu-duty-cycle CPU_DUTY_CYCLE
                        Autoscaling for CPUs. 20 forces the endpoint to autoscale immediately if --min-replica-count > 1
```

Vertex AI endpoints can autoscale when nodes are running at a set capacity (60% by default). We can also take advangate of GPUs to speed up our workloads. We will be deploying a single replica with a `T4` GPU.

First let's make sure our service account is used to create the endpoint. Verify the path to the `sa_key.json` is correct before proceding.

```shell
export GOOGLE_APPLICATION_CREDENTIALS=../sa_key.json
```

Now let's use the deploy script to deploy the model to the endpoint.

```shell
pip install google-cloud-aiplatform
python gcp_deploy.py --image-uri ${location}-docker.pkg.dev/${project_id}/${model}-repo/${model}:latest --gpu-type NVIDIA_TESLA_T4 --accelerator-count 1 --min-replica-count 1 --max-replica-count 1 --region ${location}
```

This script takes some time to finish as the serving infrastructure is being created and the container is being deployed. Once this is finished, you'll receive a log with the endpoint name. Make note of it as you'll use this in the final step. It should look like:

```text
Deploy Endpoint model backing LRO: projects/611544971877/locations/us-central1/endpoints/3370654051117083392/operations/1401138463849152512
```

## Make predictions

We can make a prediction with the following script. Change the `endpoint-name` to the one that was printed for you in the previous step.

```shell
python generate_request_vertex.py --endpoint-name projects/611544971877/locations/us-central1/endpoints/3370654051117083392/operations/1401138463849152512
```

Once this script completes, a `response.json` will be generated inside the current folder. 