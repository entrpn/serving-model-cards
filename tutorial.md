# Deploying a Model to Vertex AI Endpoints

## Let's get started!


This guide will show you how to build your own interactive tutorial (like this one). It'll also walk you through generating a button that users can use to launch your finished tutorial.

This guide will show you how to build a serving container for any of the models in this repo and deploy it on Vertex AI endpoints, a managed serving infrastructure for serving machine learning models.

**Time to complete**: About 10 minutes

Click the **Start** button to move to the next step.

## Enable APIs

In later steps, you'll see where these services are needed (and why), but for now, run this command to give your project access to Container Registry and Vertex AI services:

```shell
gcloud services enable containerregistry.googleapis.com  \
                       aiplatform.googleapis.com 
```

## Set your Project id

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
project_id=`gcloud config list project | awk '{ print $3 }'`
```

## Build the serving container

### Choose a model

Each folder in this repository contains a Dockerfile that packages the model into a serving container using fastapi. You'll need to `cd` into the folder which you would like to build. In this tutorial, we'll be using the `stable-diffusion` model. 

We will also set a variable `model` which will be used to create the container.

```shell
cd stable-diffusion
model=stable-diffusion
```

### Build the container

Now let's build the container using Docker.

```shell
docker build . -t gcr.io/{project_id}/{model}:latest
```



