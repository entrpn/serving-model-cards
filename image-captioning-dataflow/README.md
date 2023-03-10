# Large Scale Image Captioning With Dataflow

## Intro

This repo prepares a [Dataflow](https://cloud.google.com/dataflow) job for large scale image captioning using BLIP and CLIP to generate and rank image captions.

Dataflow is a fully managed streaming analytics service that minimizes latency, processing time, and cost through autoscaling and batch processing based on [Apache beam](https://beam.apache.org/).

**The model licenses can be found [BLIP](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt) [CLIP](https://github.com/openai/CLIP/blob/main/LICENSE)**

Features:
- Creates image captions using BLIP.
- Ranks captions and uses the top one.
- Parallelize a job with lots of images across multiple workers.
- Saves image/caption pairs in jsonl format in HuggingFace's datasets format.
- Can run a small subsample in a local environment before deploying the dataflow job.

## Setup

1. Clone the repo if you haven't. Navigate to the `image-captioning-dataflow` folder.
1. Install python3.8 and dependencies

    ```bash
    conda create -n py38 python=3.8
    conda activate py38
    pip install -r requirements.txt
    ```
1. Install BLIP, download weights and save state dict. Change <your-blip-location> to the absolute path of the folder where BLIP was cloned.

    ```bash
    git clone https://github.com/salesforce/BLIP
    export PYTHONPATH=$PYTHONPATH:<your-blip-location>/BLIP
    gdown 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    python load_blip_weights.py
    ```

1. Copy BLIP configs

    ```bash
    mkdir configs
    cp BLIP/configs/med_config.json configs/
    ```

1. Download clip weights

    ```bash
    git lfs install
    git clone https://huggingface.co/openai/clip-vit-base-patch32
    ```

1. Create a dataset.txt file. You'll need to upload the images you want to caption to Google cloud storage. For example, I created a bucket `jfacevedo-demos-datasets` with a folder `me` and uploaded all my images to that folder. We will upload the output file, `dataset.txt`, into the same directory where our images are located. Test this with a few images at first before using the full image dataset.

    ```
    export BUCKET_ID="jfacevedo-demos-datasets"
    export PREFIX="me"
    python create_dataset_file.py
    gsutil cp dataset.txt gs://$BUCKET_ID/$PREFIX/
    ```

1. Next we'll need to move the weights to a local directory `/captioning`. The dataflow job won't actually use local files but this is needed to deploy the dataflow job and also we'll be testing this locally before deploying.

    ```bash
    chmod 755 clip-vit-base-patch32/
    sudo mkdir /captioning/
    sudo chmod 755 /captioning/
    sudo cp -r clip-vit-base-patch32/ /captioning/
    ```

1. Test the pipeline locally. This works without GPUs but takes longer.

    ```bash
    python pipeline.py --dataset-filename gs://$BUCKET_ID/$PREFIX/dataset.txt --output-filename gs://$BUCKET_ID/$PREFIX/metadata.jsonl
    ```

    If we look at the output file (or files), beam has sharded the output into multiple files which improves the performance of running this workload in parallel. You can join the files as follows.

    ```bash
    gsutil compose \
    gs://${BUCKET_ID}/$PREFIX/metadata* \
    gs://${BUCKET_ID}/$PREFIX/metadata.jsonl
    ```

1. We'll be using a custom container to run our Dataflow job. Build and push the container. Make sure you set <project-id> to yours

    ```bash
    export PROJECT_ID=<project-id>
    docker build . -t gcr.io/$PROJECT_ID/dataflow-captioning:latest
    docker push gcr.io/$PROJECT_ID/dataflow-captioning:latest
    ```

1. Run the dataflow job. First, you'll need a service account with `Dataflow Admin`, `Dataflow Worker` and `Compute Network User`. You can either use the default service account or create a new service account. Furthermore, if you are on the default network that comes with your project, you can ommit `--subnetwork`. If you're using the default service account, you can ommit `--service_account_email`. In the following snippet, I'm using a custom service account and a VPC network. If you're using the same `--temp_location` as the command below, make sure to create a bucket `$PROJECT_ID-bucket`.

    This job uses a T4 GPU.

    ```bash
    python pipeline.py \
    --dataset-filename gs://$BUCKET_ID/$PREFIX/dataset.txt \
    --output-filename gs://$BUCKET_ID/$PREFIX/metadata.jsonl \
    --runner=DataflowRunner \
    --project=$PROJECT_ID \
    --region=us-central1 \
    --job_name=captioning \
    --temp_location=gs://$PROJECT_ID-bucket/ \
    --sdk_container_image=gcr.io/$PROJECT_ID/dataflow-captioning:latest \
    --machine_type=n1-standard-16 \
    --experiment="worker_accelerator=type:nvidia-tesla-t4;count:1;install-nvidia-driver" \
    --experiment=use_runner_v2 \
    --disk_size_gb=200 \
    --subnetwork=https://www.googleapis.com/compute/v1/projects/$PROJECT_ID/regions/us-central1/subnetworks/jfacevedo-demo-subnet \
    --service_account_email=vertex-ai@$PROJECT_ID.iam.gserviceaccount.com \
    --sdk_location=container
    ```

1. You can view the job's progress through the Dataflow console.

    <center>
    <image src="./images/dataflow_job.png">
    </center>

    Don't forget to consolidate the sharded files into one to use for training , for example, with [Stable diffusion](../finetuning-stable-diffusion).