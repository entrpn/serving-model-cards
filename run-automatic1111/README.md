# Run Automatic1111

## Intro

This repo has instructions on how to run Automatic1111's [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) in GCP using a Compute Engine instance and T4 GPUs. 

With this method, you can deploy your own Automatic1111 UI in the cloud and use it as needed and only pay for the time of use. This is a great alternative for those that do not have a GPU or don't want to deal with the difficulty of installing the right CUDA drivers in your own computer.

## Setup

1. Clone repo if you haven't, Navigate to the `run-automatic1111` folder.
1. Open a shell in your computer to create a VM. Substitute `your-project-id` to yours.

    ```bash
    ZONE=us-central1-a
    PROJECT_ID=your-project-id
    SUBNETWORK=default
    gcloud compute instances create automatic1111  --project=$PROJECT_ID --zone=$ZONE --machine-type=n1-highmem-4 --network-interface=network-tier=PREMIUM,subnet=$SUBNETWORK --maintenance-policy=TERMINATE --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=count=1,type=nvidia-tesla-t4 --tags=http-server,https-server --create-disk=auto-delete=yes,boot=yes,device-name=instance-1,image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20230501-debian-10,mode=rw,size=500,type=projects/ldm-project/zones/$ZONE/diskTypes/pd-standard --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
    ```

1. Configure ssh. You will need to install the [gcloud cli](https://cloud.google.com/sdk/docs/install).

    ```bash
    sudo gcloud compute config-ssh
    sudo chmod 666 ~/.ssh/config # Optional, if you are using VSCode remote extension, this might be needed
    ```

1. ssh into the machine and install Automatic1111. Highly recommend you do remote ssh via VSCode as it will automatically forward the internal port from the VM to your computer.

    ```bash
    rm -rf ~/.ssh/config # If you are having trouble with the command below
    gcloud compute ssh --zone $ZONE "automatic1111" --project $PROJECT_ID
    ```
    The first time you ssh into the machine, it will ask you to install the CUDA drivers, make sure to select `Y` and wait for them to install.

    Next we'll set up a conda environment to run the installation. Run this while being sshed into the VM.

    ```bash
    sudo apt install wget git python3 python3-venv
    conda create -n automatic python=3.10 -y
    conda activate automatic
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
    cd stable-diffusion-webui/
    ./webui.sh
    ```

    Optionally to not deal with port forwading, you can run the last command as follows. Change the user and password to something unique or anyone with the link will be able to access your UI.

    ```bash
    ./webui.sh --share --gradio-auth user:password
    ```

    If you ran this last command, you're done! Once the UI is running, you can click on the gradio space link and access the UI with a username and password you selected. Look in the readme below on how to download models, LoRAs, etc.
    
    **Note that using gradio space links instead of port forwarding will not allow you to install extensions**.

1. The first time, this process will take a while as the UI is installing dependencies and models. Now do a port foward to your local machine.

    ```bash
    gcloud compute ssh automatic1111 --zone $ZONE --project $PROJECT_ID -- -L 8080:localhost:7860
    ```

    Now you can access the UI from your local machine, just open a browser and go to `127.0.0.1:8080`. 

1. Alternatively you can open a Google cloud shell from the Google Cloud console and run the same port forwarding command and then click on web preview, but I haven't fully tested this.

## Installing models, LoRAs, etc

In order to install new models, LoRAs, etc, you'll need to ssh into the machine and use the `wget` command. Let's look at an example.

1. ssh into the machine

    ```bash
    gcloud compute ssh --zone $ZONE "automatic1111" --project $PROJECT_ID
    ```

1. Grab a model from civitai. In this example, I'll use [dreamshaper](https://civitai.com/models/4384/dreamshaper). Right click the download link and `Copy Link address`. Then go to the models folder and download it.

    ```bash
    cd stable-diffusion-webui/models/Stable-diffusion/
    wget https://civitai.com/api/download/models/43888 -O dreamshaperV5.safetensors
    ```

    Now refresh the models via the UI and it should be there. The same applies for LoRAs, Textual Inversion, etc.

## Installing extensions

This is only supported if you are doing port forwarding to your local machine. If you're using `--shared` when deploying the webui, it will not work. Installing extensions is the same as if you're running locally.