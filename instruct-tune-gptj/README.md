# Instruction Tunining GPTJ-6B - WIP

**This repo is a work in progress and might not work correctly**

Instruction tune [GPTJ-6B](https://github.com/kingoflolz/mesh-transformer-jax) on a T4 GPU using the cleaned [Alpaca dataset](https://github.com/gururise/AlpacaDataCleaned) and [LoRA](https://arxiv.org/abs/2106.09685).

Features
- 8 bit attention linear layers quantization to load model on less than 10GB of GPU RAM. Finetuning still requires 15GB.
- 8 bit adam optimizer.
- LoRA implementation - trains 80% less parameters than full model finetuning. 
- Gradient checkpointing - reduce memory footprint.
- Gradient accumulation - imitates larger batch sizes on less memory.

## Setup

1. I ran this on a [Vertex AI Workbench Notebook](https://cloud.google.com/vertex-ai-workbench) with a T4 GPU, but any environment with 15GB of GPU ram works.
1. Cone repo if you haven't. Navigate to the `instruct-tune-gptj` folder.
1. Install python 3.9, or use conda.

  ```bash
  conda create -n py39 python=3.9
  conda activate py39
  ```

1. Install dependencies

  ```bash
  pip install transformers tensorboard accelerate datasets torch bitsandbytes git+https://github.com/huggingface/peft.git
  ```

1. Run the training script.

  ```bash
  python main_peft.py
  ```

  **Note that there is another file main_distributed.py where I was building the LoRA models from scratch, however midway I came across the [peft library](https://github.com/huggingface/peft) which makes it easier to create the LoRA layers. I left main_distributed.py for reference on how LoRA can be implemented from scratch.**

1. When training is finished, make an inference call. This loads the model and loads the LoRA weights.

  ```bash
  python infer.py
  ```

## Distributed setup.

Running on a single T4 works and is cool but is extremely slow. Here is the setup I used to train on 2 A100 GPUs using [Compute Engine](https://cloud.google.com/compute)

1. Create the machine. Don't forget to change `<your-project-id>` to yours. This script creates a compute instance behind a VPC, if you're not behind a VPC, remote `subnet=us-east4`.

  ```bash
  PROJECT_ID=<your-project-id>
  gcloud compute instances create pytroch-ultra-gpu    --project=$PROJECT_ID    --zone=us-east4-c    --machine-type=a2-ultragpu-2g   --network-interface=network-tier=PREMIUM,subnet=us-east4    --metadata=enable-oslogin=true    --maintenance-policy=TERMINATE    --provisioning-model=STANDARD    --scopes=https://www.googleapis.com/auth/cloud-platform    --accelerator=count=2,type=nvidia-a100-80gb    --tags=http-server,https-server    --create-disk=auto-delete=yes,boot=yes,device-name=pytroch-ultra-gpu,image=projects/ml-images/global/images/c2-deeplearning-pytorch-1-12-cu113-v20220928-debian-10,mode=rw,size=500,type=projects/$PROJECT_ID/zones/us-east4-c/diskTypes/pd-ssd    --no-shielded-secure-boot    --shielded-vtpm    --shielded-integrity-monitoring    --reservation-affinity=any
  ```

1. Configure ssh

  ```bash
  gcloud compute config-ssh
  ```

  We can now use vscode [remote-ssh](https://code.visualstudio.com/docs/remote/ssh-tutorial) to use the IDE directly on the machine. If you run into trouble finding the machine from the ssh extension, you can do the following at your own risk.

  ```bash
  sudo chmod 666 ~/.ssh/config
  ```

  Then try again.

1. Once you've sshed into the machine, install python 3.9 and dependencies.

  ```bash
  conda create -n py39 python=3.9
  conda activate py39
  pip install transformers tensorboard accelerate datasets torch bitsandbytes git+https://github.com/huggingface/peft.git
  ```

1. For the distributed setup, we'll use [accelerate](https://github.com/huggingface/accelerate). Run `accelerate config` and set parameters accordingly.

  If you like to use bf16, you'll need to modify `main_peft.py`:

  - line 91 : weight_dtype=torch.bfloat16
  - line 105: mixed_precision="bf16"
  - line 177: optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

1. Run training.

  ```bash
  accelerate launch main_peft.py
  ```