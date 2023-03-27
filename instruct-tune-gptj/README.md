# Instruction Tunining GPTJ-6B

**This repo is a work in progress and might not work correctly**

Instruction tune [GPTJ-6B](https://github.com/kingoflolz/mesh-transformer-jax) on a T4 GPU using the cleaned [Alpaca dataset](https://github.com/gururise/AlpacaDataCleaned) and [LoRA](https://arxiv.org/abs/2106.09685).

Features
- 8 bit attention linear layers quantization to load model on less than 10GB of GPU RAM. Finetuning still requires 15GB.
- 8 bit adam optimizer.
- LoRA implementation - trains 80% less parameters. 
- Gradient checkpointing - reduce memory footprint.
- Gradient accumulation - imitates larger batch sizes on less memory.

## Setup

1. I ran this on a [Vertex AI Workbench Notebook](https://cloud.google.com/vertex-ai-workbench) on a n1-standard-8 with a T4 GPU, but any environment with 15GB of GPU ram and 26GB "cpu" memory works. Run this for less than $2 using a [Spot VM](https://cloud.google.com/compute/docs/instances/spot) in Compute Engine. Don't forget to change `ZONE, PROJECT_ID, SUBNET` to yours.

  ```bash
  ZONE=us-east4-b
  PROJECT_ID=<my-project-id>
  SUBNET=us-east4
  gcloud compute instances create instance-2 --preemptible --project=$PROJECT_ID --zone=$ZONE --machine-type=n1-highmem-4 --network-interface=network-tier=PREMIUM,subnet=$SUBNET --maintenance-policy=TERMINATE --provisioning-model=SPOT --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=count=1,type=nvidia-tesla-t4 --tags=http-server,https-server --create-disk=auto-delete=yes,boot=yes,device-name=instance-1,image=projects/ml-images/global/images/c1-deeplearning-tf-2-6-cu110-v20230126-debian-10,mode=rw,size=200,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-standard --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
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

1. You'll need to first ssh to the machine via command line to accept the prompt that installs the NVIDIA drivers.

  ```bash
  gcloud compute ssh instance-2 --project$PROJECT_ID --zone=$ZONE
  ```

1. Now you can ssh via VS Code or continue through the command line and install dependencies.

  ```bash
  conda create -n py39 python=3.9
  conda activate py39
  pip install transformers tensorboard accelerate datasets torch bitsandbytes git+https://github.com/huggingface/peft.git
  ```

1. Clone repo and navigate to the `instruct-tune-gptj` folder.

1. Run the training script.

  ```bash
  python main_peft.py
  ```

The loss plot:

<center>
    <image src="./images/loss_graph.png">
</center>

  **Note that there is another file main_distributed.py where I was building the LoRA models from scratch, however midway I came across the [peft library](https://github.com/huggingface/peft) which makes it easier to create the LoRA layers. I left main_distributed.py for reference on how LoRA can be implemented from scratch. Do not use!**

1. When training is finished, make an inference call. This loads the model and loads the LoRA weights.

  ```bash
  python infer.py
  ```

  Let's look at some generations:

  ```text
  Instruction : How can I make friends?
  Response : The best way to make friends is to be friendly and approachable. Show interest in other people by asking them about their lives, interests, and hobbies. Make an effort to get to know people better by inviting them to social events and inviting them to join you in activities. Make an effort to get to know people better by asking them about their lives, interests, and hobbies. Show interest in other people by asking them about their lives, interests, and hobbies. Make an effort to get to know people better by inviting them to social events and inviting them to join you in activities. Make an effort to get to know people better by
  ```

  ```text
  Instruction : Generate an example of what a resume should like for an engineering professional
  Response : An example of a resume for an engineering professional should include the following information: name, contact information, education, professional experience, skills, and references. The education section should include a list of degrees, certifications, and professional experience. The professional experience section should include a list of projects, responsibilities, and accomplishments. The skills section should include a list of technical and non-technical skills. Finally, the references section should include a list of professional references.
  ```

  ```text
  Instruction : Tell me about lions.
  Response : Lions are one of the largest cats in the world. They are found in Africa, Asia, and parts of the Middle East. They are known for their powerful jaws, sharp claws, and large manes. They are also known for their ability to roar, which can be heard up to a mile away. They are also known for their ability to hunt and kill large prey such as antelopes, zebras, and wildebeests. They are also known for their ability to defend their territory and their cubs. They are also known for their ability to roar, which can be heard up to a mile away
  ```

  ```text
  Instruction : What are alcapas? How are alpacas different from llamas?
  Response : Alpacas and llamas are both members of the camelid family. Alpacas are smaller than llamas and have shorter legs. Alpacas are native to South America, while llamas are native to the Andes Mountains of South America. Alpacas and llamas are similar in appearance, but alpacas are smaller and have shorter legs than llamas. Alpacas and llamas are both used as pack animals, but alpacas are smaller and have shorter legs than llamas. Alpacas and llamas are both used for milk production, but alpacas are smaller and have shorter legs than
  ```

  None of these questions are in the training dataset afaik.

  This model exibits biases and hallucinations, don't use in a production environment.

## Distributed setup.

**Still WIP, and doesn't work correctly atm**

Here is the setup I used to train on 2 A100 GPUs using [Compute Engine](https://cloud.google.com/compute)

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