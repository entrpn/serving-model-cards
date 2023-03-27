# serving-model-cards

A collection of OSS models that are containerized and ready to be served in GCP's [Vertex AI](https://cloud.google.com/vertex-ai) for easy deployment. 

By using Vertex AI endpoints, users don't need to manage complex infrastructure to serve models.
<br/>

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https://github.com/entrpn/serving-model-cards&cloudshell_tutorial=tutorial.md)

<center><b>Make sure to read the permissable license for each model before use!!</b></center>

| <a href="./serving-stable-diffusion"><p><center>Stable Diffusion<img src="./images/sd.png"></img><br>Generate images from a text prompt</br></center></p></a>| <a href="./serving-flant5"><p><center>FLAN-T5<img src="./images/serving_flant5.png"></img><br>Generate Text</br></center></p></a>  |<a href="./real-esrgan"><p><center>Real-ESRGAN<img src="./images/real_esrgan.jpeg"></img><br>Upscale images</br></center></p></a>
| ---- | ---- | ---- |
<a href="./blip"><p><center>BLIP<img src="./images/blip.jpeg"></img><br>Image captioning</br></center></p></a> | <a href="./bart"><p><center>BART<img src="./images/summarization.png"></img><br>Summarize Text</br></center></p></a> | <a href="./instruct-tune-gptj"><p><center>Instruct GPTJ<img src="./images/gptj-instruct-card.png"></img><br>Instruction tune GPTJ</br></center></p></a> | ---- | ---- | ---- |

# training-model-cards

A collection of OSS models that are containerized and ready to be trained in GCP's [Vertex AI](https://cloud.google.com/vertex-ai) for easy deployment.

| <a href="./training-dreambooth"><p><center>Train Dreambooth<img src="./images/dreambooth.png"></img><br>Personalize stable diffusion</br></center></p></a> | <a href="./finetuning-stable-diffusion"><p><center>Finetune Stable Diffusion<img src="./images/finetune_sd.png"></img><br>Finetune stable diffusion</br></center></p></a> | <a href="./training-image-segmentation"><p><center>Image Segmentation<img src="./images/segmentation.png"></img><br>Create masks and inpaint</br></center></p></a>
|-|-|-|

# misc

A collection of different jobs and features.

| <a href="./stable-diffusion-batch-job"><p><center>Stable Diffusion Batch Job</br><img src="./images/sd_batch_job.png"></img><br>Create a batch job with different styles </br>of stable diffusion</br></center></p></a> | <a href="./ui"><p><center>UI for Stable Diffusion Batch Job</br><img src="./images/ui.png" width='512px'></img><br>Create a batch job with different styles </br>of stable diffusion</br></center></p></a> | <a href="./image-captioning-dataflow"><p><center>Large Scale Image Captioning with Dataflow</br><img src="./images/blip.jpeg" width='512px'></img><br>Caption millions of images at scale </br>using Dataflow</br></center></p></a>
|-|-|-|

# How to guides

- Stable diffusion how to guides
    - [Convert original sd checkpoint to diffusers](./stable-diffusion-how-tos/convert-sd-checkpoint-to-diffusers)
    - [Merge checkpoints](./stable-diffusion-how-tos/checkpoint-merging)