FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8 && \
    apt-get -y install python3-pip

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install git+https://github.com/huggingface/diffusers.git
RUN pip3 install transformers datasets google-cloud-storage torch==1.13.1 torchvision opencv-python xformers accelerate uvicorn fastapi

ARG use_xformers=0
ENV USE_XFORMERS=$use_xformers

ARG use_lora=0
ENV USE_LORA=$use_lora

RUN echo "$MODEL_REVISION, $USE_XFORMERS, $MODEL_NAME, $USE_LORA"

WORKDIR = "training_dreambooth"

COPY . .
ENTRYPOINT ["./main.sh"]