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

RUN pip3 install diffusers transformers datasets google-cloud-storage torch==1.13.1 torchvision opencv-python xformers accelerate uvicorn fastapi

ARG model_name
ENV MODEL_NAME=$model_name

ARG use_xformers=0
ENV USE_XFORMERS=$use_xformers

ARG model_revision=main
ENV MODEL_REVISION=$model_revision

RUN echo "$MODEL_REVISION, $USE_XFORMERS, $MODEL_NAME"

WORKDIR = "/diffusion"

COPY diffusion_utils.py .
COPY load_weights.py . 
RUN python3 load_weights.py

COPY main.py .
COPY entrypoint.sh .

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]