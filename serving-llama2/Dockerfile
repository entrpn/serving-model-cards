FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 && \
    apt-get -y install python3-pip

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install diffusers \
    transformers \
    datasets \
    google-cloud-storage \
    torch \
    torchvision \
    opencv-python \
    xformers \
    accelerate \
    uvicorn \
    fastapi \
    bitsandbytes \
    scipy

ARG model_name
ENV MODEL_NAME=$model_name

ARG hf_auth
ENV HF_AUTH=$hf_auth

ARG max_new_tokens
ENV MAX_NEW_TOKENS=$max_new_tokens

RUN echo "$MAX_NEW_TOKENS, $HF_AUTH, $MODEL_NAME"

WORKDIR = "/llama2"

COPY load_weights.py . 
RUN python3 load_weights.py

COPY main.py .
COPY entrypoint.sh .

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]