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

RUN pip3 install --no-cache-dir diffusers transformers datasets torch google-cloud-storage uvicorn fastapi

ARG model_name
ENV MODEL_NAME=$model_name

WORKDIR "flan-t5"

COPY load_weights.py .
RUN python3 load_weights.py
COPY app .
COPY entrypoint.sh .

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]