# CodeFormer
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8 && \
    apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda

RUN git clone https://github.com/sczhou/CodeFormer.git
WORKDIR "./CodeFormer"

RUN conda create -n codeformer python=3.8 -y

RUN . /root/.bashrc && \
    conda activate codeformer && \
    pip install -r requirements.txt && \
    python basicsr/setup.py develop && \
    python scripts/download_pretrained_models.py facelib && \
    python scripts/download_pretrained_models.py CodeFormer

RUN . /root/.bashrc && \
    conda activate codeformer && \
    pip install uvicorn fastapi

COPY app .
COPY entrypoint.sh .

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]