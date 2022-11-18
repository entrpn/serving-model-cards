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

RUN git clone https://github.com/CompVis/stable-diffusion.git
WORKDIR "/stable-diffusion"
RUN conda env create -f environment.yaml

RUN mkdir models/ldm/stable-diffusion-v1
RUN apt-get update && apt-get install -y curl 

ARG HF_TOKEN
COPY download_weights.sh .
COPY download_weights.py .
RUN ./download_weights.sh $HF_TOKEN
RUN cp models--CompVis--stable-diffusion-v-1-4-original/snapshots/f0bb45b49990512c454cf2c5670b0952ef2f9c71/sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
RUN ls -ll models/ldm/stable-diffusion-v1/
RUN rm -rf models--CompVis--stable-diffusion-v-1-4-original
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

RUN . /root/.bashrc && \
    conda activate ldm && \
    pip install uvicorn fastapi

COPY app/load_weights.py .
COPY load_weights.sh .
RUN ["./load_weights.sh"]

COPY app .
COPY entrypoint.sh .

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]