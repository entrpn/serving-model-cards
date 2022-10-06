FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8

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
RUN curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > models/ldm/stable-diffusion-v1/model.ckpt

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