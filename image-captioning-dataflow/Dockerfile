FROM nvcr.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

WORKDIR "captioning"

COPY requirements.txt .

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl g++ python3.8-dev python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 \
    && curl https://bootstrap.pypa.io/get-pip.py | python

RUN apt-get update \
    && apt-get install -y git \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check

# RUN pip install gdown
RUN gdown 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
RUN git clone https://github.com/salesforce/BLIP

ENV PYTHONPATH "${PYTHONPATH}:/captioning/BLIP"
COPY load_blip_weights.py .
RUN python load_blip_weights.py

RUN apt-get update \
    && apt-get install wget

RUN wget https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz && tar -xf git-lfs-linux-amd64-v2.9.0.tar.gz && chmod 755 install.sh && ./install.sh && git clone https://huggingface.co/openai/clip-vit-base-patch32

RUN mkdir configs
RUN cp BLIP/configs/med_config.json configs/med_config.json
COPY . .

RUN apt-get install -y python3.8-venv

# Set the entrypoint to Apache Beam SDK worker launcher.
COPY --from=apache/beam_python3.8_sdk:2.45.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT ["/opt/apache/beam/boot"]
