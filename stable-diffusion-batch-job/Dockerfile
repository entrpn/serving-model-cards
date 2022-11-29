FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8 && \
    apt-get -y install python3-pip

RUN pip3 install --no-cache-dir diffusers transformers datasets google-cloud-storage torch torchvision

WORKDIR 'stable-diffusion-batch-job'

COPY . .

CMD ["python3", "main.py"]