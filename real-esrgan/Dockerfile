FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8 && \
    apt install -y python3-pip

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#cv2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN git clone https://github.com/xinntao/Real-ESRGAN.git
WORKDIR "/Real-ESRGAN"
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models

RUN pip install basicsr facexlib gfpgan

RUN pip install -r requirements.txt
RUN python3 setup.py develop
RUN pip install uvicorn fastapi opencv-python

COPY app .
COPY entrypoint.sh .
COPY load_weights.py .
RUN python3 load_weights.py

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]